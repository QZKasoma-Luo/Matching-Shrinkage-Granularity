"""
The a-priori granularity diagnostic and its validation -- the data behind the
paper's Section `sec:diagnostic`.

Two candidate rules, both computable from the released counts z and the public
sigma^2 alone (instantiating the Method section's oracle-risk comparison):

Estimated oracle risks (protected levels are unshrunk, matching Algorithm 1,
so they contribute their full noise energy and cost no coefficient):
    S_hat_G  = max(0, ||z_G||^2 - n_G sigma^2)          for a group G
    R_hat_G  = S_hat_G n_G sigma^2 / (S_hat_G + n_G sigma^2)
    R_scalar = R_hat_V                          (1 estimated coefficient)
    R_level  = sum_{n_l > 3} R_hat_l  +  sum_{n_l <= 3} n_l sigma^2
                                                (one coefficient per big level)
    R_node   = sum_v R_hat_v                    (n estimated coefficients)

RULE A (two-way, multiplicative):  NODE if R_node < tau * R_level else LEVEL.
RULE B (three-way, penalized):     argmin_P  R_P + kappa * sigma^2 * (#coeffs),
    the additive penalty matching the O(sigma^2)-per-estimated-coefficient
    excess of the plug-in (Lemma `lem:excess`).

No-gain flag: the best penalized risk is within 5% of the total noise floor
n sigma^2 -- i.e. NO granularity is predicted to shrink meaningfully.

Validation, against the already-computed experiment results:
  - synthetic: all 504 configs (archived DP draws; majority over 50 trials).
    Ground truth = argmin MSE among {Scalar, Ours(level), PerNode(node)}.
    A config is DECISIVE if best beats runner-up by > 2% of Hay's MSE.
    Balanced configs tune tau / kappa; unbalanced configs are held out.
  - DDHC-A: the five selected groups (exact run_ddhca draws), held out.
Metrics: accuracy on decisive configs; mean/median regret
MSE(chosen)/MSE(best) - 1 over all configs.

Output: ../results/diagnostic_validation.json
Run:    python diagnostic_check.py                 (WPES setting: two-pass results, DDHC-A held out)
        python diagnostic_check.py --exact-ls --p1  (CCECE setting: results_exactls.json, PL94 P1
                                                    sparse columns as the census held-out set;
                                                    writes diagnostic_validation_exactls.json)
"""
from __future__ import annotations
import json
import os
import time
import numpy as np

from hierarchy_data_generator import (create_balanced_tree, create_unbalanced_tree,
                                      UNBALANCED_SPECS)
from laplace_noise_injection import add_laplace_noise, noise_variance, EPSILONS, NUM_TRIALS
from T01001_census_loader import (SELECTED_POPGROUPS, CENSUS_DDHCA_HIER_DIR,
                                  CENSUS_DDHCA_RESULTS_DIR,
                                  load_hierarchy_archive)

RESULTS_JSON = "../results/synthetic_data_results/results.json"
RESULTS_JSON_LS = "../results/synthetic_data_results/results_exactls.json"
DP_BAL_DIR = "../synthetic_data/dp_balanced_tree"
DP_UNBAL_DIR = "../synthetic_data/dp_unbalanced_tree"
OUT_FILE = "../results/diagnostic_validation.json"
OUT_FILE_LS = "../results/diagnostic_validation_exactls.json"
P1_RESULTS = "../results/P1_sparse_results/p1_sparse_results.json"

PROTECT_MAX = 3          # matches ROOT_PROTECT_MAX in shrinkage.py
DECISIVE_FRAC = 0.02     # best must beat runner-up by >2% of Hay's MSE
NOGAIN_FRAC = 0.05       # best predicted risk within 5% of the noise floor
TAU_GRID = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
KAPPA_GRID = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

GRAN_KEYS = {"Scalar": "SCALAR", "Ours": "LEVEL", "PerNode": "NODE"}
GRANS = ["SCALAR", "LEVEL", "NODE"]   # coarse -> fine (tie-break prefers coarse)


def _rhat(ssq: float, n: int, sigma2: float) -> float:
    """Estimated oracle risk of one shrunk group with observed energy ssq."""
    s_hat = max(0.0, ssq - n * sigma2)
    return (s_hat * n * sigma2 / (s_hat + n * sigma2)) if s_hat > 0 else 0.0


def rule_stats(z: np.ndarray, h, sigma2: float) -> dict:
    """Estimated risks and coefficient counts for the three granularities."""
    n = h.num_nodes
    r_scalar = _rhat(float(np.dot(z, z)), n, sigma2)
    r_level, k_level = 0.0, 0
    for ell in range(h.depth):
        mask = h.level_mask(ell)
        n_l = int(mask.sum())
        if n_l <= PROTECT_MAX:
            r_level += n_l * sigma2          # unshrunk level keeps its noise
        else:
            r_level += _rhat(float(np.dot(z[mask], z[mask])), n_l, sigma2)
            k_level += 1
    s_v = np.maximum(0.0, z * z - sigma2)
    r_node = float(np.sum(s_v * sigma2 / (s_v + sigma2)))
    return {"R": {"SCALAR": r_scalar, "LEVEL": r_level, "NODE": r_node},
            "K": {"SCALAR": 1, "LEVEL": k_level, "NODE": n},
            "noise_floor": n * sigma2}


def decide_tau(stats: list, tau: float) -> str:
    votes = sum(1 for s in stats if s["R"]["NODE"] < tau * s["R"]["LEVEL"])
    return "NODE" if votes > len(stats) / 2 else "LEVEL"


def decide_kappa(stats: list, kappa: float) -> str:
    votes = {g: 0 for g in GRANS}
    for s in stats:
        sigma2 = s["noise_floor"] / s["K"]["NODE"]      # n*sigma2 / n
        scores = {g: s["R"][g] + kappa * sigma2 * s["K"][g] for g in GRANS}
        votes[min(GRANS, key=lambda g: scores[g])] += 1
    return max(GRANS, key=lambda g: (votes[g], -GRANS.index(g)))


def nogain_pred(stats: list, kappa: float) -> bool:
    flags = []
    for s in stats:
        sigma2 = s["noise_floor"] / s["K"]["NODE"]
        best = min(s["R"][g] + kappa * sigma2 * s["K"][g] for g in GRANS)
        flags.append(best >= (1.0 - NOGAIN_FRAC) * s["noise_floor"])
    return sum(flags) > len(flags) / 2


def config_record(spec_kind, h, Z, sigma2, methods_block):
    stats = [rule_stats(np.asarray(z, dtype=float), h, sigma2) for z in Z]
    hay = methods_block["Hay"]["mse_mean"]
    cand = {GRAN_KEYS[k]: methods_block[k]["mse_mean"] for k in GRAN_KEYS}
    best = min(cand, key=cand.get)
    runner = min(v for k, v in cand.items() if k != best)
    return {
        "stats": stats, "hay": hay, "mse": cand, "best": best,
        "decisive": (runner - cand[best]) > DECISIVE_FRAC * hay,
        "nogain_true": (hay - cand[best]) < DECISIVE_FRAC * hay,
        "spec_kind": spec_kind,
    }


def collect_synthetic(results_json: str = RESULTS_JSON) -> list:
    with open(results_json) as f:
        payload = json.load(f)
    unbal_edges = dict(UNBALANCED_SPECS)
    trees, records = {}, []
    for r in payload["results"]:
        spec, key = r["spec"], r["spec_name"]
        if key not in trees:
            if spec["kind"] == "balanced":
                trees[key] = create_balanced_tree(depth=spec["depth"],
                                                  branching_factor=spec["branching_factor"])
            else:
                trees[key] = create_unbalanced_tree(unbal_edges[key], root=0)
        e = str(r["epsilon"]).replace(".", "p")
        fn = f"dp_{key[len('tree_'):]}_{r['signal_type']}_mean{r['leaf_mean']}_eps{e}.json"
        dp_dir = DP_BAL_DIR if spec["kind"] == "balanced" else DP_UNBAL_DIR
        with open(os.path.join(dp_dir, fn)) as f:
            rec = json.load(f)
        records.append(config_record(spec["kind"], trees[key],
                                     np.asarray(rec["z"]), rec["sigma2"],
                                     r["methods"]))
    return records


def collect_ddhca() -> list:
    records = []
    for pg in SELECTED_POPGROUPS:
        arch = os.path.join(CENSUS_DDHCA_HIER_DIR,
                            f"T01001_census_county_2020_pop{pg}.json")
        h, x, _ = load_hierarchy_archive(arch)
        with open(os.path.join(CENSUS_DDHCA_RESULTS_DIR,
                               f"ddhca_results_pop{pg}.json")) as f:
            per_eps = {r["epsilon"]: r["methods"] for r in json.load(f)["results"]}
        for ei, eps in enumerate(EPSILONS):
            s2 = noise_variance(h, eps)
            Z = [add_laplace_noise(x, h, eps,
                                   np.random.default_rng(1_300_000 + ei * 10_000 + t))
                 for t in range(NUM_TRIALS)]
            records.append(config_record("census_ddhca", h, np.asarray(Z), s2,
                                         per_eps[eps]))
    return records


def collect_p1_sparse(results_json: str = P1_RESULTS) -> list:
    """Held-out real configs from the PL94 P1 sparse-column experiment
    (P1_sparse_columns.py): regenerates its exact draws from its seed schedule
    and reads the per-method MSEs from its results file. Method names there are
    Scalar / Level / Node; mapped onto the registry names used here."""
    from P1_sparse_columns import SEED_BASE, HIER_DIR   # lazy: avoids a circular import
    with open(results_json) as f:
        payload = json.load(f)
    name_map = {"Hay": "Hay", "Scalar": "Scalar", "Level": "Ours", "Node": "PerNode"}
    records = []
    trees = {}
    for r in payload["results"]:
        col = r["column"]
        if col not in trees:
            h, x, _ = load_hierarchy_archive(
                os.path.join(HIER_DIR, f"P1_census_county_2020_{col}.json"))
            trees[col] = (h, x)
        h, x = trees[col]
        eps = r["epsilon"]
        ei = EPSILONS.index(eps)
        colnum = int(col[3:6])
        s2 = noise_variance(h, eps)
        Z = [add_laplace_noise(x, h, eps,
                               np.random.default_rng(SEED_BASE + colnum * 100_000 + ei * 10_000 + t))
             for t in range(r["num_trials"])]
        block = {name_map[k]: v for k, v in r["methods"].items() if k in name_map}
        records.append(config_record("census_p1", h, np.asarray(Z), s2, block))
    return records


def evaluate(records: list, decider) -> dict:
    n_dec = n_correct = 0
    regrets = []
    for r in records:
        choice = decider(r["stats"])
        if r["decisive"]:
            n_dec += 1
            n_correct += (choice == r["best"])
        best = min(r["mse"].values())
        regrets.append(r["mse"][choice] / best - 1.0 if best > 0 else 0.0)
    return {"n_configs": len(records), "n_decisive": n_dec,
            "accuracy_decisive": (n_correct / n_dec) if n_dec else None,
            "mean_regret_pct": 100 * float(np.mean(regrets)),
            "median_regret_pct": 100 * float(np.median(regrets))}


def evaluate_single_draw(records: list, kappa: float) -> dict:
    """Deployment-relevant variant: accuracy of the RULE-B decision computed
    from ONE draw (averaged over draws), rather than the 50-draw majority.
    A release in practice is a single z."""
    fracs = []
    for r in records:
        if not r["decisive"]:
            continue
        hits = 0
        for s in r["stats"]:
            sigma2 = s["noise_floor"] / s["K"]["NODE"]
            scores = {g: s["R"][g] + kappa * sigma2 * s["K"][g] for g in GRANS}
            hits += (min(GRANS, key=lambda g: scores[g]) == r["best"])
        fracs.append(hits / len(r["stats"]))
    return {"n_decisive": len(fracs),
            "accuracy_single_draw": float(np.mean(fracs)) if fracs else None}


def nogain_report(records: list, kappa: float) -> dict:
    pred = np.array([nogain_pred(r["stats"], kappa) for r in records])
    true = np.array([r["nogain_true"] for r in records])
    return {"n": len(records),
            "n_nogain_true": int(true.sum()), "n_nogain_pred": int(pred.sum()),
            "precision": float((pred & true).sum() / pred.sum()) if pred.sum() else None,
            "recall": float((pred & true).sum() / true.sum()) if true.sum() else None}


if __name__ == "__main__":
    import sys
    EXACT_LS = "--exact-ls" in sys.argv
    USE_P1 = "--p1" in sys.argv
    results_json = RESULTS_JSON_LS if EXACT_LS else RESULTS_JSON
    out_file = OUT_FILE_LS if EXACT_LS else OUT_FILE
    census_name = "P1 sparse columns" if USE_P1 else "DDHC-A"
    t0 = time.time()
    print(f"synthetic results: {results_json} | census held-out set: {census_name}")
    print("collecting synthetic configs (reads all DP archives)...")
    syn = collect_synthetic(results_json)
    bal = [r for r in syn if r["spec_kind"] == "balanced"]
    unb = [r for r in syn if r["spec_kind"] == "unbalanced"]
    print(f"  [{time.time()-t0:.1f}s] {len(bal)} balanced + {len(unb)} unbalanced")
    print(f"collecting {census_name} configs (regenerates the experiment draws)...")
    dd = collect_p1_sparse() if USE_P1 else collect_ddhca()
    print(f"  [{time.time()-t0:.1f}s] {len(dd)} census configs")

    # where do the three granularities actually win?
    for name, group in [("balanced", bal), ("unbalanced", unb), ("census", dd)]:
        wins = {g: sum(1 for r in group if r["decisive"] and r["best"] == g)
                for g in GRANS}
        print(f"  decisive winners in {name:<10}: {wins}")

    print(f"\nRULE A (two-way, tau)          bal acc | unbal acc | census acc")
    resA = {}
    for tau in TAU_GRID:
        d = lambda s, t=tau: decide_tau(s, t)
        ea, eu, ed = evaluate(bal, d), evaluate(unb, d), evaluate(dd, d)
        resA[tau] = (ea, eu, ed)
        print(f"  tau={tau:<4}                     {ea['accuracy_decisive']:>7.3f} | "
              f"{eu['accuracy_decisive']:>9.3f} | {ed['accuracy_decisive']:>10.3f}")
    tauA = max(TAU_GRID, key=lambda t: (resA[t][0]["accuracy_decisive"],
                                        -resA[t][0]["mean_regret_pct"]))

    print(f"\nRULE B (three-way, kappa)      bal acc | unbal acc | census acc")
    resB = {}
    for k in KAPPA_GRID:
        d = lambda s, k=k: decide_kappa(s, k)
        ea, eu, ed = evaluate(bal, d), evaluate(unb, d), evaluate(dd, d)
        resB[k] = (ea, eu, ed)
        print(f"  kappa={k:<4}                   {ea['accuracy_decisive']:>7.3f} | "
              f"{eu['accuracy_decisive']:>9.3f} | {ed['accuracy_decisive']:>10.3f}")
    kappaB = max(KAPPA_GRID, key=lambda k: (resB[k][0]["accuracy_decisive"],
                                            -resB[k][0]["mean_regret_pct"]))

    print(f"\nselected on balanced only: tau={tauA}, kappa={kappaB}")
    for label, (ea, eu, ed) in [(f"RULE A tau={tauA}", resA[tauA]),
                                (f"RULE B kappa={kappaB}", resB[kappaB])]:
        print(f"\n  {label}")
        for name, ev in [("balanced (tuning)", ea), ("unbalanced (held out)", eu),
                         ("census (held out)", ed)]:
            print(f"    {name:<24} acc={ev['accuracy_decisive']:.3f} on "
                  f"{ev['n_decisive']}/{ev['n_configs']} decisive; regret mean "
                  f"{ev['mean_regret_pct']:.1f}% median {ev['median_regret_pct']:.1f}%")

    ng = {"synthetic": nogain_report(syn, kappaB), "census": nogain_report(dd, kappaB)}
    print(f"\nno-gain detector (kappa={kappaB}): synthetic {ng['synthetic']}")
    print(f"                                 census    {ng['census']}")

    sd = {name: evaluate_single_draw(grp, kappaB)
          for name, grp in [("balanced", bal), ("unbalanced", unb), ("census", dd)]}
    print(f"\nsingle-draw accuracy (rule B, kappa={kappaB}): "
          + ", ".join(f"{k} {v['accuracy_single_draw']:.3f}" for k, v in sd.items()))

    payload = {
        "rule_a": {"tau": tauA, "balanced": resA[tauA][0],
                   "unbalanced_heldout": resA[tauA][1], "census_heldout": resA[tauA][2]},
        "rule_b": {"kappa": kappaB, "balanced": resB[kappaB][0],
                   "unbalanced_heldout": resB[kappaB][1], "census_heldout": resB[kappaB][2]},
        "rule_a_grid": {str(t): [resA[t][0], resA[t][1], resA[t][2]] for t in TAU_GRID},
        "rule_b_grid": {str(k): [resB[k][0], resB[k][1], resB[k][2]] for k in KAPPA_GRID},
        "nogain": ng,
        "single_draw": sd,
        "decisive_winners": {name: {g: sum(1 for r in grp if r["decisive"] and r["best"] == g)
                                    for g in GRANS}
                             for name, grp in [("balanced", bal), ("unbalanced", unb),
                                               ("census", dd)]},
        "params": {"protect_max": PROTECT_MAX, "decisive_frac": DECISIVE_FRAC,
                   "nogain_frac": NOGAIN_FRAC},
        "setting": {"synthetic_results": results_json,
                    "projection": "exact_ls" if EXACT_LS else "hay_two_pass",
                    "census_heldout_set": "p1_sparse" if USE_P1 else "ddhca"},
    }
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"\nwrote -> {out_file} in {time.time()-t0:.1f}s")


# ---------------------------------------------------------------------------
# [PATCH v13] Sensitivity analysis for the diagnostic's heuristic constants
# (advisor Q3). Additive only: everything below this marker is new.
#   (a) lambda robustness  -- reuses resB, already computed on the full grid
#   (b) decisiveness-threshold sweep (paper default 2%)
#   (c) no-gain-threshold sweep      (paper default 5%)
# diagnostic_validation.json is untouched; new output goes to
# ../results/diagnostic_sensitivity.json.
# ---------------------------------------------------------------------------

def _fmt(x):
    return f"{x:.3f}" if isinstance(x, float) else "  -- "


def _re_decisive(r: dict, frac: float) -> bool:
    """Recompute decisiveness of an already-collected record at margin `frac`."""
    best_v = r["mse"][r["best"]]
    runner = min(v for g, v in r["mse"].items() if g != r["best"])
    return (runner - best_v) > frac * r["hay"]


def evaluate_at(records: list, decider, frac: float) -> dict:
    """evaluate(), but with the decisiveness margin recomputed at `frac`."""
    n_dec = n_correct = 0
    regrets = []
    for r in records:
        choice = decider(r["stats"])
        if _re_decisive(r, frac):
            n_dec += 1
            n_correct += (choice == r["best"])
        best = min(r["mse"].values())
        regrets.append(r["mse"][choice] / best - 1.0 if best > 0 else 0.0)
    return {"n_decisive": n_dec,
            "accuracy_decisive": (n_correct / n_dec) if n_dec else None,
            "median_regret_pct": 100 * float(np.median(regrets))}


def nogain_report_at(records: list, kappa: float, frac: float) -> dict:
    """nogain_report(), but with the predictor threshold set to `frac`."""
    def pred_one(stats):
        flags = []
        for s in stats:
            sigma2 = s["noise_floor"] / s["K"]["NODE"]
            best = min(s["R"][g] + kappa * sigma2 * s["K"][g] for g in GRANS)
            flags.append(best >= (1.0 - frac) * s["noise_floor"])
        return sum(flags) > len(flags) / 2
    pred = np.array([pred_one(r["stats"]) for r in records])
    true = np.array([r["nogain_true"] for r in records])
    return {"n_nogain_pred": int(pred.sum()),
            "precision": float((pred & true).sum() / pred.sum()) if pred.sum() else None,
            "recall": float((pred & true).sum() / true.sum()) if true.sum() else None}


if __name__ == "__main__":
    t1 = time.time()
    print("\n" + "=" * 74)
    print("SENSITIVITY ANALYSIS -- heuristic constants of Eq. (12) / Section 6")
    print("=" * 74)
    splits = [("balanced", bal), ("unbalanced", unb), ("census", dd)]

    # (a) lambda robustness: majority-vote accuracy, straight out of resB
    LAM_FOCUS = [0.25, 0.5, 1.0]
    print(f"\n(a) accuracy across lambda in {LAM_FOCUS} (tuned lambda={kappaB}):")
    lam_rows, lam_spread = {}, {}
    for i, (name, _) in enumerate(splits):
        accs = {str(k): resB[k][i]["accuracy_decisive"] for k in LAM_FOCUS}
        lam_rows[name] = accs
        vals = [v for v in accs.values() if v is not None]
        lam_spread[name] = 100 * (max(vals) - min(vals)) if vals else None
        print(f"    {name:<10} " +
              "  ".join(f"l={k}:{_fmt(v)}" for k, v in accs.items()) +
              f"   spread {_fmt(lam_spread[name])} pp" if vals else f"    {name}: --")

    print("\n    single-draw accuracy per lambda (deployment mode):")
    sd_rows = {}
    for k in LAM_FOCUS:
        row = {name: evaluate_single_draw(grp, k)["accuracy_single_draw"]
               for name, grp in splits}
        sd_rows[str(k)] = row
        print(f"    l={k:<5} " + "  ".join(f"{n}:{_fmt(v)}" for n, v in row.items()))

    # (b) decisiveness-threshold sweep at the tuned lambda
    DEC_SWEEP = [0.01, 0.02, 0.03, 0.05]
    dB = lambda s: decide_kappa(s, kappaB)
    print(f"\n(b) decisiveness margin sweep (lambda={kappaB}; paper default 2%):")
    dec_rows = {}
    for frac in DEC_SWEEP:
        row = {name: evaluate_at(grp, dB, frac) for name, grp in splits}
        dec_rows[str(frac)] = row
        print(f"    margin>{100*frac:>3g}%  " +
              " | ".join(f"{n} {_fmt(r['accuracy_decisive'])} (n={r['n_decisive']})"
                         for n, r in row.items()))

    # (c) no-gain-threshold sweep at the tuned lambda
    NOG_SWEEP = [0.03, 0.05, 0.075, 0.10]
    print(f"\n(c) no-gain threshold sweep (lambda={kappaB}; paper default 5%):")
    nog_rows = {}
    for frac in NOG_SWEEP:
        row = {"synthetic": nogain_report_at(syn, kappaB, frac),
               "census": nogain_report_at(dd, kappaB, frac)}
        nog_rows[str(frac)] = row
        sr, cr = row["synthetic"], row["census"]
        print(f"    within {100*frac:>4g}%   synthetic P={_fmt(sr['precision'])} "
              f"R={_fmt(sr['recall'])}   census false alarms={cr['n_nogain_pred']}")

    out = {"tuned": {"lambda": kappaB, "tau": tauA},
           "lambda_grid_accuracy": lam_rows,
           "lambda_spread_pp": lam_spread,
           "single_draw_by_lambda": sd_rows,
           "decisive_sweep": dec_rows,
           "nogain_sweep": nog_rows}
    SENS_FILE = ("../results/diagnostic_sensitivity_exactls.json" if EXACT_LS
                 else "../results/diagnostic_sensitivity.json")
    with open(SENS_FILE, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote -> {SENS_FILE} in {time.time()-t1:.1f}s")

    print("\nPAPER SENTENCE INPUTS: held-out accuracy spread over lambda in "
          f"[0.25, 1]: unbalanced {_fmt(lam_spread.get('unbalanced'))} pp, "
          f"census {_fmt(lam_spread.get('census'))} pp")