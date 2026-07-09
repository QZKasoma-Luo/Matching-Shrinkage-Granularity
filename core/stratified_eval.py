"""
Evaluation suite on the archived-seed draws -- the data behind the paper's
Section `sec:stratified`, the threshold remark in Section 4.2, and the error
bars of the main real-data table (tab:ddhca).

One pass over the exact run_ddhca draws (seeds 1_300_000 + ei*10_000 + trial)
computes, for each selected DDHC-A population group and epsilon:

  1. Cell-type-stratified MSE -- populated leaves (x > 0), empty leaves
     (x == 0), internal nodes -- for every method below.
  2. Aggregate improvement over Hay, including the two universal-threshold
     per-node variants (lambda = sigma*sqrt(2 ln n)):
       SoftU: sign(z) max(|z| - lambda, 0)      HardU: z * 1{|z| > lambda}
     which locate the garrote (threshold sigma) on the threshold axis.
  3. Paired error bars: the per-draw improvement 100*(1 - MSE_m(t)/MSE_Hay(t))
     and its standard error over the 50 draws (pairing on the shared z is what
     makes these tight). Also computed for the PL94 control (seeds 900_000+t).

The recomputed aggregate MSEs are asserted to match the published
ddhca_results_pop*.json exactly (same draws), cross-validating both scripts.
Also measures, once per census tree, the deviation of Hay's two-pass from the
exact least-squares projection (the Method-section remark).

Output: ../results/census_ddhca_results/stratified_results.json
Run:    python stratified_eval.py          (from core/)
"""
from __future__ import annotations
import json
import math
import os
import time
import numpy as np

from T01001_census_loader import (SELECTED_POPGROUPS, CENSUS_DDHCA_HIER_DIR,
                                  CENSUS_DDHCA_RESULTS_DIR,
                                  load_hierarchy_archive)
from laplace_noise_injection import add_laplace_noise, noise_variance, EPSILONS, NUM_TRIALS
from projection import hay_two_pass, ls_projection
from shrinkage import level_wise_shrinkage
from ablations import scalar_shrinkage, per_node_shrinkage

P1_ARCHIVE = "../census_data/census_hierarchy_format/P1_census_county_2020.json"
OUT_FILE = os.path.join(CENSUS_DDHCA_RESULTS_DIR, "stratified_results.json")


def _lam(h, s2):
    return math.sqrt(s2) * math.sqrt(2.0 * math.log(h.num_nodes))


def soft_universal(z, h, s2):
    lam = _lam(h, s2)
    return hay_two_pass(np.sign(z) * np.maximum(np.abs(z) - lam, 0.0), h)


def hard_universal(z, h, s2):
    lam = _lam(h, s2)
    return hay_two_pass(z * (np.abs(z) > lam), h)


# local name -> (method, name in the published ddhca results files or None)
METHODS = {
    "Laplace": (lambda z, h, s2: z,                  "Laplace"),
    "Hay":     (lambda z, h, s2: hay_two_pass(z, h), "Hay"),
    "Scalar":  (scalar_shrinkage,                    "Scalar"),
    "Level":   (level_wise_shrinkage,                "Ours"),
    "Node":    (per_node_shrinkage,                  "PerNode"),
    "SoftU":   (soft_universal,                      None),   # threshold variants:
    "HardU":   (hard_universal,                      None),   # not in the main runs
}
SE_METHODS = ("Scalar", "Level", "Node")   # error bars reported in tab:ddhca


def strata_masks(h, x):
    leaf = h.level_mask(h.depth - 1)
    return {
        "present":  leaf & (x > 0),
        "empty":    leaf & (x == 0),
        "internal": ~leaf,
    }


def run_group(pg: str, results: list, t0: float) -> None:
    arch = os.path.join(CENSUS_DDHCA_HIER_DIR,
                        f"T01001_census_county_2020_pop{pg}.json")
    h, x, rec = load_hierarchy_archive(arch)
    masks = strata_masks(h, x)
    counts = {k: int(m.sum()) for k, m in masks.items()}

    # published aggregate results for the cross-check
    with open(os.path.join(CENSUS_DDHCA_RESULTS_DIR,
                           f"ddhca_results_pop{pg}.json")) as f:
        published = {r["epsilon"]: r["methods"] for r in json.load(f)["results"]}

    for ei, eps in enumerate(EPSILONS):
        s2 = noise_variance(h, eps)
        acc = {m: {"all": [], "present": [], "empty": [], "internal": []}
               for m in METHODS}
        for trial in range(NUM_TRIALS):
            rng = np.random.default_rng(1_300_000 + ei * 10_000 + trial)
            z = add_laplace_noise(x, h, eps, rng)
            for name, (fn, _) in METHODS.items():
                d2 = (fn(z, h, s2) - x) ** 2
                acc[name]["all"].append(d2.mean())
                for k, m in masks.items():
                    acc[name][k].append(d2[m].mean())

        hay_draws = np.asarray(acc["Hay"]["all"])
        block = {}
        for name, (_, pub_name) in METHODS.items():
            mse_all = float(np.mean(acc[name]["all"]))
            if pub_name is not None:
                pub = published[eps][pub_name]["mse_mean"]
                if not np.isclose(mse_all, pub, rtol=1e-6):
                    raise AssertionError(
                        f"pop{pg} eps={eps} {name}: recomputed {mse_all} != "
                        f"published {pub} -- seed/method mismatch?")
            block[name] = {
                "mse_all": mse_all,
                "mse_present": float(np.mean(acc[name]["present"])),
                "mse_empty": float(np.mean(acc[name]["empty"])),
                "mse_internal": float(np.mean(acc[name]["internal"])),
                "improve_over_hay_pct":
                    100.0 * (float(np.mean(hay_draws)) - mse_all)
                    / float(np.mean(hay_draws)),
            }
            if name in SE_METHODS:
                paired = 100.0 * (1.0 - np.asarray(acc[name]["all"]) / hay_draws)
                block[name]["imp_se"] = float(np.std(paired, ddof=1)
                                              / np.sqrt(NUM_TRIALS))
        results.append({
            "popgroup": pg, "popgroup_label": rec["popgroup_label"],
            "epsilon": eps, "sigma2": s2, "num_trials": NUM_TRIALS,
            "strata_counts": counts, "methods": block,
        })
        hp = block["Hay"]["mse_present"]
        print(f"  [{time.time()-t0:6.1f}s] pop{pg} eps={eps:>5}: present-cell MSE "
              f"Hay={hp:.2f} Level={block['Level']['mse_present']:.2f} "
              f"(x{block['Level']['mse_present']/hp:.1f}) "
              f"Node={block['Node']['mse_present']:.2f} "
              f"(x{block['Node']['mse_present']/hp:.1f}) | "
              f"agg HardU {block['HardU']['improve_over_hay_pct']:+.1f}%  "
              f"[aggregate check ok]")


def p1_improvement_se() -> dict:
    """Paired improvement SEs for the PL94 control (seeds 900_000 + t)."""
    h, x, _ = load_hierarchy_archive(P1_ARCHIVE)
    out = {}
    for eps in EPSILONS:
        s2 = noise_variance(h, eps)
        per_draw = {m: [] for m in SE_METHODS}
        for t in range(NUM_TRIALS):
            z = add_laplace_noise(x, h, eps, np.random.default_rng(900_000 + t))
            mh = float(np.mean((hay_two_pass(z, h) - x) ** 2))
            for m in SE_METHODS:
                mm = float(np.mean((METHODS[m][0](z, h, s2) - x) ** 2))
                per_draw[m].append(100.0 * (1.0 - mm / mh))
        out[str(eps)] = {m: {"imp_se": float(np.std(np.asarray(per_draw[m]),
                                                    ddof=1) / np.sqrt(NUM_TRIALS))}
                         for m in SE_METHODS}
    return out


def measure_hay_vs_ls(n_draws: int = 5, eps: float = 1.0) -> dict:
    """Hay two-pass vs exact LS projection on the census tree (once; the tree
    is identical across groups). Uses the run_ddhca draws for that epsilon."""
    arch = os.path.join(CENSUS_DDHCA_HIER_DIR,
                        f"T01001_census_county_2020_pop{SELECTED_POPGROUPS[0]}.json")
    h, x, _ = load_hierarchy_archive(arch)
    ei = EPSILONS.index(eps)
    maxdiff, mse_hay, mse_ls = [], [], []
    for trial in range(n_draws):
        rng = np.random.default_rng(1_300_000 + ei * 10_000 + trial)
        z = add_laplace_noise(x, h, eps, rng)
        a, b = ls_projection(z, h), hay_two_pass(z, h)
        maxdiff.append(float(np.max(np.abs(a - b))))
        mse_ls.append(float(np.mean((a - x) ** 2)))
        mse_hay.append(float(np.mean((b - x) ** 2)))
    out = {
        "epsilon": eps, "n_draws": n_draws, "num_nodes": int(h.num_nodes),
        "max_coord_diff": max(maxdiff),
        "mse_hay_mean": float(np.mean(mse_hay)),
        "mse_ls_mean": float(np.mean(mse_ls)),
        "mse_rel_diff_pct": float(100 * abs(np.mean(mse_hay) - np.mean(mse_ls))
                                  / np.mean(mse_ls)),
    }
    print(f"\nHay vs exact LS on the census tree (eps={eps}, {n_draws} draws): "
          f"max|diff|={out['max_coord_diff']:.3f}, "
          f"MSE Hay {out['mse_hay_mean']:.3f} vs LS {out['mse_ls_mean']:.3f} "
          f"({out['mse_rel_diff_pct']:.2f}% apart)")
    return out


if __name__ == "__main__":
    t0 = time.time()
    results: list = []
    for pg in SELECTED_POPGROUPS:
        run_group(pg, results, t0)
    print(f"  [{time.time()-t0:6.1f}s] PL94 control (paired SEs)...")
    p1_se = p1_improvement_se()
    hay_ls = measure_hay_vs_ls()
    os.makedirs(CENSUS_DDHCA_RESULTS_DIR, exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump({"strata": ["present", "empty", "internal"],
                   "note": "same noise draws as run_ddhca; aggregate MSE "
                           "cross-checked against ddhca_results_pop*.json; "
                           "imp_se = paired per-draw improvement SE over 50 "
                           "draws; SoftU/HardU use lambda = sigma*sqrt(2 ln n)",
                   "hay_vs_ls": hay_ls,
                   "p1_improvement_se": p1_se,
                   "results": results}, f, indent=1)
    print(f"\nwrote {len(results)} (group, eps) blocks -> {OUT_FILE} "
          f"in {time.time()-t0:.1f}s")
