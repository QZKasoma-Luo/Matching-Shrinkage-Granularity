"""
PL94 P1 sparse race-combination columns as fully observed sparse real releases.

Why. The Detailed DHC-A groups reach the sparse regime only because counties
below the publication threshold (noisy count < 22) are absent from the table and
were padded with 0, so their true counts are "fewer than 22", not "zero". The
PL94-171 P1 table has no threshold: every one of its 71 race columns is
published for every county, zeros included. Its multi-race combination columns
are as sparse as the DDHC-A groups (81-95 % of counties at zero, all counts
small) and therefore give a sparse real signal whose proxy truth is fully
observed. Same tree as the P1 total-population control (nation -> state ->
county), same mechanism, same three granularities.

    python P1_sparse_columns.py                       # default 5 columns, Hay two-pass
    python P1_sparse_columns.py --exact-ls            # exact LS projection (cached, any tree)
    python P1_sparse_columns.py --columns P1_060N P1_043N
    python P1_sparse_columns.py --trials 20           # quick pass (default NUM_TRIALS = 50)
    python P1_sparse_columns.py --list                # rank all P1 columns by sparsity
    python P1_sparse_columns.py --columns P1_001N --exact-ls   # the total-population control row
                                                      # (blocks merge into the same JSON by column)

Outputs (run from core/):
    ../results/P1_sparse_results/p1_sparse_results.json
    ../census_data/census_hierarchy_format/P1_census_county_2020_<col>.json  (one per column)

Per (column, epsilon) the JSON holds, for Hay / Scalar / Level / Node: per-cell
MSE (mean, std over draws), improvement over Hay (ratio of means, as in the
paper's Table 1), the paired per-draw improvement SE (the +/- of Table 1), and
the cell-type-stratified MSE (populated leaves x > 0, empty leaves x == 0,
internal nodes) behind the paper's Table 2.
"""
from __future__ import annotations
import csv
import json
import os
import sys
import time
import numpy as np

from hierarchy_data_generator import create_unbalanced_tree
from laplace_noise_injection import add_laplace_noise, noise_variance, EPSILONS, NUM_TRIALS
from projection import hay_two_pass, projection_matrix
from shrinkage import level_coefficients, _apply_level_scaling
from diagnostic_check import rule_stats, decide_kappa, GRANS, DECISIVE_FRAC

P1_CSV = "../census_data/original_data/P1/DECENNIALPL2020.P1-Data.csv"
HIER_DIR = "../census_data/census_hierarchy_format"
RESULTS_DIR = "../results/P1_sparse_results"
OUT_FILE = os.path.join(RESULTS_DIR, "p1_sparse_results.json")

# Six multi-race combination columns spanning the regime the way the DDHC-A
# groups did, ordered by the expected leaf-level coefficient E[c] at eps = 0.5
# (E[c] = ||x_leaf||^2 / (||x_leaf||^2 + n sigma^2); well below 1 => shrinkage
# active). All are "population of N races" columns, i.e. small population groups.
#   P1_060N  E[c]@.5 = 0.006  94.5 % zeros, max  19   (every cell below the noise)
#   P1_058N  E[c]@.5 = 0.150  86.0 % zeros, max 165
#   P1_068N  E[c]@.5 = 0.318  89.9 % zeros, max 303   (closest to the Emirati group)
#   P1_044N  E[c]@.5 = 0.555  81.2 % zeros, max 374
#   P1_043N  E[c]@.5 = 0.733  81.1 % zeros, max 681
#   P1_057N  E[c]@.5 = 0.956  75.9 % zeros, max 2046  (high-SNR end)
DEFAULT_COLUMNS = ["P1_060N", "P1_058N", "P1_068N", "P1_044N", "P1_043N", "P1_057N"]

SEED_BASE = 1_500_000        # distinct from the P1-total (900k) and DDHC-A (1.3M) schedules
DIAG_FILE = "../results/diagnostic_validation.json"
GRAN_OF = {"Scalar": "SCALAR", "Level": "LEVEL", "Node": "NODE"}


def tuned_kappa(default: float = 0.25) -> float:
    """The diagnostic's penalty constant, tuned on balanced synthetic configs
    only (Section 6); read from the shipped validation file if present."""
    try:
        with open(DIAG_FILE) as f:
            return float(json.load(f)["rule_b"]["kappa"])
    except (OSError, KeyError, ValueError):
        return default


# --------------------------------------------------------------------------- data
def read_p1(csv_path: str):
    """Return (header, labels, county_rows). Rows keep the file order, which is
    the order the P1 total-population archive used, so the tree is identical."""
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    header = [h.strip().strip('"').strip() for h in rows[0]]
    labels = rows[1]
    gi = header.index("GEO_ID")
    county_rows = [r for r in rows[2:] if r and r[gi] and r[gi].startswith("0500000US")]
    return header, labels, county_rows


def rank_columns(csv_path: str = P1_CSV, top: int = 40):
    """Rank all P1_xxxN columns by the expected leaf-level shrinkage coefficient
    E[c] at eps = 0.5 (ascending = sparsest / lowest SNR first), with the zero
    fraction, max county count and total, to pick a regime-spanning set."""
    header, labels, rows = read_p1(csv_path)
    cols = [i for i, h in enumerate(header) if h.startswith("P1_") and h.endswith("N")]
    n = len(rows)
    s2 = {e: 2.0 * (3.0 / e) ** 2 for e in (0.5, 1.0, 2.0)}      # sensitivity 3
    out = []
    for i in cols:
        v = np.array([int(r[i]) for r in rows])
        ssq = float((v ** 2).sum())
        ec = {e: ssq / (ssq + n * s) for e, s in s2.items()}
        out.append((header[i], 100 * np.mean(v == 0), int(v.max()), int(v.sum()), ec,
                    labels[i].replace("!!", " ").strip()))
    out.sort(key=lambda t: t[4][0.5])
    print(f"{'column':<9}{'zero%':>7}{'max':>7}{'total':>10}  {'E[c]@.5':>8}{'@1':>7}{'@2':>7}  label")
    for name, z, mx, tot, ec, lab in out[:top]:
        print(f"{name:<9}{z:7.1f}{mx:7d}{tot:10d}  {ec[0.5]:8.3f}{ec[1.0]:7.3f}{ec[2.0]:7.3f}  {lab[:60]}")
    return out


def load_p1_column(csv_path: str, column: str, keep_puerto_rico: bool = True):
    """Build the (nation -> state -> county) tree and the proxy signal x for ONE
    P1 column. Same construction as P1_census_loader.load_census (file order,
    states sorted by FIPS), so the tree matches the existing P1 archive."""
    header, labels, rows = read_p1(csv_path)
    gi, ni, ci = header.index("GEO_ID"), header.index("NAME"), header.index(column)
    label = labels[ci].replace("!!", " ").strip()

    counties = []   # (geoid, name, count, state_fips, state_name)
    for r in rows:
        geoid = r[gi]
        fips = geoid.split("US")[-1]
        sf = fips[:2]
        if not keep_puerto_rico and sf == "72":
            continue
        name = r[ni]
        sn = name.rsplit(",", 1)[-1].strip() if "," in name else sf
        counties.append((geoid, name, int(str(r[ci]).replace(",", "").strip()), sf, sn))

    state_fips_list = sorted({c[3] for c in counties})
    state_node = {sf: 1 + i for i, sf in enumerate(state_fips_list)}
    n_states = len(state_fips_list)
    county_node, edges = {}, []
    nid = 1 + n_states
    for geoid, name, cnt, sf, sn in counties:
        county_node[geoid] = nid
        edges.append((state_node[sf], nid))
        nid += 1
    for sf in state_fips_list:
        edges.append((0, state_node[sf]))
    h = create_unbalanced_tree(edges, root=0)

    x = np.zeros(h.num_nodes)
    for geoid, name, cnt, sf, sn in counties:
        x[county_node[geoid]] = cnt
    for sf in state_fips_list:
        x[state_node[sf]] = sum(x[county_node[c[0]]] for c in counties if c[3] == sf)
    x[0] = x[[state_node[sf] for sf in state_fips_list]].sum()

    node_info = {0: {"geoid": "0100000US", "name": "United States", "kind": "nation"}}
    seen = {}
    for geoid, name, cnt, sf, sn in counties:
        seen.setdefault(sf, sn)
        node_info[county_node[geoid]] = {"geoid": geoid, "name": name, "kind": "county"}
    for sf, sn in seen.items():
        node_info[state_node[sf]] = {"geoid": f"0400000US{sf}", "name": sn, "kind": "state"}

    leaf_counts = np.array([c[2] for c in counties])
    meta = {
        "column": column, "label": label,
        "n_counties": len(counties), "n_states": n_states,
        "n_counties_nonzero": int((leaf_counts > 0).sum()),
        "zero_fraction": float(np.mean(leaf_counts == 0)),
        "max_county_count": int(leaf_counts.max()),
        "total_population": int(x[0]),
        "node_info": node_info, "edges": [[int(u), int(v)] for u, v in edges],
    }
    return h, x, meta


def save_archive(h, x, meta, out_dir: str = HIER_DIR) -> str:
    os.makedirs(out_dir, exist_ok=True)
    record = {
        "source": f"2020 Census PL94-171, table P1 ({meta['column']}), county level",
        "note": "Published PL94 counts used as ground-truth PROXY (already TopDown/"
                "DP-processed). Every county is published, zeros included (no "
                "suppression, no threshold); state/nation values are bottom-up sums.",
        "column": meta["column"], "label": meta["label"],
        "num_nodes": int(h.num_nodes), "n_states": meta["n_states"],
        "n_counties": meta["n_counties"], "n_counties_nonzero": meta["n_counties_nonzero"],
        "zero_fraction": meta["zero_fraction"], "max_county_count": meta["max_county_count"],
        "height": int(h.height), "sensitivity": int(h.sensitivity),
        "total_population": meta["total_population"],
        "edges": meta["edges"], "levels": [int(v) for v in h.levels],
        "level_sizes": [int(h.num_nodes_at_level(l)) for l in range(h.depth)],
        "x": [float(v) for v in x],
        "node_info": {str(k): v for k, v in meta["node_info"].items()},
    }
    path = os.path.join(out_dir, f"P1_census_county_2020_{meta['column']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=1)
    return path


# ------------------------------------------------------------------ estimators
def make_exact_projector(h):
    """Exact orthogonal LS projection with the projector cached once per tree
    (dense n x n; ~86 MB for the 3,274-node county tree, a few seconds)."""
    P = projection_matrix(h)
    return lambda y, hh: P @ y


def make_methods(h, project):
    """The four estimators of Table 1, all ending with the SAME projection."""
    n = h.num_nodes

    def hay(z, hh, s2):
        return project(z, hh)

    def scalar(z, hh, s2):
        e = float(np.dot(z, z))
        c = max(0.0, 1.0 - n * s2 / e) if e > 0 else 0.0
        return project(c * z, hh)

    def level(z, hh, s2):
        return project(_apply_level_scaling(z, hh, level_coefficients(z, hh, s2)), hh)

    def node(z, hh, s2):
        with np.errstate(divide="ignore", invalid="ignore"):
            c = np.where(z * z > 0, np.maximum(0.0, 1.0 - s2 / (z * z)), 0.0)
        return project(c * z, hh)

    return {"Hay": hay, "Scalar": scalar, "Level": level, "Node": node}


# ------------------------------------------------------------------ experiment
def run_column(column: str, csv_path: str, n_trials: int, exact_ls: bool, t0: float,
               kappa: float) -> list:
    h, x, meta = load_p1_column(csv_path, column)
    save_archive(h, x, meta)
    colnum = int(column[3:6])
    project = make_exact_projector(h) if exact_ls else hay_two_pass
    methods = make_methods(h, project)

    leaf = h.level_mask(h.depth - 1)
    masks = {"present": leaf & (x > 0), "empty": leaf & (x == 0), "internal": ~leaf}
    print(f"\n=== {column}  {meta['label']}\n    {meta['n_counties']} counties, "
          f"{meta['n_counties_nonzero']} nonzero ({100*(1-meta['zero_fraction']):.1f} %), "
          f"max {meta['max_county_count']}, total {meta['total_population']:,}; "
          f"projection = {'exact LS' if exact_ls else 'Hay two-pass'}")

    rows = []
    for ei, eps in enumerate(EPSILONS):
        s2 = noise_variance(h, eps)
        acc = {m: {"all": [], "present": [], "empty": [], "internal": []} for m in methods}
        stats = []
        for t in range(n_trials):
            rng = np.random.default_rng(SEED_BASE + colnum * 100_000 + ei * 10_000 + t)
            z = add_laplace_noise(x, h, eps, rng)
            stats.append(rule_stats(z, h, s2))
            for m, fn in methods.items():
                d2 = (fn(z, h, s2) - x) ** 2
                acc[m]["all"].append(d2.mean())
                for k, msk in masks.items():
                    acc[m][k].append(d2[msk].mean())
        hay_draws = np.asarray(acc["Hay"]["all"])
        hay_mean = float(hay_draws.mean())
        block = {}
        for m in methods:
            all_draws = np.asarray(acc[m]["all"])
            paired = 100.0 * (1.0 - all_draws / hay_draws)
            block[m] = {
                "mse_mean": float(all_draws.mean()),
                "mse_std": float(all_draws.std()),
                "improve_over_hay_pct": 100.0 * (hay_mean - float(all_draws.mean())) / hay_mean,
                "imp_se": float(np.std(paired, ddof=1) / np.sqrt(n_trials)),
                "mse_present": float(np.mean(acc[m]["present"])),
                "mse_empty": float(np.mean(acc[m]["empty"])),
                "mse_internal": float(np.mean(acc[m]["internal"])),
            }
        # granularity diagnostic (Section 6, rule B): majority over draws and
        # single-draw accuracy, against the empirical best of the three.
        cand = {GRAN_OF[m]: block[m]["mse_mean"] for m in GRAN_OF}
        best = min(cand, key=cand.get)
        runner = min(v for g, v in cand.items() if g != best)
        decisive = (runner - cand[best]) > DECISIVE_FRAC * hay_mean
        choice = decide_kappa(stats, kappa)
        single = [decide_kappa([st], kappa) for st in stats]
        diag = {"kappa": kappa, "empirical_best": best, "decisive": bool(decisive),
                "choice_majority": choice, "correct_majority": bool(choice == best),
                "single_draw_accuracy": float(np.mean([c == best for c in single])),
                "regret_pct": 100.0 * (cand[choice] / cand[best] - 1.0) if cand[best] > 0 else 0.0}
        rows.append({
            "column": column, "label": meta["label"], "epsilon": eps, "sigma2": s2,
            "num_trials": n_trials, "num_nodes": int(h.num_nodes),
            "projection": "exact_ls" if exact_ls else "hay_two_pass",
            "strata_counts": {k: int(v.sum()) for k, v in masks.items()},
            "methods": block, "diagnostic": diag,
        })
        b = block
        print(f"  [{time.time()-t0:6.1f}s] eps={eps:<5} improve%: Scalar {b['Scalar']['improve_over_hay_pct']:6.1f}"
              f"  Level {b['Level']['improve_over_hay_pct']:6.1f}  Node {b['Node']['improve_over_hay_pct']:6.1f}"
              f" | present x(Hay): Level {b['Level']['mse_present']/b['Hay']['mse_present']:5.2f}"
              f"  Node {b['Node']['mse_present']/b['Hay']['mse_present']:4.2f}"
              f" | diag: {choice:<6} best {best:<6}{'' if decisive else ' (not decisive)'}")
    return rows


def print_tables(rows: list) -> None:
    cols = list(dict.fromkeys(r["column"] for r in rows))
    eps_list = list(dict.fromkeys(r["epsilon"] for r in rows))
    print("\n" + "=" * 100)
    print("TABLE 1 style -- improvement in per-cell MSE over consistency projection (%), +/- paired SE")
    print(f"{'column':<9}{'method':<8}" + "".join(f"{('eps='+str(e)):>13}" for e in eps_list))
    for c in cols:
        for m in ("Scalar", "Level", "Node"):
            cells = []
            for e in eps_list:
                r = next(r for r in rows if r["column"] == c and r["epsilon"] == e)
                b = r["methods"][m]
                cells.append(f"{b['improve_over_hay_pct']:6.1f} ±{b['imp_se']:3.1f}")
            print(f"{c:<9}{m:<8}" + "".join(f"{s:>13}" for s in cells))
    print("\nTABLE 2 style -- per-cell MSE by cell type (populated leaves / empty leaves / all), eps = 0.5 and 1.0")
    print(f"{'column':<9}{'eps':>5} {'#pop':>5} | {'Hay pop':>8} {'Level pop':>10} {'Node pop':>9} | "
          f"{'Hay empty':>9} {'Level empty':>11} {'Node empty':>10} | {'Level agg%':>10} {'Node agg%':>9}")
    for c in cols:
        for e in (0.5, 1.0):
            r = next((r for r in rows if r["column"] == c and r["epsilon"] == e), None)
            if r is None:
                continue
            b = r["methods"]
            print(f"{c:<9}{e:>5} {r['strata_counts']['present']:>5} | {b['Hay']['mse_present']:8.2f} "
                  f"{b['Level']['mse_present']:10.2f} {b['Node']['mse_present']:9.2f} | "
                  f"{b['Hay']['mse_empty']:9.2f} {b['Level']['mse_empty']:11.2f} {b['Node']['mse_empty']:10.2f} | "
                  f"{b['Level']['improve_over_hay_pct']:10.1f} {b['Node']['improve_over_hay_pct']:9.1f}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--list" in args:
        rank_columns()
        sys.exit(0)
    exact_ls = "--exact-ls" in args
    n_trials = NUM_TRIALS
    if "--trials" in args:
        i = args.index("--trials")
        n_trials = int(args[i + 1])
        del args[i:i + 2]
    columns = DEFAULT_COLUMNS
    if "--columns" in args:
        i = args.index("--columns")
        columns = [a for a in args[i + 1:] if a.startswith("P1_")]
    kappa = tuned_kappa()
    print(f"P1 sparse columns | {columns} | {n_trials} trials x {len(EPSILONS)} eps | "
          f"projection: {'exact LS' if exact_ls else 'Hay two-pass'} | diagnostic kappa={kappa}")
    t0 = time.time()
    rows = []
    for c in columns:
        rows += run_column(c, P1_CSV, n_trials, exact_ls, t0, kappa)
    print_tables(rows)
    dec = [r for r in rows if r["diagnostic"]["decisive"]]
    if dec:
        acc = np.mean([r["diagnostic"]["correct_majority"] for r in dec])
        sd = np.mean([r["diagnostic"]["single_draw_accuracy"] for r in dec])
        reg = [r["diagnostic"]["regret_pct"] for r in rows]
        print(f"\nDIAGNOSTIC (rule B, kappa={kappa}): accuracy {acc:.3f} on {len(dec)}/{len(rows)} "
              f"decisive configs; single-draw {sd:.3f}; regret mean {np.mean(reg):.1f}% "
              f"median {np.median(reg):.1f}%")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # Merge by column: blocks of columns run now replace their old blocks, all
    # other columns' blocks are kept, so the control (P1_001N) and the sparse
    # columns can be run in separate invocations into one file.
    kept = []
    if os.path.exists(OUT_FILE):
        with open(OUT_FILE) as f:
            old = json.load(f)
        kept = [r for r in old["results"] if r["column"] not in columns]
        mixed = {r["projection"] for r in kept} - {"exact_ls" if exact_ls else "hay_two_pass"}
        if mixed:
            print(f"  !! WARNING: kept blocks use projection {mixed}; this run uses "
                  f"{'exact_ls' if exact_ls else 'hay_two_pass'} -- re-run those columns too.")
    all_rows = kept + rows
    all_cols = list(dict.fromkeys(r["column"] for r in all_rows))
    with open(OUT_FILE, "w") as f:
        json.dump({"config": {"columns": all_cols, "epsilons": EPSILONS, "num_trials": n_trials,
                              "projection": "exact_ls" if exact_ls else "hay_two_pass",
                              "seed_base": SEED_BASE, "diagnostic_kappa": kappa,
                              "source": "2020 PL94-171 P1, county level, proxy GT, no suppression"},
                   "results": all_rows}, f, indent=1)
    print(f"\nwrote {len(rows)} new + {len(kept)} kept (column, eps) blocks -> {OUT_FILE} "
          f"in {time.time()-t0:.1f}s")
