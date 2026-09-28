"""
Sensitivity of the DDHC-A results to the treatment of ABSENT counties.

Context. In table T01001 a population group is listed for a county only if it
met the publication threshold (noisy count >= 22); counties below the threshold
are simply absent from the file, and the archives set them to 0. Their true
counts are therefore "fewer than 22", not "known to be zero". (The literal "X"
cells are a different, rare phenomenon -- negative noisy counts or an alone
count exceeding its alone-or-in-combination count; only 2 of the 16,105 cells
of the five selected groups are "X".)

This script re-runs Scalar / Level / Node on the archived census trees after
assigning a fraction f of the absent counties a count drawn uniformly from
{1, ..., 21}, and reports how the aggregate improvement over Hay and the
populated-cell error ratio move with f.

    python sensitivity_absent_counties.py                 # all groups, default grid
    python sensitivity_absent_counties.py 1190 1009       # selected groups only
    python sensitivity_absent_counties.py --draws 50      # more noise draws

Output: ../results/census_ddhca_results/sensitivity_absent_counties.json

Next step (preferred): replace the uniform imputation by the state residual --
download the state-level T01001 rows (published for every group), compute
state_total - sum(published county counts) per state, and spread that mass over
the absent counties of the state. Hook: pass `--residuals <json>` mapping
state FIPS -> hidden count (not implemented here; see rebuild()).
"""
from __future__ import annotations
import json
import os
import sys
import time
import numpy as np

from T01001_census_loader import (SELECTED_POPGROUPS, CENSUS_DDHCA_HIER_DIR,
                                  CENSUS_DDHCA_RESULTS_DIR, load_hierarchy_archive)
from laplace_noise_injection import add_laplace_noise, noise_variance
from projection import hay_two_pass
from shrinkage import level_wise_shrinkage
from ablations import per_node_shrinkage, scalar_shrinkage

FRACTIONS = [0.0, 0.05, 0.10, 0.30]
EPS_LIST = [0.5, 1.0, 2.0]
OUT_FILE = os.path.join(CENSUS_DDHCA_RESULTS_DIR, "sensitivity_absent_counties.json")
IMPUTE_SEED = 7
NOISE_SEED_BASE = 2_000_000

METHODS = {
    "Hay":    lambda z, h, s2: hay_two_pass(z, h),
    "Scalar": scalar_shrinkage,
    "Level":  level_wise_shrinkage,
    "Node":   per_node_shrinkage,
}


def rebuild(h, x_leaf):
    """Consistent x from leaf values: internal nodes = bottom-up sums."""
    x = np.zeros(h.num_nodes)
    leaf = h.level_mask(h.depth - 1)
    x[leaf] = x_leaf[leaf]
    for v in sorted(range(h.num_nodes), key=lambda v: -h.levels[v]):
        kids = list(h.graph.successors(v))
        if kids:
            x[v] = sum(x[c] for c in kids)
    return x


def run_group(pg: str, fractions, eps_list, n_draws: int) -> list:
    arch = os.path.join(CENSUS_DDHCA_HIER_DIR, f"T01001_census_county_2020_pop{pg}.json")
    h, x0, rec = load_hierarchy_archive(arch)
    leaf = h.level_mask(h.depth - 1)
    present, absent = leaf & (x0 > 0), leaf & (x0 == 0)
    print(f"\n=== pop{pg} {rec['popgroup_label']}: published {present.sum()}, "
          f"absent {absent.sum()}, published total {x0[leaf].sum():.0f}")
    rows = []
    for f in fractions:
        rng = np.random.default_rng(IMPUTE_SEED)
        idx = np.where(absent)[0]
        k = int(round(f * len(idx)))
        chosen = rng.choice(idx, size=k, replace=False) if k else np.array([], dtype=int)
        x = x0.copy()
        if k:
            x[chosen] = rng.integers(1, 22, size=k)
        x = rebuild(h, x)
        imputed = np.zeros(h.num_nodes, bool)
        imputed[chosen] = True
        for ei, eps in enumerate(eps_list):
            s2 = noise_variance(h, eps)
            acc = {m: {"all": [], "present": [], "imputed": []} for m in METHODS}
            for t in range(n_draws):
                z = add_laplace_noise(x, h, eps,
                                      np.random.default_rng(NOISE_SEED_BASE + ei * 1000 + t))
                for m, fn in METHODS.items():
                    d2 = (fn(z, h, s2) - x) ** 2
                    acc[m]["all"].append(d2.mean())
                    acc[m]["present"].append(d2[present].mean())
                    acc[m]["imputed"].append(d2[imputed].mean() if k else float("nan"))
            H = float(np.mean(acc["Hay"]["all"]))
            Hp = float(np.mean(acc["Hay"]["present"]))
            row = {"popgroup": pg, "fraction": f, "hidden_mass": float(x[chosen].sum()) if k else 0.0,
                   "n_imputed": int(k), "epsilon": eps, "n_draws": n_draws, "methods": {}}
            for m in METHODS:
                row["methods"][m] = {
                    "improve_over_hay_pct": 100.0 * (H - float(np.mean(acc[m]["all"]))) / H,
                    "present_ratio_vs_hay": float(np.mean(acc[m]["present"])) / Hp,
                    "mse_imputed": float(np.nanmean(acc[m]["imputed"])) if k else None,
                }
            rows.append(row)
            r = row["methods"]
            print(f"  f={f:<5} hidden={row['hidden_mass']:>7.0f} eps={eps:<4} | "
                  f"Level {r['Level']['improve_over_hay_pct']:5.1f}%  "
                  f"Node {r['Node']['improve_over_hay_pct']:5.1f}%  "
                  f"Scalar {r['Scalar']['improve_over_hay_pct']:5.1f}% | present x(Hay): "
                  f"Level {r['Level']['present_ratio_vs_hay']:5.2f}  Node {r['Node']['present_ratio_vs_hay']:4.2f}")
    return rows


if __name__ == "__main__":
    args = sys.argv[1:]
    n_draws = 20
    if "--draws" in args:
        i = args.index("--draws")
        n_draws = int(args[i + 1])
        del args[i:i + 2]
    groups = args or SELECTED_POPGROUPS
    t0 = time.time()
    rows = []
    for pg in groups:
        rows += run_group(pg, FRACTIONS, EPS_LIST, n_draws)
    os.makedirs(CENSUS_DDHCA_RESULTS_DIR, exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump({"note": "absent (below-threshold) counties imputed with a fraction f "
                           "of Uniform{1..21} counts; same methods as run_ddhca; "
                           "improvements relative to Hay two-pass on the imputed truth",
                   "fractions": FRACTIONS, "epsilons": EPS_LIST, "results": rows}, f, indent=1)
    print(f"\nwrote {len(rows)} rows -> {OUT_FILE} in {time.time() - t0:.1f}s")
