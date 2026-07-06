"""
Stratified (cell-type) evaluation on the DDHC-A groups -- the data behind the
paper's Section `sec:stratified`.

For each selected population group, per-node squared error is split into three
strata -- populated leaves (x > 0), empty leaves (x == 0), and internal
nodes -- for each granularity at each epsilon. Noise draws use the SAME seeds
as run_ddhca (1_300_000 + ei*10_000 + trial), so the aggregate MSE recomputed
here must match results/census_ddhca_results/ddhca_results_pop*.json; this is
asserted, which cross-validates both scripts.

Also measures, once per census tree, the deviation of Hay's two-pass from the
exact least-squares projection (the Method-section remark about unbalanced
trees), on the same draws.

Output: ../results/census_ddhca_results/stratified_results.json
Run:    python stratified_eval.py          (from core/)
"""
from __future__ import annotations
import json
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

OUT_FILE = os.path.join(CENSUS_DDHCA_RESULTS_DIR, "stratified_results.json")

# granularity name here -> method, and its name in the ddhca results files
METHODS = {
    "Laplace": (lambda z, h, s2: z,            "Laplace"),
    "Hay":     (lambda z, h, s2: hay_two_pass(z, h), "Hay"),
    "Scalar":  (scalar_shrinkage,               "Scalar"),
    "Level":   (level_wise_shrinkage,           "Ours"),
    "Node":    (per_node_shrinkage,             "PerNode"),
}


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

        block = {}
        for name, (_, pub_name) in METHODS.items():
            mse_all = float(np.mean(acc[name]["all"]))
            # cross-check against the published aggregate (same draws)
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
            }
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
              f"(x{block['Node']['mse_present']/hp:.1f})   [aggregate check ok]")


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
    hay_ls = measure_hay_vs_ls()
    os.makedirs(CENSUS_DDHCA_RESULTS_DIR, exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump({"strata": ["present", "empty", "internal"],
                   "note": "same noise draws as run_ddhca; aggregate MSE "
                           "cross-checked against ddhca_results_pop*.json",
                   "hay_vs_ls": hay_ls,
                   "results": results}, f, indent=1)
    print(f"\nwrote {len(results)} (group, eps) blocks -> {OUT_FILE} "
          f"in {time.time()-t0:.1f}s")
