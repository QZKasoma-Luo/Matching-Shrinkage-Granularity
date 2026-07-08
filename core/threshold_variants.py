"""
Per-node threshold variants on the DDHC-A groups -- the data behind the
paper's remark that the garrote (threshold sigma) is one point on a threshold
axis: universal-threshold soft/hard rules clean empty cells further and raise
aggregate gains at large epsilon, at the cost of populated-cell error in the
moderate-epsilon regime.

Rules compared, all zero-budget post-processing followed by Hay projection,
on the SAME archived-seed draws as run_ddhca:
  Node  (garrote):  c_v = max(0, 1 - sigma^2/z_v^2)          threshold sigma
  SoftU:            sign(z) max(|z| - lambda, 0),   lambda = sigma sqrt(2 ln n)
  HardU:            z * 1{|z| > lambda},            lambda = sigma sqrt(2 ln n)

Output: ../results/census_ddhca_results/threshold_variants.json
Run:    python threshold_variants.py          (from core/)
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
from projection import hay_two_pass
from ablations import per_node_shrinkage

OUT_FILE = os.path.join(CENSUS_DDHCA_RESULTS_DIR, "threshold_variants.json")


def _lam(h, s2):
    return math.sqrt(s2) * math.sqrt(2.0 * math.log(h.num_nodes))


def soft_universal(z, h, s2):
    lam = _lam(h, s2)
    return hay_two_pass(np.sign(z) * np.maximum(np.abs(z) - lam, 0.0), h)


def hard_universal(z, h, s2):
    lam = _lam(h, s2)
    return hay_two_pass(z * (np.abs(z) > lam), h)


METHODS = {
    "Hay":   lambda z, h, s2: hay_two_pass(z, h),
    "Node":  per_node_shrinkage,
    "SoftU": soft_universal,
    "HardU": hard_universal,
}


if __name__ == "__main__":
    t0 = time.time()
    results = []
    for pg in SELECTED_POPGROUPS:
        arch = os.path.join(CENSUS_DDHCA_HIER_DIR,
                            f"T01001_census_county_2020_pop{pg}.json")
        h, x, rec = load_hierarchy_archive(arch)
        leaf = h.level_mask(h.depth - 1)
        pres, emp = leaf & (x > 0), leaf & (x == 0)
        for ei, eps in enumerate(EPSILONS):
            s2 = noise_variance(h, eps)
            acc = {m: {"all": [], "present": [], "empty": []} for m in METHODS}
            for t in range(NUM_TRIALS):
                z = add_laplace_noise(x, h, eps,
                                      np.random.default_rng(1_300_000 + ei * 10_000 + t))
                for m, fn in METHODS.items():
                    d2 = (fn(z, h, s2) - x) ** 2
                    acc[m]["all"].append(d2.mean())
                    acc[m]["present"].append(d2[pres].mean())
                    acc[m]["empty"].append(d2[emp].mean())
            hay = float(np.mean(acc["Hay"]["all"]))
            block = {m: {"mse_all": float(np.mean(acc[m]["all"])),
                         "mse_present": float(np.mean(acc[m]["present"])),
                         "mse_empty": float(np.mean(acc[m]["empty"])),
                         "improve_over_hay_pct": 100 * (hay - float(np.mean(acc[m]["all"]))) / hay}
                     for m in METHODS}
            results.append({"popgroup": pg,
                            "popgroup_label": rec["popgroup_label"],
                            "epsilon": eps, "sigma2": s2,
                            "num_trials": NUM_TRIALS, "methods": block})
            print(f"  [{time.time()-t0:6.1f}s] pop{pg} eps={eps:>5}: "
                  f"agg improvement Node {block['Node']['improve_over_hay_pct']:+.1f}% "
                  f"SoftU {block['SoftU']['improve_over_hay_pct']:+.1f}% "
                  f"HardU {block['HardU']['improve_over_hay_pct']:+.1f}%  | "
                  f"present MSE Hay {block['Hay']['mse_present']:.1f} "
                  f"Node {block['Node']['mse_present']:.1f} "
                  f"HardU {block['HardU']['mse_present']:.1f}")
    with open(OUT_FILE, "w") as f:
        json.dump({"note": "same archived-seed draws as run_ddhca; "
                           "lambda = sigma*sqrt(2 ln n), n = num_nodes",
                   "results": results}, f, indent=1)
    print(f"\nwrote {len(results)} (group, eps) blocks -> {OUT_FILE} "
          f"in {time.time()-t0:.1f}s")
