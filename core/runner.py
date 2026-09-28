"""
Experiment runner -- the top-level assembly.

Reads the pre-generated DP releases, runs every registered method on every
(tree, signal_type, leaf_mean, epsilon) configuration, and writes a structured
results JSON (per-node MSE mean/std and improvement-over-Hay per method). It
does NOT plot: producing figures from results.json is a separate step.

It consumes data only; no other module needs to change. The oracle spectral
method legitimately needs the true signal, so it is bound per configuration via
make_spectral_oracle_method and added alongside the registry.

Run:  python runner.py                          # WPES setting: Hay two-pass, all methods incl. spectral
      python runner.py --exact-ls --no-spectral  # CCECE setting: exact LS projection, no spectral (minutes)
      python runner.py --quick                   # tiny subset to check the pipeline
Output: results/synthetic_data_results/results.json          (Hay two-pass)
        results/synthetic_data_results/results_exactls.json  (--exact-ls)
Balanced trees give identical numbers under both projections (two-pass == exact
LS there, bit for bit); only the unbalanced trees differ.
"""
from __future__ import annotations
import json
import os
import time
import numpy as np

from hierarchy_data_generator import (create_balanced_tree, create_unbalanced_tree,
                                      SYNTHETIC_SPECS, UNBALANCED_SPECS)
from signal_generator import generate_signal, SIGNAL_TYPES, LEAF_MEANS, SIGNAL_SEED
from laplace_noise_injection import EPSILONS, NUM_TRIALS
import baseline, ablations, spectral          # importing registers all methods
from baseline import METHODS
from spectral import make_spectral_oracle_method
import projection as _proj
# Cache the graph Fourier basis per tree: spectral methods recompute the
# eigendecomposition on every call, but it only depends on the tree's structure.
# Key the cache by (num_nodes, sorted edges) -- a STABLE identity. (Keying by
# id(graph) is a bug: CPython reuses memory addresses after garbage collection,
# so a later tree can collide with an earlier one and receive the wrong basis.)
_orig_gft = spectral._gft
_GFT_CACHE = {}
def _gft_memo(h):
    key = (h.num_nodes, tuple(sorted(h.graph.edges())))
    if key not in _GFT_CACHE:
        _GFT_CACHE[key] = _orig_gft(h)
    return _GFT_CACHE[key]
spectral._gft = _gft_memo

DP_BAL_DIR = "../synthetic_data/dp_balanced_tree"
DP_UNBAL_DIR = "../synthetic_data/dp_unbalanced_tree"
RESULTS_DIR = "../results/synthetic_data_results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "results.json")
RESULTS_FILE_LS = os.path.join(RESULTS_DIR, "results_exactls.json")


def results_path(exact_ls: bool = False) -> str:
    return RESULTS_FILE_LS if exact_ls else RESULTS_FILE


def _dp_filename(spec_name, st, lm, eps):
    e = str(eps).replace(".", "p")
    return f"dp_{spec_name[len('tree_'):]}_{st}_mean{lm}_eps{e}.json"


def _evaluate_config(h, x, Z, sigma2, with_spectral=True):
    """Run every method on all trials; return {method: (mse_mean, mse_std)}."""
    methods = {k: v for k, v in METHODS.items() if with_spectral or "Spectral" not in k}
    if with_spectral:
        methods["Spectral(oracle)"] = make_spectral_oracle_method(x)  # needs truth
    out = {}
    for name, fn in methods.items():
        errs = np.array([np.mean((fn(z, h, sigma2) - x) ** 2) for z in Z])
        out[name] = (float(errs.mean()), float(errs.std()))
    return out


def _run_group(specs, dp_dir, build_tree, kind, results, t0, quick=False,
               with_spectral=True):
    for spec in specs:
        if kind == "balanced":
            depth, bf, spec_name = spec
            h = build_tree(depth=depth, branching_factor=bf)
            spec_meta = {"kind": "balanced", "depth": depth, "branching_factor": bf}
        else:
            spec_name, edges = spec
            h = build_tree(edges, root=0)
            spec_meta = {"kind": "unbalanced", "name": spec_name}

        sig_types = ["smooth"] if quick else SIGNAL_TYPES
        means = [1] if quick else LEAF_MEANS
        eps_list = [0.1, 0.5, 1.0] if quick else EPSILONS
        for st in sig_types:
            for lm in means:
                x = generate_signal(h, st, lm, SIGNAL_SEED)
                for eps in eps_list:
                    path = os.path.join(dp_dir, _dp_filename(spec_name, st, lm, eps))
                    with open(path) as f:
                        rec = json.load(f)
                    Z = np.array(rec["z"])
                    sigma2 = rec["sigma2"]
                    per_method = _evaluate_config(h, x, Z, sigma2, with_spectral)

                    hay_mean = per_method["Hay"][0]
                    methods_block = {}
                    for name, (m, s) in per_method.items():
                        methods_block[name] = {
                            "mse_mean": m,
                            "mse_std": s,
                            "improve_over_hay_pct":
                                (100.0 * (hay_mean - m) / hay_mean
                                 if hay_mean > 0 else 0.0),
                        }
                    results.append({
                        "spec": spec_meta,
                        "spec_name": spec_name,
                        "signal_type": st,
                        "leaf_mean": lm,
                        "epsilon": eps,
                        "num_nodes": int(h.num_nodes),
                        "num_trials": int(Z.shape[0]),
                        "sigma2": sigma2,
                        "methods": methods_block,
                    })
        print(f"  [{time.time()-t0:6.1f}s] done {kind} {spec_name} "
              f"({len(SIGNAL_TYPES)*len(LEAF_MEANS)*len(EPSILONS)} configs)")


def run_all(balanced=True, unbalanced=True, quick=False, exact_ls=False,
            with_spectral=True) -> str:
    """Run experiments and write the results JSON (path returned).
    quick=True runs only a tiny subset (main tree, smooth, mean1, a few eps) to
    verify the pipeline end-to-end in seconds before committing to the full run.
    exact_ls=True projects with the exact LS projector instead of Hay two-pass
    and writes results_exactls.json; with_spectral=False skips the (slow)
    spectral references."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    _proj.set_projection("ls" if exact_ls else "hay")
    out_file = results_path(exact_ls)
    results: list = []
    t0 = time.time()
    names = [k for k in METHODS if with_spectral or "Spectral" not in k]
    print(f"Methods: {names}" + (" + Spectral(oracle)" if with_spectral else "")
          + f" | projection: {'exact LS' if exact_ls else 'Hay two-pass'}"
          + ("  [QUICK SUBSET]" if quick else ""))
    bal_specs = [SYNTHETIC_SPECS[0]] if quick else SYNTHETIC_SPECS
    if balanced:
        print("Balanced trees:")
        _run_group(bal_specs, DP_BAL_DIR, create_balanced_tree,
                   "balanced", results, t0, quick=quick, with_spectral=with_spectral)
    if unbalanced and UNBALANCED_SPECS and not quick:
        print("Unbalanced trees:")
        _run_group(UNBALANCED_SPECS, DP_UNBAL_DIR, create_unbalanced_tree,
                   "unbalanced", results, t0, with_spectral=with_spectral)

    payload = {
        "config": {
            "methods": names + (["Spectral(oracle)"] if with_spectral else []),
            "projection": "exact_ls" if exact_ls else "hay_two_pass",
            "signal_types": SIGNAL_TYPES,
            "leaf_means": LEAF_MEANS,
            "epsilons": EPSILONS,
            "num_trials": NUM_TRIALS,
            "signal_seed": SIGNAL_SEED,
        },
        "results": results,
    }
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"\nwrote {len(results)} configurations -> {out_file} "
          f"in {time.time()-t0:.1f}s")
    return out_file


def summarize(results_file: str = RESULTS_FILE) -> None:
    """Print a compact view of the headline configuration as a sanity check."""
    payload = json.load(open(results_file))
    for r in payload["results"]:
        if (r["spec_name"] == "tree_d6_b3" and r["signal_type"] == "smooth"
                and r["leaf_mean"] == 1 and r["epsilon"] == 0.5):
            print("\nHeadline (tree_d6_b3, smooth, mean1, eps0.5) per-node MSE:")
            order = sorted(r["methods"].items(), key=lambda kv: kv[1]["mse_mean"])
            for name, d in order:
                print(f"  {name:<18} {d['mse_mean']:>9.2f} "
                      f"(improve vs Hay {d['improve_over_hay_pct']:>6.1f}%)")
            break


if __name__ == "__main__":
    import sys
    quick = "--quick" in sys.argv
    exact_ls = "--exact-ls" in sys.argv
    with_spectral = "--no-spectral" not in sys.argv
    out = run_all(balanced=True, unbalanced=True, quick=quick, exact_ls=exact_ls,
                  with_spectral=with_spectral)
    summarize(out)