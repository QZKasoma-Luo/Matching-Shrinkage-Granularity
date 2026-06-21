"""
Laplace noise injection: read a true signal x, add calibrated Laplace noise to
every node to produce a DP release z = x + Lap(0, b), b = sensitivity/epsilon.

This file ONLY injects noise. It reads true signals produced by
signal_generator and writes noised releases. Pure epsilon-DP.

Key facts:
  - b = h.sensitivity / epsilon  (sensitivity from the hierarchy's one definition)
  - per-node variance sigma^2 = 2 b^2  (numpy's `scale` is b, NOT the variance)
  - reproducible: every draw goes through np.random.default_rng(noise_seed)

Archive layout (avoids ~20k tiny files): for each
(spec, signal_type, leaf_mean, epsilon) we pack all NUM_TRIALS noise seeds into
ONE file as a (NUM_TRIALS, num_nodes) array of z.  ->  7*4*2*7 = 392 files.
Floats are checked with a tolerance (np.allclose) for the golden-master check.
Note: epsilon 0.01 and 10 are included only to draw the "operating envelope"
boundary figure, not for the main tables.
"""
from __future__ import annotations
import json
import os
import numpy as np

from hierarchy_data_generator import create_balanced_tree, create_unbalanced_tree           
from hierarchy_data_generator import SYNTHETIC_SPECS, UNBALANCED_SPECS    # reuse the spec list
from signal_generator import (generate_signal, SIGNAL_TYPES, LEAF_MEANS,
                              SIGNAL_SEED)

__all__ = ["noise_scale", "noise_variance", "add_laplace_noise"]

EPSILONS = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]   # 0.01 & 10: envelope figure only
NUM_TRIALS = 50
DP_DIR = "../synthetic_data/dp_balanced_tree"
DP_UNBAL_DIR = "../synthetic_data/dp_unbalanced_tree"
_RTOL, _ATOL = 1e-9, 1e-9


def noise_scale(hierarchy, epsilon):
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0.")
    return hierarchy.sensitivity / epsilon


def noise_variance(hierarchy, epsilon):
    b = noise_scale(hierarchy, epsilon)
    return 2.0 * b * b


def add_laplace_noise(counts, hierarchy, epsilon, rng):
    """z = x + Lap(0, b) per node. counts shape (num_nodes,), node order 0..n-1."""
    counts = np.asarray(counts, dtype=float)
    if counts.shape != (hierarchy.num_nodes,):
        raise ValueError(f"counts must be ({hierarchy.num_nodes},), got {counts.shape}.")
    b = noise_scale(hierarchy, epsilon)
    return counts + rng.laplace(0.0, b, hierarchy.num_nodes)


def _noise_seed(spec_idx, type_idx, mean_idx, eps_idx, trial):
    """Deterministic, collision-free seed per (config, trial)."""
    return ((((spec_idx * 10 + type_idx) * 10 + mean_idx) * 10 + eps_idx) * 1000
            + trial)


def _dp_name(spec_name, st, lm, eps):
    e = str(eps).replace(".", "p")
    return f"dp_{spec_name[len('tree_'):]}_{st}_mean{lm}_eps{e}.json"


def generate_or_check_dp(out_dir: str = DP_DIR) -> None:
    os.makedirs(out_dir, exist_ok=True)
    print(f"DP releases -> {out_dir}")
    n_written = n_checked = 0
    for si, (depth, bf, spec_name) in enumerate(SYNTHETIC_SPECS):
        h = create_balanced_tree(depth=depth, branching_factor=bf)
        for ti, st in enumerate(SIGNAL_TYPES):
            for mi, lm in enumerate(LEAF_MEANS):
                x = generate_signal(h, signal_type=st, leaf_mean=lm, seed=SIGNAL_SEED)
                for ei, eps in enumerate(EPSILONS):
                    Z = np.empty((NUM_TRIALS, h.num_nodes))
                    for trial in range(NUM_TRIALS):
                        rng = np.random.default_rng(_noise_seed(si, ti, mi, ei, trial))
                        Z[trial] = add_laplace_noise(x, h, eps, rng)
                    rec = {
                        "spec": {"depth": depth, "branching_factor": bf},
                        "signal_type": st, "leaf_mean": lm, "epsilon": eps,
                        "num_trials": NUM_TRIALS, "num_nodes": int(h.num_nodes),
                        "sensitivity": int(h.sensitivity),
                        "sigma2": float(noise_variance(h, eps)),
                        "signal_seed": SIGNAL_SEED,
                        "z": Z.tolist(),   # shape (num_trials, num_nodes)
                    }
                    path = os.path.join(out_dir, _dp_name(spec_name, st, lm, eps))
                    if os.path.exists(path):
                        with open(path) as f:
                            saved = json.load(f)
                        meta = ["spec", "signal_type", "leaf_mean", "epsilon",
                                "num_trials", "num_nodes", "sensitivity"]
                        if any(saved.get(k) != rec[k] for k in meta):
                            raise AssertionError(f"REFERENCE MISMATCH (metadata) {path}.")
                        if not np.allclose(saved["z"], rec["z"], rtol=_RTOL, atol=_ATOL):
                            md = np.max(np.abs(np.array(saved["z"]) - np.array(rec["z"])))
                            raise AssertionError(
                                f"REFERENCE MISMATCH (z) {path}: max|Δ|={md:.2e}.")
                        n_checked += 1
                    else:
                        with open(path, "w") as f:
                            json.dump(rec, f)
                        n_written += 1
    total = len(SYNTHETIC_SPECS) * len(SIGNAL_TYPES) * len(LEAF_MEANS) * len(EPSILONS)
    print(f"  balanced: wrote {n_written}, checked {n_checked} "
          f"({len(SYNTHETIC_SPECS)}x{len(SIGNAL_TYPES)}x{len(LEAF_MEANS)}x{len(EPSILONS)} "
          f"= {total} files, {NUM_TRIALS} trials each)")
    
    # ---- unbalanced trees ----
    if UNBALANCED_SPECS:
        os.makedirs(DP_UNBAL_DIR, exist_ok=True)
        print(f"DP releases (unbalanced) -> {DP_UNBAL_DIR}")
        uw = uc = 0
        for ui, (uname, edges) in enumerate(UNBALANCED_SPECS):
            h = create_unbalanced_tree(edges, root=0)
            for ti, st in enumerate(SIGNAL_TYPES):
                for mi, lm in enumerate(LEAF_MEANS):
                    x = generate_signal(h, signal_type=st, leaf_mean=lm, seed=SIGNAL_SEED)
                    for ei, eps in enumerate(EPSILONS):
                        Z = np.empty((NUM_TRIALS, h.num_nodes))
                        for trial in range(NUM_TRIALS):
                            # offset seeds by 500000 to avoid collision with balanced
                            rng = np.random.default_rng(
                                500000 + _noise_seed(ui, ti, mi, ei, trial))
                            Z[trial] = add_laplace_noise(x, h, eps, rng)
                        rec = {
                            "spec": {"kind": "unbalanced", "name": uname},
                            "signal_type": st, "leaf_mean": lm, "epsilon": eps,
                            "num_trials": NUM_TRIALS, "num_nodes": int(h.num_nodes),
                            "sensitivity": int(h.sensitivity),
                            "sigma2": float(noise_variance(h, eps)),
                            "signal_seed": SIGNAL_SEED, "z": Z.tolist(),
                        }
                        e = str(eps).replace(".", "p")
                        fn = f"dp_{uname[len('tree_'):]}_{st}_mean{lm}_eps{e}.json"
                        path = os.path.join(DP_UNBAL_DIR, fn)
                        if os.path.exists(path):
                            with open(path) as f:
                                saved = json.load(f)
                            mk = ["spec","signal_type","leaf_mean","epsilon",
                                  "num_trials","num_nodes","sensitivity"]
                            if any(saved.get(k) != rec[k] for k in mk):
                                raise AssertionError(f"REFERENCE MISMATCH (meta) {path}.")
                            if not np.allclose(saved["z"], rec["z"], rtol=_RTOL, atol=_ATOL):
                                raise AssertionError(f"REFERENCE MISMATCH (z) {path}.")
                            uc += 1
                        else:
                            with open(path, "w") as f:
                                json.dump(rec, f)
                            uw += 1
                            
    ut = len(UNBALANCED_SPECS)*len(SIGNAL_TYPES)*len(LEAF_MEANS)*len(EPSILONS)
    print(f"  unbalanced: wrote {uw}, checked {uc} "
        f"({len(UNBALANCED_SPECS)}x{len(SIGNAL_TYPES)}x{len(LEAF_MEANS)}x{len(EPSILONS)} "
        f"= {ut} files, {NUM_TRIALS} trials each)")


if __name__ == "__main__":
    h = create_balanced_tree(depth=6, branching_factor=3)
    x = generate_signal(h, "smooth", 1, SIGNAL_SEED)

    # calibration
    assert noise_scale(h, 0.5) == 12.0 and noise_variance(h, 0.5) == 288.0
    print(f"eps=0.5: b={noise_scale(h,0.5):.1f}, sigma^2={noise_variance(h,0.5):.1f}  (b=12, 288)")

    # reproducibility: same seed -> identical z
    z1 = add_laplace_noise(x, h, 0.5, np.random.default_rng(0))
    z2 = add_laplace_noise(x, h, 0.5, np.random.default_rng(0))
    assert np.array_equal(z1, z2)

    # empirical: mean(z) ~ x, var(z - x) ~ sigma^2, over many seeds
    diffs = np.concatenate([add_laplace_noise(x, h, 0.5, np.random.default_rng(s)) - x
                            for s in range(300)])
    print(f"empirical noise var ~ {diffs.var():.1f} (theory 288), mean ~ {diffs.mean():.3f} (0)")

    # noise breaks consistency
    root = h.root; kids = list(h.graph.successors(root))
    idx = {u: i for i, u in enumerate(h.graph.nodes)}
    print(f"after noise: root {z1[idx[root]]:.1f} vs sum(children) "
          f"{sum(z1[idx[c]] for c in kids):.1f} (broken, as expected)")

    # tiny archive smoke test in a temp dir (2 specs only), then check pass
    # import tempfile, shutil
    # tmp = tempfile.mkdtemp()
    # saved_specs = SYNTHETIC_SPECS[:]
    # try:
    #     import laplace_noise_injection as M
    #     M.SYNTHETIC_SPECS[:] = SYNTHETIC_SPECS[:1]   # just the main tree
    #     M.EPSILONS[:] = [0.5, 1.0]                    # just 2 eps
    #     M.generate_or_check_dp(tmp)                   # writes
    #     M.generate_or_check_dp(tmp)                   # checks -> must pass
    #     nfiles = len(os.listdir(tmp))
    #     print(f"archive smoke test: {nfiles} files written then re-checked OK "
    #           f"(1 spec x 4 types x 2 means x 2 eps = 16)")
    # finally:
    #     M.SYNTHETIC_SPECS[:] = saved_specs
    #     M.EPSILONS[:] = EPSILONS
    #     shutil.rmtree(tmp)
    # print("Self-test passed.")
    generate_or_check_dp()