"""
Synthetic true-signal generator for the hierarchical DP experiments.

Produces the ground-truth node counts x that satisfy hierarchical consistency
(parent == sum of children), BEFORE any noise. Noise injection is separate.

Construction is TOP-DOWN: root holds a total, each node splits its mass among
its children down to the leaves. Two independent knobs:
  - leaf_mean  -> SCALE. Root total = num_leaves * leaf_mean, so average leaf
    value is leaf_mean regardless of distribution. (Sets the SNR once noised.)
  - signal type + parameter -> DIFFERENCE STRUCTURE among leaves.

Types: uniform (even splits, no rng), smooth (s=0.8 even + rng), random (s=0),
sparse (fraction p=0.3 of leaves hot, rest ~0). Values are floats (continuous
allocation; real releases round, orthogonal to the studied post-processing).

Running this file directly generates/checks a signal reference archive:
  7 balanced specs x 4 types x {1,100} leaf means x 1 fixed seed = 56 signals,
  written to SIGNAL_DIR. Floats are compared with a tolerance (np.allclose).
"""
from __future__ import annotations
import json
import os
from typing import List
import numpy as np

from hierarchy_data_generator import Hierarchy, create_balanced_tree, create_unbalanced_tree
from hierarchy_data_generator import SYNTHETIC_SPECS, UNBALANCED_SPECS    # reuse the spec list

__all__ = ["generate_signal", "DEFAULT_SMOOTHNESS", "DEFAULT_SPARSITY"]

DEFAULT_SMOOTHNESS = 0.8
DEFAULT_SPARSITY = 0.3


def _children_in_id_order(h: Hierarchy, node: int):
    return sorted(h.graph.successors(node))


def _split_weights(kind, k, rng, smoothness):
    if k == 1:
        return np.array([1.0])
    if kind == "uniform":
        return np.full(k, 1.0 / k)
    s = 0.0 if kind == "random" else smoothness
    rand = rng.dirichlet(np.ones(k))
    w = s * (1.0 / k) + (1.0 - s) * rand
    return w / w.sum()


def generate_signal(hierarchy, signal_type="smooth", leaf_mean=1.0, seed=0,
                    smoothness=DEFAULT_SMOOTHNESS, sparsity=DEFAULT_SPARSITY):
    """Consistent ground-truth x, shape (num_nodes,), node order 0..n-1."""
    if signal_type not in ("uniform", "smooth", "random", "sparse"):
        raise ValueError(f"unknown signal_type {signal_type!r}.")
    rng = np.random.default_rng(seed)
    n = hierarchy.num_nodes
    leaves = hierarchy.leaves
    total = len(leaves) * float(leaf_mean)
    x = np.zeros(n)
    x[hierarchy.root] = total

    if signal_type == "sparse":
        m = len(leaves)
        n_hot = max(1, int(round(sparsity * m)))
        hot = rng.choice(m, size=n_hot, replace=False)
        leaf_vals = np.zeros(m)
        leaf_vals[hot] = rng.random(n_hot) + 0.1
        leaf_vals = leaf_vals / leaf_vals.sum() * total
        for j, lf in enumerate(leaves):
            x[lf] = leaf_vals[j]
        for node in sorted(range(n), key=lambda u: -hierarchy.levels[u]):
            if hierarchy.graph.out_degree(node) > 0:
                x[node] = sum(x[c] for c in hierarchy.graph.successors(node))
        return x

    for node in sorted(range(n), key=lambda u: hierarchy.levels[u]):  # root first
        kids = _children_in_id_order(hierarchy, node)
        if not kids:
            continue
        w = _split_weights(signal_type, len(kids), rng, smoothness)
        for child, wi in zip(kids, w):
            x[child] = x[node] * wi
    return x


# ----------------------------------------------------------------------------
# Signal reference archive
# ----------------------------------------------------------------------------
SIGNAL_DIR = "../synthetic_data/signal_balanced_tree"
SIGNAL_UNBAL_DIR = "../synthetic_data/signal_unbalanced_tree"
SIGNAL_TYPES = ["uniform", "smooth", "random", "sparse"]
LEAF_MEANS = [1, 100]
SIGNAL_SEED = 0
_RTOL, _ATOL = 1e-9, 1e-9


def _signal_name(spec_name, signal_type, leaf_mean, seed):
    return f"signal_{spec_name[len('tree_'):]}_{signal_type}_mean{leaf_mean}_seed{seed}.json"


def generate_or_check_signals(out_dir: str = SIGNAL_DIR) -> None:
    os.makedirs(out_dir, exist_ok=True)
    print(f"True signals -> {out_dir}")
    n_written = n_checked = 0
    for depth, bf, spec_name in SYNTHETIC_SPECS:
        h = create_balanced_tree(depth=depth, branching_factor=bf)
        for st in SIGNAL_TYPES:
            for lm in LEAF_MEANS:
                x = generate_signal(h, signal_type=st, leaf_mean=lm, seed=SIGNAL_SEED)
                rec = {
                    "spec": {"depth": depth, "branching_factor": bf},
                    "signal_type": st,
                    "leaf_mean": lm,
                    "seed": SIGNAL_SEED,
                    "num_nodes": int(h.num_nodes),
                    "leaf_mean_actual": float(x[np.array(h.leaves)].mean()),
                    "x": [float(v) for v in x],
                }
                path = os.path.join(out_dir, _signal_name(spec_name, st, lm, SIGNAL_SEED))
                if os.path.exists(path):
                    with open(path) as f:
                        saved = json.load(f)
                    # metadata must match exactly; the float array uses a tolerance
                    meta_keys = ["spec", "signal_type", "leaf_mean", "seed", "num_nodes"]
                    if any(saved.get(k) != rec[k] for k in meta_keys):
                        raise AssertionError(f"REFERENCE MISMATCH (metadata) in {path}.")
                    if not np.allclose(saved["x"], rec["x"], rtol=_RTOL, atol=_ATOL):
                        md = np.max(np.abs(np.array(saved["x"]) - np.array(rec["x"])))
                        raise AssertionError(
                            f"REFERENCE MISMATCH (signal) in {path}: max|Δ|={md:.2e}.")
                    n_checked += 1
                else:
                    with open(path, "w") as f:
                        json.dump(rec, f, indent=1)
                    n_written += 1
    print(f"  wrote {n_written}, checked {n_checked} "
          f"(7 specs x 4 types x 2 means = {7*4*2} signals)")
    
    #unbalanced trees -> unbalanced_dir
    if UNBALANCED_SPECS:
        os.makedirs(SIGNAL_UNBAL_DIR, exist_ok=True)
        print(f"True signals (unbalanced) -> {SIGNAL_UNBAL_DIR}")
        uw = uc = 0
        for uname, edges in UNBALANCED_SPECS:
            h = create_unbalanced_tree(edges, root=0)
            for st in SIGNAL_TYPES:
                for lm in LEAF_MEANS:
                    x = generate_signal(h, signal_type=st, leaf_mean=lm, seed=SIGNAL_SEED)
                    rec = {
                        "spec": {"kind": "unbalanced", "name": uname},
                        "signal_type": st, "leaf_mean": lm, "seed": SIGNAL_SEED,
                        "num_nodes": int(h.num_nodes),
                        "leaf_mean_actual": float(x[np.array(h.leaves)].mean()),
                        "x": [float(v) for v in x],
                    }
                    fn = f"signal_{uname[len('tree_'):]}_{st}_mean{lm}_seed{SIGNAL_SEED}.json"
                    path = os.path.join(SIGNAL_UNBAL_DIR, fn)
                    if os.path.exists(path):
                        with open(path) as f:
                            saved = json.load(f)
                        mk = ["spec", "signal_type", "leaf_mean", "seed", "num_nodes"]
                        if any(saved.get(k) != rec[k] for k in mk):
                            raise AssertionError(f"REFERENCE MISMATCH (metadata) in {path}.")
                        if not np.allclose(saved["x"], rec["x"], rtol=_RTOL, atol=_ATOL):
                            raise AssertionError(f"REFERENCE MISMATCH (signal) in {path}.")
                        uc += 1
                    else:
                        with open(path, "w") as f:
                            json.dump(rec, f, indent=1)
                        uw += 1
        print(f"  unbalanced: wrote {uw}, checked {uc} "
              f"({len(UNBALANCED_SPECS)} x 4 x 2 = {len(UNBALANCED_SPECS)*4*2} signals)")


if __name__ == "__main__":
    h = create_balanced_tree(depth=6, branching_factor=3)
    idx_leaf = np.array(h.leaves)
    print(f"Tree (3,5): {h.num_nodes} nodes, {len(h.leaves)} leaves\n")
    print(f"{'type':<9}{'mean=1 std':>12}{'mean=100 std':>14}{'consist err':>14}")
    for st in SIGNAL_TYPES:
        x1 = generate_signal(h, st, 1, 0)
        x100 = generate_signal(h, st, 100, 0)
        err = max(abs(x1[u] - sum(x1[c] for c in h.graph.successors(u)))
                  for u in h.internal_nodes)
        print(f"{st:<9}{x1[idx_leaf].std():>12.4f}{x100[idx_leaf].std():>14.3f}{err:>14.1e}")
        assert abs(x1[idx_leaf].mean() - 1.0) < 1e-9
        assert abs(x100[idx_leaf].mean() - 100.0) < 1e-7
    # scale check: mean=100 signal is exactly 100x the mean=1 signal (same seed)
    a = generate_signal(h, "smooth", 1, 0)
    b = generate_signal(h, "smooth", 100, 0)
    assert np.allclose(b, 100 * a), "leaf_mean should scale the signal linearly"
    print("\nGenerating / checking signal archive:")
    generate_or_check_signals()
    print("Done.")