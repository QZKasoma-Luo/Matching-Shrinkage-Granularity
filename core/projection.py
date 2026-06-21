"""
Consistency projection for hierarchical DP releases.

Two implementations of "project the noisy counts z back onto the consistency
subspace (parent == sum of children)":

  ls_projection(z, h)  -- exact orthogonal least-squares projection,
                          x_hat = M (M^T M)^{-1} M^T z, where M is the
                          leaf-incidence matrix. No implementation freedom;
                          this is the GROUND TRUTH. Works on any tree.

  hay_two_pass(z, h)   -- Hay et al. (2010): bottom-up inverse-variance
                          weighted averaging, then top-down equal-share mean
                          consistency. The fast O(n) algorithm the literature
                          uses as baseline. Works on any tree.

Verification design ("validate the baseline with a theorem"): on BALANCED trees
hay_two_pass == ls_projection bit-for-bit. On UNBALANCED trees the equal-share
top-down is not variance-optimal across differently shaped sibling subtrees, so
Hay deviates slightly from exact LS -- we MEASURE that deviation rather than
hide it.

Node variance is uniform (each node gets i.i.d. noise of the same variance), so
it cancels and the projection is purely structural; we set sigma^2 = 1 in the
recursion WLOG.
"""
from __future__ import annotations
import numpy as np

from hierarchy_data_generator import Hierarchy 

__all__ = ["ls_projection", "hay_two_pass", "leaf_incidence_matrix",
           "projection_matrix"]


def leaf_incidence_matrix(h: Hierarchy):
    """M (num_nodes x num_leaves): M[i, j] = 1 iff leaf j is a descendant of
    node i (or i is leaf j). Columns ordered by sorted leaf id."""
    leaves = sorted(h.leaves)
    leaf_col = {lf: j for j, lf in enumerate(leaves)}
    M = np.zeros((h.num_nodes, len(leaves)))
    for lf in leaves:
        j = leaf_col[lf]
        u = lf
        while True:
            M[u, j] = 1.0
            preds = list(h.graph.predecessors(u))
            if not preds:
                break
            u = preds[0]
    return M, leaves


def projection_matrix(h: Hierarchy) -> np.ndarray:
    """Orthogonal projector P = M (M^T M)^{-1} M^T onto the consistency subspace.
    Dense; for analysis/verification only (trace(P) = number of leaves)."""
    M, _ = leaf_incidence_matrix(h)
    return M @ np.linalg.solve(M.T @ M, M.T)


def ls_projection(z: np.ndarray, h: Hierarchy) -> np.ndarray:
    """Exact LS consistency projection of z. Ground-truth, any tree."""
    z = np.asarray(z, dtype=float)
    if z.shape != (h.num_nodes,):
        raise ValueError(f"z must be ({h.num_nodes},), got {z.shape}.")
    M, _ = leaf_incidence_matrix(h)
    a = np.linalg.solve(M.T @ M, M.T @ z)   # leaf-value LS solution
    return M @ a


def hay_two_pass(z: np.ndarray, h: Hierarchy) -> np.ndarray:
    """Hay (2010) two-pass consistency. Any tree; == ls_projection on balanced."""
    z = np.asarray(z, dtype=float)
    if z.shape != (h.num_nodes,):
        raise ValueError(f"z must be ({h.num_nodes},), got {z.shape}.")
    n, G, root = h.num_nodes, h.graph, h.root
    children = {v: sorted(G.successors(v)) for v in range(n)}

    # bottom-up: inverse-variance combination of own measurement with children
    u = np.zeros(n)      # subtree BLUE estimate
    V = np.zeros(n)      # its variance (sigma^2 = 1)
    for v in sorted(range(n), key=lambda v: -h.levels[v]):   # leaves first
        C = children[v]
        if not C:
            u[v], V[v] = z[v], 1.0
        else:
            S = sum(u[c] for c in C)
            W = sum(V[c] for c in C)
            u[v] = (z[v] * W + S) / (1.0 + W)
            V[v] = W / (1.0 + W)

    # top-down: distribute parent's correction equally among children
    out = np.zeros(n)
    out[root] = u[root]
    for v in sorted(range(n), key=lambda v: h.levels[v]):     # root first
        if v == root:
            continue
        p = next(iter(G.predecessors(v)))
        C = children[p]
        correction = out[p] - sum(u[c] for c in C)
        out[v] = u[v] + correction / len(C)
    return out


def _max_consistency_violation(x, h):
    return max((abs(x[v] - sum(x[c] for c in h.graph.successors(v)))
                for v in h.internal_nodes), default=0.0)


if __name__ == "__main__":
    from hierarchy_data_generator import (create_balanced_tree,
                                          create_unbalanced_tree, SYNTHETIC_SPECS)
    from signal_generator import generate_signal
    from laplace_noise_injection import add_laplace_noise

    rng = np.random.default_rng(0)

    # ---- 1) Hay == LS bit-for-bit on every balanced spec ----
    print("Hay two-pass vs exact LS on balanced trees:")
    worst = 0.0
    for depth, bf, name in SYNTHETIC_SPECS:
        h = create_balanced_tree(depth=depth, branching_factor=bf)
        x = generate_signal(h, "smooth", 1, 0)
        z = add_laplace_noise(x, h, 0.5, np.random.default_rng(1))
        a, b = ls_projection(z, h), hay_two_pass(z, h)
        d = float(np.max(np.abs(a - b)))
        worst = max(worst, d)
        # both must be consistent
        assert _max_consistency_violation(a, h) < 1e-9
        assert _max_consistency_violation(b, h) < 1e-9
        print(f"  {name:<12} n={h.num_nodes:<4} max|Hay - LS| = {d:.2e}")
    print(f"  worst over all balanced specs: {worst:.2e}  (== 0 to numerical precision)")
    assert worst < 1e-9, "Hay must equal LS on balanced trees"

    # ---- 2) trace(P) == number of leaves (the sigma^2 * m floor) ----
    h = create_balanced_tree(depth=6, branching_factor=3)
    P = projection_matrix(h)
    print(f"\ntrace(P) = {np.trace(P):.4f}  (== num_leaves = {len(h.leaves)})")
    assert abs(np.trace(P) - len(h.leaves)) < 1e-6

    # ---- 3) Lemma 4: projection never increases L2 error ----
    x = generate_signal(h, "smooth", 1, 0)
    contract_ok = True
    for s in range(200):
        z = add_laplace_noise(x, h, 0.5, np.random.default_rng(s))
        if np.linalg.norm(ls_projection(z, h) - x) > np.linalg.norm(z - x) + 1e-9:
            contract_ok = False
            break
    print(f"\nLemma 4 (||Pz - x|| <= ||z - x||) held in 200/200 draws: {contract_ok}")
    assert contract_ok

    # projection reduces MSE on average
    raw = np.mean([np.mean((add_laplace_noise(x, h, 0.5, np.random.default_rng(s)) - x)**2)
                   for s in range(200)])
    proj = np.mean([np.mean((ls_projection(add_laplace_noise(x, h, 0.5,
                    np.random.default_rng(s)), h) - x)**2) for s in range(200)])
    print(f"mean per-node MSE: raw noise {raw:.1f} -> after projection {proj:.1f} "
          f"({100*(raw-proj)/raw:.0f}% lower)")

    # ---- 4) On an UNBALANCED tree, Hay deviates from exact LS (measured) ----
    edges = ([(0, i) for i in range(1, 31)]            # 30-wide fan-out at root
             + [(2, 31), (31, 32), (32, 33), (33, 34)]) # plus a deep chain
    hu = create_unbalanced_tree(edges, root=0)
    xu = np.zeros(hu.num_nodes)
    for v in sorted(range(hu.num_nodes), key=lambda v: -hu.levels[v]):
        kids = list(hu.graph.successors(v))
        xu[v] = 1.0 if not kids else sum(xu[c] for c in kids)
    zu = add_laplace_noise(xu, hu, 0.5, np.random.default_rng(3))
    au, bu = ls_projection(zu, hu), hay_two_pass(zu, hu)
    assert _max_consistency_violation(bu, hu) < 1e-9   # Hay still consistent
    print(f"\nUnbalanced tree ({hu.num_nodes} nodes): both consistent, but "
          f"max|Hay - LS| = {np.max(np.abs(au - bu)):.3f}  (Hay != exact LS here)")
    print(f"  MSE: LS {np.mean((au-xu)**2):.2f} vs Hay {np.mean((bu-xu)**2):.2f} "
          f"(deviation tiny relative to error)")
    print("\nSelf-test passed.")