"""
Level-wise shrinkage post-processing -- the core method.

Given a DP release z (noisy node counts) and the per-node noise variance sigma^2,
shrink each tree LEVEL toward zero by a closed-form positive-part coefficient,
then project back to consistency. Two lines of real work, parameter-free, zero
additional privacy cost (uses only z and the public sigma^2).

Per level l with n_l nodes and noisy values z_l:
    c_hat_l = max(0, 1 - n_l * sigma^2 / ||z_l||^2)
Root protection: levels with n_l <= ROOT_PROTECT_MAX nodes are left unshrunk
(c=1), because ||z_l||^2 is an unreliable energy estimate when the level has very
few nodes (a single sample at the root).

We take sigma^2 directly (not epsilon): the coefficient depends only on the
second moment of the noise, so the method is identical for Laplace or Gaussian
releases -- this is exactly the distribution-free property the paper proves.

Variant: shrink_to_mean shrinks each level toward its own mean instead of zero
(protects small areas; leaves level totals ~unchanged).
"""
from __future__ import annotations
import numpy as np

from hierarchy_data_generator import Hierarchy
from projection import hay_two_pass, ls_projection

__all__ = ["level_coefficients", "level_wise_shrinkage", "shrink_to_mean",
           "ROOT_PROTECT_MAX"]

ROOT_PROTECT_MAX = 3   # levels with <= this many nodes are not shrunk


def level_coefficients(z: np.ndarray, h: Hierarchy, sigma2: float,
                       root_protect_max: int = ROOT_PROTECT_MAX) -> np.ndarray:
    """Return the per-level shrinkage coefficients c_hat (length = h.depth)."""
    z = np.asarray(z, dtype=float)
    c = np.ones(h.depth)
    for ell in range(h.depth):
        mask = h.level_mask(ell)
        n_l = int(mask.sum())
        if n_l <= root_protect_max:
            c[ell] = 1.0
            continue
        energy = float(np.dot(z[mask], z[mask]))
        c[ell] = max(0.0, 1.0 - n_l * sigma2 / energy) if energy > 0 else 0.0
    return c


def _apply_level_scaling(z, h, c):
    """Multiply each node by its level's coefficient."""
    out = z.copy()
    for ell in range(h.depth):
        out[h.level_mask(ell)] *= c[ell]
    return out


def level_wise_shrinkage(z: np.ndarray, h: Hierarchy, sigma2: float,
                         root_protect_max: int = ROOT_PROTECT_MAX,
                         project=hay_two_pass) -> np.ndarray:
    """Shrink each level toward zero, then project for consistency.
    Returns the post-processed estimate x_hat (shape (num_nodes,))."""
    z = np.asarray(z, dtype=float)
    if z.shape != (h.num_nodes,):
        raise ValueError(f"z must be ({h.num_nodes},), got {z.shape}.")
    c = level_coefficients(z, h, sigma2, root_protect_max)
    return project(_apply_level_scaling(z, h, c), h)


def shrink_to_mean(z: np.ndarray, h: Hierarchy, sigma2: float,
                   root_protect_max: int = ROOT_PROTECT_MAX,
                   project=hay_two_pass) -> np.ndarray:
    """Variant: shrink each level toward its own mean (protects small areas).
    c'_l = max(0, 1 - (n_l - 1) sigma^2 / ||z_l - mean||^2); level means kept."""
    z = np.asarray(z, dtype=float)
    out = z.copy()
    for ell in range(h.depth):
        mask = h.level_mask(ell)
        n_l = int(mask.sum())
        vals = z[mask]
        if n_l <= root_protect_max:
            continue
        mu = vals.mean()
        energy = float(np.dot(vals - mu, vals - mu))
        c = max(0.0, 1.0 - (n_l - 1) * sigma2 / energy) if energy > 0 else 0.0
        out[mask] = mu + c * (vals - mu)
    return project(out, h)


if __name__ == "__main__":
    from hierarchy_data_generator import create_balanced_tree
    from signal_generator import generate_signal
    from laplace_noise_injection import add_laplace_noise, noise_variance

    h = create_balanced_tree(depth=6, branching_factor=3)

    # ---- 1) coefficient behaviour: root ~1, leaves shrink hard at low SNR ----
    x = generate_signal(h, "smooth", 1, 0)        # leaf mean 1 (low SNR)
    sig2 = noise_variance(h, 0.5)                  # 288
    z = add_laplace_noise(x, h, 0.5, np.random.default_rng(1))
    c = level_coefficients(z, h, sig2)
    print(f"sigma^2={sig2:.0f}, per-level coefficients (level 0=root .. 5=leaves):")
    print("  " + "  ".join(f"L{l}:{c[l]:.3f}" for l in range(h.depth)))
    assert c[0] == 1.0, "root (n=1<=3) must be protected (c=1)"
    assert c[h.depth - 1] < 0.1, "leaves should shrink hard at leaf_mean=1, eps=0.5"

    # ---- 2) the headline: shrinkage beats Hay by ~1-2 orders at low SNR ----
    print("\nPer-node MSE over 50 trials, (3,5) tree, leaf_mean=1:")
    print(f"{'eps':>6}{'Hay':>10}{'Ours(shrink)':>14}{'improvement':>13}")
    for eps in [0.1, 0.5, 1.0, 5.0]:
        s2 = noise_variance(h, eps)
        hay_e, our_e = [], []
        for t in range(50):
            zz = add_laplace_noise(x, h, eps, np.random.default_rng(100 + t))
            hay_e.append(np.mean((hay_two_pass(zz, h) - x) ** 2))
            our_e.append(np.mean((level_wise_shrinkage(zz, h, s2) - x) ** 2))
        hm, om = np.mean(hay_e), np.mean(our_e)
        print(f"{eps:>6}{hm:>10.2f}{om:>14.2f}{100*(hm-om)/hm:>12.1f}%")

    # ---- 3) high SNR (leaf_mean=100): should degrade gracefully toward Hay ----
    xh = generate_signal(h, "smooth", 100, 0)
    s2 = noise_variance(h, 5.0)
    hay_e = np.mean([np.mean((hay_two_pass(add_laplace_noise(xh, h, 5.0,
                     np.random.default_rng(t)), h) - xh)**2) for t in range(50)])
    our_e = np.mean([np.mean((level_wise_shrinkage(add_laplace_noise(xh, h, 5.0,
                     np.random.default_rng(t)), h, s2) - xh)**2) for t in range(50)])
    print(f"\nHigh SNR (leaf_mean=100, eps=5): Hay {hay_e:.2f} vs Ours {our_e:.2f} "
          f"-> degrades to ~Hay (no harm): {abs(hay_e-our_e)/hay_e*100:.1f}% diff")

    # ---- 4) output stays consistent (projection is the last step) ----
    out = level_wise_shrinkage(z, h, sig2)
    viol = max(abs(out[v] - sum(out[c2] for c2 in h.graph.successors(v)))
               for v in h.internal_nodes)
    print(f"\noutput consistency violation: {viol:.2e} (projection enforces it)")
    assert viol < 1e-9
    print("Self-test passed.")