"""
Graph-spectral (GFT--Tikhonov) post-processing -- the COMPLEX comparison method
this paper dominates. Two versions:

  spectral_sure(z, h, sigma2)        : Tikhonov filtering with the regularization
                                       parameter gamma chosen by SURE. Realistic
                                       (does not peek at the truth).
  spectral_oracle(z, h, sigma2, x)   : gamma chosen to minimize the true MSE
                                       against x. An (infeasible) performance
                                       UPPER BOUND for the spectral family.

Pipeline: normalized graph Laplacian L = I - D^{-1/2} A D^{-1/2}; eigendecompose
L = U diag(lam) U^T (graph Fourier basis); Tikhonov filter h(lam) = 1/(1+gamma*lam)
applied in the spectral domain; then consistency projection.

Why it is in the paper: level-wise shrinkage dominates this across the grid,
*even against the oracle gamma*. The reason is a basis mismatch -- consistent
signal energy is organized by tree LEVEL, which graph-frequency eigenvectors mix.
SURE is classically a Gaussian tool; reporting the SURE-vs-oracle gap under
Laplace noise addresses that head-on instead of asserting it.

Registers Spectral(SURE) and a truth-bound Spectral(oracle) into baselines.METHODS.
"""
from __future__ import annotations
import numpy as np

from hierarchy_data_generator import Hierarchy
from projection import hay_two_pass
import baseline

__all__ = ["spectral_sure", "spectral_oracle", "GAMMA_GRID",
           "make_spectral_oracle_method"]

GAMMA_GRID = np.logspace(-3, 5, 100)   # Tikhonov regularization grid


def _gft(h: Hierarchy):
    """Normalized-Laplacian graph Fourier basis (eigenvalues, eigenvectors).
    Treats the tree as undirected for the Laplacian."""
    n = h.num_nodes
    A = np.zeros((n, n))
    for u, v in h.graph.edges():
        A[u, v] = A[v, u] = 1.0
    d = A.sum(1)
    dinv = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    L = np.eye(n) - (dinv[:, None] * A) * dinv[None, :]
    lam, U = np.linalg.eigh(L)
    lam = np.clip(lam, 0.0, None)
    return lam, U


def _filtered(zc, lam, gamma):
    """Apply Tikhonov filter h(lam)=1/(1+gamma*lam) to GFT coefficients zc."""
    return zc / (1.0 + gamma * lam)


def spectral_sure(z: np.ndarray, h: Hierarchy, sigma2: float) -> np.ndarray:
    """Tikhonov filtering with gamma selected by SURE, then projection."""
    z = np.asarray(z, dtype=float)
    lam, U = _gft(h)
    zc = U.T @ z                       # GFT coefficients
    n = h.num_nodes
    # SURE(gamma) for the diagonal filter h(lam): for shrinkage factor f_i in [0,1],
    # SURE = ||(1-f) zc||^2 - n*sigma^2 + 2 sigma^2 sum f_i
    best_g, best_s = GAMMA_GRID[0], np.inf
    for g in GAMMA_GRID:
        f = 1.0 / (1.0 + g * lam)
        sure = np.sum(((1.0 - f) * zc) ** 2) - n * sigma2 + 2.0 * sigma2 * np.sum(f)
        if sure < best_s:
            best_s, best_g = sure, g
    xhat_c = _filtered(zc, lam, best_g)
    return hay_two_pass(U @ xhat_c, h)


def spectral_oracle(z: np.ndarray, h: Hierarchy, sigma2: float,
                    x: np.ndarray) -> np.ndarray:
    """Tikhonov filtering with gamma minimizing the TRUE MSE against x (infeasible
    upper bound), then projection."""
    z = np.asarray(z, dtype=float)
    x = np.asarray(x, dtype=float)
    lam, U = _gft(h)
    zc = U.T @ z
    best_g, best_e = GAMMA_GRID[0], np.inf
    for g in GAMMA_GRID:
        cand = hay_two_pass(U @ _filtered(zc, lam, g), h)
        e = np.mean((cand - x) ** 2)
        if e < best_e:
            best_e, best_g = e, g
    return hay_two_pass(U @ _filtered(zc, lam, best_g), h)


def make_spectral_oracle_method(x: np.ndarray):
    """Bind the truth x so the oracle fits the unified (z, h, sigma2) signature.
    The runner builds this per (signal) since the oracle legitimately needs x."""
    def method(z, h, sigma2):
        return spectral_oracle(z, h, sigma2, x)
    return method


# SURE version fits the unified signature directly; register it.
baseline.register("Spectral(SURE)", spectral_sure)
# The oracle is registered per-signal by the runner via make_spectral_oracle_method.


if __name__ == "__main__":
    from hierarchy_data_generator import create_balanced_tree
    from signal_generator import generate_signal
    from laplace_noise_injection import add_laplace_noise, noise_variance
    from shrinkage import level_wise_shrinkage

    h = create_balanced_tree(depth=6, branching_factor=3)
    x = generate_signal(h, "smooth", 1, 0)

    # sanity: gamma->0 means no filtering, so spectral collapses to Hay
    lam, U = _gft(h)
    z = add_laplace_noise(x, h, 0.5, np.random.default_rng(0))
    near0 = hay_two_pass(U @ _filtered(U.T @ z, lam, 1e-12), h)
    print(f"gamma->0 spectral vs Hay: max|diff| = "
          f"{np.max(np.abs(near0 - hay_two_pass(z, h))):.2e} (should be ~0)")

    # the headline comparisons over 50 trials
    print("\nPer-node MSE, (3,5) leaf_mean=1 (50 trials):")
    print(f"{'eps':>6}{'Hay':>9}{'Spec(SURE)':>12}{'Spec(oracle)':>14}{'Ours':>9}")
    for eps in [0.1, 0.5, 1.0]:
        s2 = noise_variance(h, eps)
        hay_e, ss_e, so_e, ours_e = [], [], [], []
        for t in range(50):
            zz = add_laplace_noise(x, h, eps, np.random.default_rng(t))
            hay_e.append(np.mean((hay_two_pass(zz, h) - x) ** 2))
            ss_e.append(np.mean((spectral_sure(zz, h, s2) - x) ** 2))
            so_e.append(np.mean((spectral_oracle(zz, h, s2, x) - x) ** 2))
            ours_e.append(np.mean((level_wise_shrinkage(zz, h, s2) - x) ** 2))
        print(f"{eps:>6}{np.mean(hay_e):>9.2f}{np.mean(ss_e):>12.2f}"
              f"{np.mean(so_e):>14.2f}{np.mean(ours_e):>9.2f}")

    # the two claims the paper rests on
    s2 = noise_variance(h, 0.5)
    so = np.mean([np.mean((spectral_oracle(add_laplace_noise(x, h, 0.5,
                  np.random.default_rng(t)), h, s2, x) - x)**2) for t in range(50)])
    ours = np.mean([np.mean((level_wise_shrinkage(add_laplace_noise(x, h, 0.5,
                    np.random.default_rng(t)), h, s2) - x)**2) for t in range(50)])
    ss = np.mean([np.mean((spectral_sure(add_laplace_noise(x, h, 0.5,
                  np.random.default_rng(t)), h, s2) - x)**2) for t in range(50)])
    print(f"\nClaim 1 - Ours beats even ORACLE spectral: "
          f"Ours {ours:.2f} < Spec(oracle) {so:.2f}: {ours < so}")
    print(f"Claim 2 - SURE-vs-oracle gap (Laplace): "
          f"SURE {ss:.2f} vs oracle {so:.2f} -> SURE is {ss/so:.1f}x the oracle MSE")
    assert ours < so, "level-wise shrinkage must beat oracle-tuned spectral"
    print("\nSelf-test passed.")