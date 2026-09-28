"""
Baseline comparison methods + the unified method registry.

Every post-processing method is wrapped to ONE signature so the experiment
runner can iterate uniformly:

    method(z, h, sigma2) -> x_hat        # shape (h.num_nodes,)

z       : the DP release (noisy node counts), shape (num_nodes,)
h       : the Hierarchy
sigma2  : per-node noise variance (public); methods that don't need it ignore it

This file owns the COMPARISON baselines (Zero, Laplace) and registers the
already-implemented methods (Hay from projection, our shrinkage). Ablations and
the spectral method live in their own files and register themselves into
METHODS there (or extend it), keeping each file single-purpose.
"""
from __future__ import annotations
from typing import Callable, Dict
import numpy as np

from hierarchy_data_generator import Hierarchy
import projection as _proj
from shrinkage import level_wise_shrinkage

__all__ = ["zero_release", "laplace_release", "hay_release", "ours_release",
           "METHODS", "Method"]

Method = Callable[[np.ndarray, Hierarchy, float], np.ndarray]


# ---- comparison baselines (the genuinely new methods in this file) ----
def zero_release(z: np.ndarray, h: Hierarchy, sigma2: float) -> np.ndarray:
    """Trivial all-zeros release. The boundary competitor: at tiny epsilon no
    linear method beats this, which the operating-envelope figure shows."""
    return np.zeros(h.num_nodes)


def laplace_release(z: np.ndarray, h: Hierarchy, sigma2: float) -> np.ndarray:
    """Raw noisy counts, no post-processing. The 'do nothing' reference."""
    return np.asarray(z, dtype=float)


# ---- thin wrappers giving the existing methods the unified signature ----
def hay_release(z: np.ndarray, h: Hierarchy, sigma2: float) -> np.ndarray:
    """Consistency projection alone -- the canonical baseline (Hay two-pass by
    default; exact LS when projection.set_projection("ls") is active)."""
    return _proj.project(z, h)


def ours_release(z: np.ndarray, h: Hierarchy, sigma2: float) -> np.ndarray:
    """This paper: level-wise shrinkage + consistency projection."""
    return level_wise_shrinkage(z, h, sigma2)


# ---- the registry the runner iterates over ----
# Insertion order is the natural display order in tables.
METHODS: Dict[str, Method] = {
    "Laplace": laplace_release,   # do nothing
    "Zero":    zero_release,      # boundary competitor
    "Hay":     hay_release,       # canonical baseline
    "Ours":    ours_release,      # level-wise shrinkage (this paper)
}


def register(name: str, fn: Method, *, overwrite: bool = False) -> None:
    """Let ablations.py / spectral.py add their methods to the shared registry."""
    if name in METHODS and not overwrite:
        raise KeyError(f"method {name!r} already registered (use overwrite=True).")
    METHODS[name] = fn


if __name__ == "__main__":
    from hierarchy_data_generator import create_balanced_tree
    from signal_generator import generate_signal
    from laplace_noise_injection import add_laplace_noise, noise_variance

    h = create_balanced_tree(depth=6, branching_factor=3)
    x = generate_signal(h, "smooth", 1, 0)
    sig2 = noise_variance(h, 0.5)

    # every registered method returns the right shape under one signature
    z = add_laplace_noise(x, h, 0.5, np.random.default_rng(0))
    print(f"Registered methods: {list(METHODS)}")
    for name, fn in METHODS.items():
        out = fn(z, h, sig2)
        assert out.shape == (h.num_nodes,), f"{name} wrong shape"
    print("All methods honor the (z, h, sigma2) -> x_hat signature.\n")

    # quick MSE sanity over 50 trials so the ordering looks right
    print(f"Per-node MSE, (3,5) leaf_mean=1, eps=0.5 (50 trials):")
    acc = {name: [] for name in METHODS}
    for t in range(50):
        zz = add_laplace_noise(x, h, 0.5, np.random.default_rng(t))
        for name, fn in METHODS.items():
            acc[name].append(np.mean((fn(zz, h, sig2) - x) ** 2))
    for name in METHODS:
        print(f"  {name:<9} {np.mean(acc[name]):>9.2f}")
    # sanity: Ours < Hay < Laplace at this setting; Zero somewhere by signal scale
    assert np.mean(acc["Ours"]) < np.mean(acc["Hay"]) < np.mean(acc["Laplace"])
    print("\nSelf-test passed.")