"""
Ablation variants of the level-wise shrinkage method.

Each ablation DISABLES one component of the full method to show that component
is necessary. None of them re-implements the method: they call the existing
shrinkage functions with a component switched off. Importing this module
registers the ablations into baselines.METHODS so the runner picks them up.

Full method = per-level shrink-to-zero  +  root protection (n_l<=3)  +  projection.

  Scalar         : ONE global coefficient instead of per-level  -> tests that
                   per-LEVEL granularity matters (vs a single tree-wide factor).
  NoProject      : per-level shrink but SKIP the consistency projection  -> tests
                   that the projection step matters (and Lemma 4: it never hurts).
  NoRootProtect  : per-level shrink with root protection OFF (root_protect_max=0)
                   -> tests that protecting tiny levels matters.
  PerNode        : a separate coefficient per NODE (no grouping)  -> tests that
                   level is the right granularity (not too coarse, not too fine).
"""
from __future__ import annotations
import numpy as np

from hierarchy_data_generator import Hierarchy
from projection import hay_two_pass
from shrinkage import level_wise_shrinkage, _apply_level_scaling, ROOT_PROTECT_MAX
import baseline

__all__ = ["scalar_shrinkage", "no_projection_shrinkage",
           "no_root_protect_shrinkage", "per_node_shrinkage"]


def _identity_projection(y, h):
    """A 'projection' that does nothing -- for the NoProject ablation."""
    return y


# ---- Ablation 1: global single coefficient (disable per-level) ----
def scalar_shrinkage(z: np.ndarray, h: Hierarchy, sigma2: float) -> np.ndarray:
    """One coefficient for the WHOLE tree: c = max(0, 1 - n*sigma^2/||z||^2),
    then project. Tests whether per-level beats a single global factor."""
    z = np.asarray(z, dtype=float)
    n = h.num_nodes
    energy = float(np.dot(z, z))
    c = max(0.0, 1.0 - n * sigma2 / energy) if energy > 0 else 0.0
    return hay_two_pass(c * z, h)


# ---- Ablation 2: per-level shrink but no projection (disable projection) ----
def no_projection_shrinkage(z: np.ndarray, h: Hierarchy, sigma2: float) -> np.ndarray:
    """Per-level shrink-to-zero with NO consistency projection. Output is
    generally inconsistent; isolates the projection's contribution."""
    return level_wise_shrinkage(z, h, sigma2, project=_identity_projection)


# ---- Ablation 3: no root protection (disable the n_l<=3 guard) ----
def no_root_protect_shrinkage(z: np.ndarray, h: Hierarchy, sigma2: float) -> np.ndarray:
    """Per-level shrink with root protection turned OFF (all levels shrink,
    including the 1-node root). Tests the root-protection rule."""
    return level_wise_shrinkage(z, h, sigma2, root_protect_max=0)


# ---- Ablation 4 (optional): per-node coefficient (disable grouping) ----
def per_node_shrinkage(z: np.ndarray, h: Hierarchy, sigma2: float) -> np.ndarray:
    """A coefficient per NODE: c_i = max(0, 1 - sigma^2/z_i^2), then project.
    The over-fine extreme; tests that LEVEL is the right granularity."""
    z = np.asarray(z, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.where(z**2 > 0, np.maximum(0.0, 1.0 - sigma2 / z**2), 0.0)
    return hay_two_pass(c * z, h)


# register into the shared method registry (runner will iterate over it)
baseline.register("Scalar", scalar_shrinkage)
baseline.register("NoProject", no_projection_shrinkage)
baseline.register("NoRootProtect", no_root_protect_shrinkage)
baseline.register("PerNode", per_node_shrinkage)


if __name__ == "__main__":
    from hierarchy_data_generator import create_balanced_tree
    from signal_generator import generate_signal
    from laplace_noise_injection import add_laplace_noise, noise_variance
    from baseline import METHODS

    h = create_balanced_tree(depth=6, branching_factor=3)
    x = generate_signal(h, "smooth", 1, 0)         # low SNR: ablations bite here
    sig2 = noise_variance(h, 0.5)

    print(f"All registered methods now: {list(METHODS)}\n")

    # MSE over 50 trials: the FULL method should beat every ablation
    acc = {name: [] for name in METHODS}
    for t in range(50):
        zz = add_laplace_noise(x, h, 0.5, np.random.default_rng(t))
        for name, fn in METHODS.items():
            acc[name].append(np.mean((fn(zz, h, sig2) - x) ** 2))
    means = {name: float(np.mean(v)) for name, v in acc.items()}

    print("Per-node MSE, (3,5) leaf_mean=1, eps=0.5 (50 trials):")
    for name in METHODS:
        print(f"  {name:<15}{means[name]:>10.2f}")

    ours = means["Ours"]
    print("\nablation check (each disabled component should cost accuracy):")
    for ab in ["Scalar", "NoProject", "NoRootProtect", "PerNode"]:
        worse = means[ab] > ours
        print(f"  Ours ({ours:.2f}) vs {ab} ({means[ab]:.2f}): "
              f"{'full method better ✓' if worse else 'NOT worse ✗'}")
        assert worse, f"full method should beat {ab}"

    # NoProject output must be inconsistent (that's the point of the ablation)
    z = add_laplace_noise(x, h, 0.5, np.random.default_rng(0))
    out = no_projection_shrinkage(z, h, sig2)
    viol = max(abs(out[v] - sum(out[c] for c in h.graph.successors(v)))
               for v in h.internal_nodes)
    print(f"\nNoProject consistency violation: {viol:.1f} (nonzero, as expected -- "
          f"shows projection is what restores consistency)")
    assert viol > 1.0
    print("Self-test passed.")