"""
Hierarchy data structures and generators for level-wise shrinkage DP experiments.

Public API (import these):
    Hierarchy, create_balanced_tree, create_unbalanced_tree

Running this file directly (`python hierarchy_data_generator.py`) generates a
reference archive of the synthetic tree specifications used in the paper and
verifies it (golden-master check):
  - For each spec, the deterministic STRUCTURE (topology, levels, per-level
    counts, sensitivity) is computed and written to reference_data/<name>.json.
  - If the file already exists, the freshly computed structure is compared
    field-by-field against it and any mismatch raises (catches regressions).
  - No randomness, no noise, no signal: this archive is the raw, fully
    reproducible structural ground truth that everything downstream references.

Design principles:
  - A node's level is its OWN BFS distance from the root, independent of how
    deep other nodes are. levels.max() only sets the tree's overall height/depth.
  - Levels are derived from graph structure, NEVER inferred from node ids.
  - `depth` = number of levels (levels 0..depth-1). `height` = depth - 1.
    L1 sensitivity = depth (a leaf->root path touches `depth` nodes). ONE
    definition of sensitivity, exposed as h.sensitivity.
"""
from __future__ import annotations
import dataclasses as dc
import json
import os
from typing import Dict, List
import numpy as np
import networkx as nx

__all__ = ["Hierarchy", "create_balanced_tree", "create_unbalanced_tree"]


@dc.dataclass
class Hierarchy:
    graph: nx.DiGraph
    root: int
    levels: np.ndarray
    level_nodes: Dict[int, List[int]]

    @property
    def num_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def depth(self) -> int:
        return int(self.levels.max()) + 1

    @property
    def height(self) -> int:
        return int(self.levels.max())

    @property
    def sensitivity(self) -> int:
        return self.depth

    @property
    def leaves(self) -> List[int]:
        return [n for n in self.graph.nodes if self.graph.out_degree(n) == 0]

    @property
    def internal_nodes(self) -> List[int]:
        return [n for n in self.graph.nodes if self.graph.out_degree(n) > 0]

    def num_nodes_at_level(self, ell: int) -> int:
        return len(self.level_nodes.get(ell, []))

    def nodes_at_level(self, ell: int) -> np.ndarray:
        return np.array(self.level_nodes.get(ell, []), dtype=int)

    def level_mask(self, ell: int) -> np.ndarray:
        return self.levels == ell

    def to_reference_dict(self) -> dict:
        """Deterministic structural summary for the golden-master archive.
        Edges are sorted so the JSON is stable across runs."""
        return {
            "num_nodes": int(self.num_nodes),
            "num_leaves": int(len(self.leaves)),
            "height": int(self.height),
            "depth": int(self.depth),
            "sensitivity": int(self.sensitivity),
            "levels": [int(x) for x in self.levels],
            "level_sizes": [int(self.num_nodes_at_level(l)) for l in range(self.depth)],
            "edges": sorted([[int(u), int(v)] for u, v in self.graph.edges()]),
        }


def _build_hierarchy(graph: nx.DiGraph, root: int) -> Hierarchy:
    n = graph.number_of_nodes()
    if sorted(graph.nodes) != list(range(n)):
        raise ValueError("Nodes must be labeled 0..n-1.")
    dist = nx.single_source_shortest_path_length(graph, root)
    if len(dist) != n:
        raise ValueError("Graph is not fully reachable from the given root.")
    levels = np.array([dist[i] for i in range(n)], dtype=int)
    level_nodes: Dict[int, List[int]] = {}
    for node in range(n):
        level_nodes.setdefault(int(levels[node]), []).append(node)
    return Hierarchy(graph=graph, root=root, levels=levels, level_nodes=level_nodes)


def create_balanced_tree(depth: int, branching_factor: int) -> Hierarchy:
    """Balanced `branching_factor`-ary tree with `depth` levels (0..depth-1).
    Example: create_balanced_tree(depth=6, branching_factor=3) is the (3,5) tree."""
    if depth < 1:
        raise ValueError("depth must be >= 1.")
    if branching_factor < 1:
        raise ValueError("branching_factor must be >= 1.")
    G = nx.balanced_tree(r=branching_factor, h=depth - 1, create_using=nx.DiGraph)
    return _build_hierarchy(G, root=0)


def create_unbalanced_tree(edges: List[tuple], root: int = 0) -> Hierarchy:
    """Arbitrary tree from an explicit (parent, child) edge list."""
    G = nx.DiGraph()
    G.add_edges_from(edges)
    return _build_hierarchy(G, root=root)


# ----------------------------------------------------------------------------
# Reference archive: the synthetic tree specifications used in the paper.
# (depth, branching_factor, filename). Edit this list to change what is stored.
# ----------------------------------------------------------------------------
SYNTHETIC_SPECS = [
    (6, 3, "tree_d6_b3"),   # main (3,5) tree: 364 nodes
    (5, 2, "tree_d5_b2"),   # scale sweep: 31 nodes
    (7, 2, "tree_d7_b2"),   # scale sweep: 127 nodes
    (9, 2, "tree_d9_b2"),   # scale sweep: 511 nodes
    (5, 3, "tree_d5_b3"),   # branching variation: 121 nodes
    (4, 5, "tree_d4_b5"),   # wide fan-out: 781 nodes
    (4, 4, "tree_d4_b4"),   # wide fan-out: 85 nodes
]

# Unbalanced specs: list of (name, edges). Empty for now; add (parent, child)
# edge lists here when you want unbalanced reference trees. They are written to
# UNBALANCED_DIR. Example:
#   ("tree_unbal_chainfan", [(0,1),(0,2),(2,3),(3,4),(0,5),(0,6),(0,7)])
UNBALANCED_SPECS = [
    ("tree_unbal_census", [
        (0, 1), (0, 2), (0, 3),
        (1, 4), (1, 5), (4, 8), (4, 9), (5, 10), (5, 11),
        (2, 6), (6, 7), (7, 12), (7, 13),          # 2->6->7 single-child chain
        (3, 14), (3, 15), (3, 16), (14, 17), (14, 18),
    ]),
    ("tree_unbal_patho",
        [(0, i) for i in range(1, 31)]              # 30-wide fan-out
        + [(0, 31), (31, 32), (32, 33), (33, 34), (34, 35)]   # deep chain
        + [(35, 36), (35, 37), (35, 38)]),          # chain-tail fan-out
]

# Output directories (relative to where this script is run).
BALANCED_DIR = "../synthetic_data/raw_balanced_data"
UNBALANCED_DIR = "../synthetic_data/raw_unbalanced_data"


def _write_or_check(ref: dict, path: str) -> None:
    """Write the reference dict to path, or verify against it if it already exists."""
    if os.path.exists(path):
        with open(path) as f:
            saved = json.load(f)
        if saved != ref:
            diffs = [k for k in ref if saved.get(k) != ref.get(k)]
            raise AssertionError(
                f"REFERENCE MISMATCH in {path}: fields differ -> {diffs}. "
                f"Code change altered tree structure; investigate before overwriting."
            )
        print(f"  [check ✓] {os.path.basename(path):<22} "
              f"{ref['num_nodes']} nodes, {ref['num_leaves']} leaves, sens {ref['sensitivity']}")
    else:
        with open(path, "w") as f:
            json.dump(ref, f, indent=1)
        print(f"  [wrote  ] {os.path.basename(path):<22} "
              f"{ref['num_nodes']} nodes, {ref['num_leaves']} leaves, sens {ref['sensitivity']}")


def generate_or_check_reference(
    balanced_dir: str = BALANCED_DIR,
    unbalanced_dir: str = UNBALANCED_DIR,
) -> None:
    """Generate (or verify) the structural reference archive.
    Balanced trees -> balanced_dir; unbalanced trees -> unbalanced_dir."""
    os.makedirs(balanced_dir, exist_ok=True)
    print(f"Balanced trees -> {balanced_dir}")
    for depth, bf, name in SYNTHETIC_SPECS:
        h = create_balanced_tree(depth=depth, branching_factor=bf)
        ref = {"spec": {"kind": "balanced", "depth": depth, "branching_factor": bf},
               **h.to_reference_dict()}
        _write_or_check(ref, os.path.join(balanced_dir, name + ".json"))

    if UNBALANCED_SPECS:
        os.makedirs(unbalanced_dir, exist_ok=True)
        print(f"Unbalanced trees -> {unbalanced_dir}")
        for name, edges in UNBALANCED_SPECS:
            h = create_unbalanced_tree(edges, root=0)
            ref = {"spec": {"kind": "unbalanced", "name": name},
                   **h.to_reference_dict()}
            _write_or_check(ref, os.path.join(unbalanced_dir, name + ".json"))


if __name__ == "__main__":
    # quick sanity on the main tree
    h = create_balanced_tree(depth=6, branching_factor=3)
    assert h.num_nodes == 364 and len(h.leaves) == 243 and h.sensitivity == 6
    print("Main (3,5) tree OK.")
    print("Generating / checking structural reference archive:")
    print("start generation_or_check_reference() for rest of the trees...")
    generate_or_check_reference()
    print("Done.")