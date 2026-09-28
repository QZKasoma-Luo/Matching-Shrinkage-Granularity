"""
Load the 2020 Census Detailed DHC-A table T01001 (Total Population) as a
(nation -> state -> county) hierarchy for ONE detailed race/ethnicity group, and
treat the published per-group county counts as the ground-truth PROXY x.

This is the Detailed-DHC-A analogue of census_tree_loader.py. The structural
differences from the PL94 P1 file (which is why the P1 loader cannot read this
file directly) are handled here:

  1. Count column is `T01001_001N` (not `P1_001N`).
  2. The file is a LONG table: one row per (county, population group). There are
     ~2962 groups x ~3221 counties. We therefore FILTER to a single POPGROUP.
  3. Publication threshold (the important one). At substate geographies the
     Detailed DHC-A adaptive design publishes a group's total for a county only
     if its noisy count reached the threshold (22); below it the (group, county)
     row is simply ABSENT from the file. Absent counties are therefore "fewer
     than 22 (noisy)", not "known to be zero".
  4. The literal "X" is something else and rare: per the table notes it marks a
     negative noisy count or an "alone" count exceeding its "alone or in any
     combination" count. Across the five selected groups only 2 cells are "X".
     "X" / non-numeric / negative -> treated as 0, like absent rows.
  5. To put the method in its working regime, the tree is built over the FULL
     county universe (every county that appears for ANY group), with the chosen
     group's count = 0 wherever it is absent or "X". This makes x a sparse,
     mostly-zero signal -- the low-count regime -- but 85-99 % of its leaves are
     IMPUTED zeros (see sensitivity_absent_counties.py). The PL94 P1 columns
     (P1_sparse_columns.py) have no threshold and are the fully observed
     alternative used in the conference version.

Note on the proxy: the published DDHC-A counts are already DP-processed
(SafeTab-P), so -- exactly as in the PL94 pipeline -- they are a proxy ground
truth, not the secret truth; we re-impose consistency by bottom-up summation and
add our OWN calibrated noise on top. Setting below-threshold counties to 0
biases those cells downward by up to ~21 persons each; state it in the write-up
and calibrate it with the state-level T01001 totals (published for every group)
before drawing conclusions that depend on the empty stratum.

Output matches the synthetic / PL94 pipeline interface:
    load_ddhca(csv, popgroup) -> (Hierarchy, x, meta)
so the same mechanism / projection / shrinkage / METHODS code applies unchanged.

GEO_ID format: "0500000US01001" -> after "US" is FIPS "01001"; first 2 = state
(01 = Alabama), full 5 = county.
"""
from __future__ import annotations
import csv
import json
import os
import time
from collections import defaultdict
import numpy as np

from hierarchy_data_generator import create_unbalanced_tree
from laplace_noise_injection import (add_laplace_noise, noise_variance,
                                     EPSILONS, NUM_TRIALS)
import baseline, ablations, spectral          # registers all methods
from baseline import METHODS
from spectral import make_spectral_oracle_method

__all__ = ["load_ddhca", "save_hierarchy", "load_hierarchy_archive",
           "level_ec", "popgroup_table", "run_ddhca",
           "COUNTY_PREFIX", "COUNT_COL"]

CENSUS_DDHCA_RESULTS_DIR = "../results/census_ddhca_results"
CENSUS_DDHCA_HIER_DIR = "../census_data/census_hierarchy_format"
DDHCA_CSV = "../census_data/original_data/T01001/DECENNIALDDHCA2020.T01001-Data.csv"

COUNTY_PREFIX = "0500000US"   # county-level GEO_IDs in this file
COUNT_COL = "T01001_001N"     # Detailed DHC-A total-population column

# Population groups chosen (via popgroup_table) to span a county-leaf sparsity
# range, from very sparse to near-saturated. Comment = leaf E[c] at eps=0.5
# (well below 1 => shrinkage activates; ~1 => high-SNR, no-gain control).
SELECTED_POPGROUPS = [
    "1190",   # Emirati alone or in any combination          E[c]@.5 ~ 0.22 (sparsest)
    "1016",   # Carpatho Rusyn alone                          E[c]@.5 ~ 0.40
    "2031",   # Pueblo alone                                  E[c]@.5 ~ 0.60
    "3657",   # Guatemalan Indian alone or in any combination E[c]@.5 ~ 0.82
    "1009",   # Basque alone                                  E[c]@.5 ~ 0.96 (high-SNR control)
]

# Cache the graph Fourier basis. The spectral baselines recompute the full
# eigendecomposition on every call, but it depends only on the tree STRUCTURE --
# which is IDENTICAL across all groups (each is padded to the same county
# universe) and across every trial/epsilon. Memoizing turns hundreds of
# 3274x3274 eigendecompositions into one. (Same fix as runner.py; key by a
# STABLE identity, not id(graph).)
_orig_gft = spectral._gft
_GFT_CACHE = {}
def _gft_memo(h):
    key = (h.num_nodes, tuple(sorted(h.graph.edges())))
    if key not in _GFT_CACHE:
        _GFT_CACHE[key] = _orig_gft(h)
    return _GFT_CACHE[key]
spectral._gft = _gft_memo


def _parse_count(s):
    """Published count -> non-negative int, or None if suppressed/absent.
    DDHC-A suppresses small cells as the literal 'X'; other non-numeric or
    negative sentinels are likewise treated as 'not a real count'."""
    if s is None:
        return None
    v = str(s).replace(",", "").strip()
    if not v:
        return None
    try:
        iv = int(v)
    except ValueError:
        return None
    return iv if iv >= 0 else None


def load_ddhca(csv_path: str, popgroup, keep_puerto_rico: bool = True,
               pad_full_universe: bool = True):
    """Build the (nation -> state -> county) Hierarchy and proxy signal x for a
    single Detailed DHC-A population group (table T01001).

    popgroup           : the POPGROUP code to select (e.g. "2625"); int or str.
    keep_puerto_rico   : keep state FIPS 72 if True.
    pad_full_universe  : if True (default) the tree spans EVERY county in the
                         file and the group's count is 0 where absent/suppressed
                         (the sparse low-count regime). If False, the tree spans
                         only counties where the group is actually present.

    Returns (hierarchy, x, meta).
    """
    popgroup = str(popgroup).strip()
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    header = [h.strip().strip('"').strip() for h in rows[0]]
    gi = header.index("GEO_ID")
    ni = header.index("NAME")
    ci = header.index(COUNT_COL)
    gpi = header.index("POPGROUP")
    gli = header.index("POPGROUP_LABEL")

    # county universe (geoid -> name, state_fips, state_name), from ALL groups,
    # plus the chosen group's per-county counts.
    universe = {}            # geoid -> (name, state_fips, state_name)
    group_count = {}         # geoid -> count (chosen group only)
    popgroup_label = None
    for r in rows[2:]:       # row 0 = names, row 1 = label descriptions
        if not r or not r[gi] or not r[gi].startswith(COUNTY_PREFIX):
            continue
        geoid = r[gi]
        fips = geoid.split("US")[-1]
        state_fips = fips[:2]
        if not keep_puerto_rico and state_fips == "72":
            continue
        name = r[ni]
        state_name = name.rsplit(",", 1)[-1].strip() if "," in name else state_fips
        universe.setdefault(geoid, (name, state_fips, state_name))
        if r[gpi].strip() == popgroup:
            popgroup_label = r[gli]
            c = _parse_count(r[ci])
            if c is not None:
                group_count[geoid] = c

    if popgroup_label is None and not group_count:
        raise ValueError(f"POPGROUP {popgroup!r} not found in {csv_path}.")
    if not pad_full_universe:
        universe = {g: universe[g] for g in group_count}
    if not universe:
        raise ValueError("Empty county universe for the chosen settings.")

    # node numbering: 0 = nation; then states (sorted by fips); then counties
    state_fips_list = sorted({sf for (_, sf, _) in universe.values()})
    state_node = {sf: 1 + i for i, sf in enumerate(state_fips_list)}
    n_states = len(state_fips_list)
    county_node = {}
    edges = []
    nid = 1 + n_states
    for geoid in sorted(universe):          # deterministic county order
        county_node[geoid] = nid
        edges.append((state_node[universe[geoid][1]], nid))   # state -> county
        nid += 1
    for sf in state_fips_list:
        edges.append((0, state_node[sf]))   # nation -> state

    h = create_unbalanced_tree(edges, root=0)

    # consistent x: counties = chosen-group counts (0 where absent/suppressed),
    # states and nation = bottom-up sums.
    counties_by_state = defaultdict(list)
    for geoid in universe:
        counties_by_state[universe[geoid][1]].append(geoid)
    x = np.zeros(h.num_nodes)
    for geoid in universe:
        x[county_node[geoid]] = group_count.get(geoid, 0)
    for sf in state_fips_list:
        x[state_node[sf]] = sum(x[county_node[g]] for g in counties_by_state[sf])
    x[0] = x[[state_node[sf] for sf in state_fips_list]].sum()

    # readable per-node labels + a state_fips -> state_name map
    seen_state = {}
    node_info = {0: {"geoid": "0100000US", "name": "United States", "kind": "nation"}}
    for geoid, (name, sf, sn) in universe.items():
        seen_state.setdefault(sf, sn)
        node_info[county_node[geoid]] = {"geoid": geoid, "name": name, "kind": "county"}
    for sf in state_fips_list:
        node_info[state_node[sf]] = {"geoid": f"0400000US{sf}",
                                     "name": seen_state[sf], "kind": "state"}

    meta = {
        "popgroup": popgroup,
        "popgroup_label": popgroup_label,
        "n_counties": len(universe),
        "n_counties_present": len(group_count),   # group actually published here
        "n_states": n_states,
        "total_population": int(x[0]),
        "pad_full_universe": pad_full_universe,
        "state_names": {state_node[sf]: seen_state[sf] for sf in state_fips_list},
        "node_info": node_info,
        "edges": [[int(u), int(v)] for u, v in edges],
    }
    return h, x, meta


def save_hierarchy(h, x, meta, out_dir: str = CENSUS_DDHCA_HIER_DIR) -> str:
    """Persist the treelized DDHC-A data (structure + proxy x + labels) for one
    population group, so the exact tree is reproducible without the raw CSV.
    File name carries the POPGROUP so multiple groups never collide."""
    os.makedirs(out_dir, exist_ok=True)
    record = {
        "source": "2020 Census Detailed DHC-A, table T01001 (T01001_001N), county level",
        "note": "Published DDHC-A counts used as ground-truth PROXY (already "
                "SafeTab-P/DP-processed). Counties where the group did not reach "
                "the publication threshold (noisy count < 22) are absent from the "
                "table and set to 0 here (imputed zeros, true value 0-21); the "
                "rare literal 'X' (negative noisy count / alone > combination) is "
                "also set to 0; state/nation values are bottom-up sums.",
        "popgroup": meta["popgroup"],
        "popgroup_label": meta["popgroup_label"],
        "num_nodes": int(h.num_nodes),
        "n_states": meta["n_states"],
        "n_counties": meta["n_counties"],
        "n_counties_present": meta["n_counties_present"],
        "pad_full_universe": meta["pad_full_universe"],
        "height": int(h.height),
        "sensitivity": int(h.sensitivity),
        "total_population": meta["total_population"],
        "edges": meta["edges"],
        "levels": [int(v) for v in h.levels],
        "level_sizes": [int(h.num_nodes_at_level(l)) for l in range(h.depth)],
        "x": [float(v) for v in x],
        "node_info": {str(k): v for k, v in meta["node_info"].items()},
    }
    safe = str(meta["popgroup"]).replace("/", "-")
    # base name parallels the PL94 archive (P1_census_county_2020.json); the
    # POPGROUP suffix distinguishes the several groups we hierarchize.
    path = os.path.join(out_dir, f"T01001_census_county_2020_pop{safe}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=1)
    return path


def build_archives(csv_path: str = DDHCA_CSV, popgroups=None,
                   out_dir: str = CENSUS_DDHCA_HIER_DIR) -> list:
    """Hierarchize and SAVE each chosen population group to out_dir. This only
    builds + writes the tree archives (no noise, no methods, no experiment run);
    defaults to SELECTED_POPGROUPS. Returns the list of written paths."""
    popgroups = SELECTED_POPGROUPS if popgroups is None else popgroups
    paths = []
    for pg in popgroups:
        h, x, meta = load_ddhca(csv_path, pg)
        p = save_hierarchy(h, x, meta, out_dir)
        paths.append(p)
        print(f"  wrote {os.path.basename(p)}  (pop{pg} {meta['popgroup_label']}; "
              f"{meta['n_counties_present']}/{meta['n_counties']} counties present, "
              f"total {meta['total_population']:,})")
    return paths


def load_hierarchy_archive(path: str):
    """Rebuild (Hierarchy, x, record) from a saved DDHC-A archive."""
    with open(path, encoding="utf-8") as f:
        rec = json.load(f)
    h = create_unbalanced_tree([tuple(e) for e in rec["edges"]], root=0)
    x = np.array(rec["x"], dtype=float)
    return h, x, rec


# ----------------------------------------------------------------------------
# Regime pre-check: will level-wise shrinkage actually do anything here?
# E[c_l] ~ ||x_l||^2 / (||x_l||^2 + n_l * sigma^2)  (since E||z_l||^2 = ||x_l||^2
# + n_l sigma^2). Values well below 1 at the leaf level => the method activates.
# ----------------------------------------------------------------------------
def level_ec(h, x, eps):
    """Return {level: (n_nodes, expected_shrinkage_coefficient)} at this epsilon."""
    sigma2 = noise_variance(h, eps)
    out = {}
    for l in range(h.depth):
        mask = h.level_mask(l)
        n = int(mask.sum())
        e = float(np.dot(x[mask], x[mask]))
        denom = e + n * sigma2
        out[l] = (n, (e / denom) if denom > 0 else 0.0)
    return out


def popgroup_table(csv_path: str, eps_list=(0.1, 0.5, 1.0), min_counties: int = 30,
                   top: int = 25, keep_puerto_rico: bool = True):
    """Scan the file ONCE and, for every population group present in >=
    min_counties counties, report the county-level expected coefficient E[c]
    (padded to the full county universe) at each epsilon, so you can pick a group
    that lands in the working regime. Lower E[c] => more shrinkage potential.
    Returns a list of dicts sorted by E[c] at the smallest epsilon (ascending)."""
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    header = [h.strip().strip('"').strip() for h in rows[0]]
    gi = header.index("GEO_ID"); ci = header.index(COUNT_COL)
    gpi = header.index("POPGROUP"); gli = header.index("POPGROUP_LABEL")

    counties = set()
    by_group = defaultdict(dict)     # popgroup -> {geoid: count}
    labels = {}
    for r in rows[2:]:
        if not r or not r[gi] or not r[gi].startswith(COUNTY_PREFIX):
            continue
        sf = r[gi].split("US")[-1][:2]
        if not keep_puerto_rico and sf == "72":
            continue
        counties.add(r[gi])
        c = _parse_count(r[ci])
        if c is not None:
            by_group[r[gpi].strip()][r[gi]] = c
        labels[r[gpi].strip()] = r[gli]

    U = len(counties)
    # county tree is nation->state->county: 3 levels => sensitivity 3.
    def sigma2(eps):
        return 2.0 * (3.0 / eps) ** 2

    out = []
    for g, d in by_group.items():
        if len(d) < min_counties:
            continue
        ssq = float(sum(v * v for v in d.values()))
        rms2 = ssq / U                     # padded to full universe
        ecs = {e: rms2 / (rms2 + sigma2(e)) for e in eps_list}
        out.append({
            "popgroup": g, "label": labels[g],
            "n_counties": len(d), "total": int(sum(d.values())),
            "rms_padded": rms2 ** 0.5, "ec": ecs,
        })
    out.sort(key=lambda r: r["ec"][min(eps_list)])
    return out[:top]


def run_ddhca(csv_path: str, popgroup, with_spectral: bool = True,
              keep_puerto_rico: bool = True) -> None:
    """Full experiment for one population group: load -> add noise -> run all
    methods over EPSILONS -> results JSON. Mirrors run_census so plotting is
    shared. Noise seeds vary with epsilon (independent draws per (eps, trial))."""
    h, x, meta = load_ddhca(csv_path, popgroup, keep_puerto_rico=keep_puerto_rico)
    arch = save_hierarchy(h, x, meta)
    print(f"Saved DDHC-A archive -> {arch}")
    print(f"Group {meta['popgroup']} ({meta['popgroup_label']}): "
          f"{h.num_nodes} nodes ({meta['n_states']} states, {meta['n_counties']} "
          f"counties, {meta['n_counties_present']} with the group present), "
          f"height {h.height}, sensitivity {h.sensitivity}")
    print(f"Total population (proxy GT): {meta['total_population']:,}")

    methods = dict(METHODS)
    if not with_spectral:
        methods = {k: v for k, v in methods.items() if "Spectral" not in k}
    methods_with_oracle = dict(methods)
    if with_spectral:
        methods_with_oracle["Spectral(oracle)"] = make_spectral_oracle_method(x)

    os.makedirs(CENSUS_DDHCA_RESULTS_DIR, exist_ok=True)
    results = []
    t0 = time.time()
    for ei, eps in enumerate(EPSILONS):
        sigma2 = noise_variance(h, eps)
        acc = {name: [] for name in methods_with_oracle}
        for trial in range(NUM_TRIALS):
            # distinct, collision-free per (eps, trial); independent across eps
            rng = np.random.default_rng(1_300_000 + ei * 10_000 + trial)
            z = add_laplace_noise(x, h, eps, rng)
            for name, fn in methods_with_oracle.items():
                acc[name].append(np.mean((fn(z, h, sigma2) - x) ** 2))
        hay_mean = float(np.mean(acc["Hay"]))
        methods_block = {}
        for name, errs in acc.items():
            m, s = float(np.mean(errs)), float(np.std(errs))
            methods_block[name] = {
                "mse_mean": m, "mse_std": s,
                "improve_over_hay_pct": (100 * (hay_mean - m) / hay_mean
                                         if hay_mean > 0 else 0.0),
            }
        results.append({
            "spec": {"kind": "census_ddhca", "name": f"pop{meta['popgroup']}"},
            "spec_name": "census_ddhca",
            "popgroup": meta["popgroup"], "popgroup_label": meta["popgroup_label"],
            "signal_type": "real", "leaf_mean": None,
            "epsilon": eps, "num_nodes": int(h.num_nodes),
            "num_trials": NUM_TRIALS, "sigma2": sigma2,
            "methods": methods_block,
        })
        print(f"  [{time.time()-t0:6.1f}s] eps={eps:>5}: "
              f"Hay={hay_mean:.2f}, Ours={methods_block['Ours']['mse_mean']:.2f} "
              f"(improve {methods_block['Ours']['improve_over_hay_pct']:+.1f}%)")

    safe = str(meta["popgroup"]).replace("/", "-")
    out = os.path.join(CENSUS_DDHCA_RESULTS_DIR, f"ddhca_results_pop{safe}.json")
    with open(out, "w") as f:
        json.dump({"config": {"methods": list(methods_with_oracle),
                              "epsilons": EPSILONS, "num_trials": NUM_TRIALS,
                              "source": "2020 Detailed DHC-A T01001, county, proxy GT",
                              "popgroup": meta["popgroup"],
                              "popgroup_label": meta["popgroup_label"]},
                   "results": results}, f, indent=1)
    print(f"\nwrote {len(results)} configs -> {out}")


if __name__ == "__main__":
    import sys
    # A bare `python T01001_census_loader.py` runs the FULL experiment (load ->
    # add noise -> all methods -> results JSON) for every group in
    # SELECTED_POPGROUPS, writing one results file per group to
    # CENSUS_DDHCA_RESULTS_DIR and one archive per group to CENSUS_DDHCA_HIER_DIR.
    #   python T01001_census_loader.py                 # all SELECTED_POPGROUPS
    #   python T01001_census_loader.py 1190 2031       # only these group codes
    #   python T01001_census_loader.py --no-spectral   # skip the slow spectral baselines
    raw = sys.argv[1:]
    with_spectral = "--no-spectral" not in raw
    groups = [a for a in raw if a != "--no-spectral"] or SELECTED_POPGROUPS

    print(f"DDHC-A T01001 experiment | CSV: {DDHCA_CSV}")
    print(f"groups: {groups} | spectral baselines: {with_spectral} "
          f"| {NUM_TRIALS} trials x {len(EPSILONS)} eps each")
    t0 = time.time()
    for i, pg in enumerate(groups, 1):
        print(f"\n===== [{i}/{len(groups)}] POPGROUP {pg} =====")
        run_ddhca(DDHCA_CSV, pg, with_spectral=with_spectral)
    print(f"\nAll {len(groups)} group(s) done in {time.time()-t0:.1f}s "
          f"-> {CENSUS_DDHCA_RESULTS_DIR}/")
