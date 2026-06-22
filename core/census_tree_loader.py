"""
Load the real US census tree (nation -> state -> county) from a 2020 PL 94-171
P1 "Data" CSV, and treat the published populations as the ground-truth proxy x.

The downloaded PL94 counts are already TopDown/DP-processed, so they are not the
secret true counts; following standard census-research practice (e.g. BlueDown)
we use them as a ground-truth PROXY and add our OWN Laplace noise on top. State
and nation values are obtained by bottom-up summation, so x is consistent by
construction.

Output matches the synthetic pipeline's interface:
    load_census(csv) -> (Hierarchy, x)
so the same mechanism / projection / shrinkage / METHODS code applies unchanged.

GEO_ID format: "0500000US01001" -> after "US" is FIPS "01001"; first 2 = state
(01 = Alabama), full 5 = county. P1_001N column = total population.
"""
from __future__ import annotations
import csv
import json
import os
import time
import numpy as np

from hierarchy_data_generator import create_unbalanced_tree
from laplace_noise_injection import add_laplace_noise, noise_variance, EPSILONS, NUM_TRIALS
import baseline, ablations, spectral          # registers all methods
from baseline import METHODS
from spectral import make_spectral_oracle_method

CENSUS_RESULTS_DIR = "../results/census_results"
CENSUS_HIER_DIR = "../census_data/hierarchy_format"
CENSUS_CSV = "../census_data/original_data/DECENNIALPL2020.P1-Data.csv"


def load_census(csv_path: str, keep_puerto_rico: bool = True):
    """Build the (nation -> state -> county) Hierarchy and the proxy true signal x.
    Returns (hierarchy, x, meta)."""
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))
    header = [h.replace("\ufeff", "").strip().strip('"').strip() for h in rows[0]]
    gi, ni, pi = header.index("GEO_ID"), header.index("NAME"), header.index("P1_001N")

    counties = []   # (geoid, name, pop, state_fips, state_name)
    for r in rows[2:]:              # row 0 = names, row 1 = label descriptions
        if not r or not r[gi]:
            continue
        geoid = r[gi]
        fips = geoid.split("US")[-1]
        state_fips = fips[:2]
        name = r[ni]
        state_name = name.rsplit(",", 1)[-1].strip() if "," in name else state_fips
        if not keep_puerto_rico and state_fips == "72":
            continue
        pop = int(str(r[pi]).replace(",", "").strip())
        counties.append((geoid, name, pop, state_fips, state_name))

    # node numbering: 0 = nation; then states (sorted by fips); then counties
    state_fips_list = sorted({c[3] for c in counties})
    state_node = {sf: 1 + i for i, sf in enumerate(state_fips_list)}
    n_states = len(state_fips_list)
    county_node = {}
    edges = []
    nid = 1 + n_states
    for geoid, name, pop, sf, sn in counties:
        county_node[geoid] = nid
        edges.append((state_node[sf], nid))     # state -> county
        nid += 1
    for sf in state_fips_list:
        edges.append((0, state_node[sf]))        # nation -> state

    h = create_unbalanced_tree(edges, root=0)

    # build consistent x: counties = populations, sum up to states and nation
    x = np.zeros(h.num_nodes)
    for geoid, name, pop, sf, sn in counties:
        x[county_node[geoid]] = pop
    for sf in state_fips_list:
        cnode = state_node[sf]
        x[cnode] = sum(x[county_node[g]] for g, *_ , in
                       [(c[0],) for c in counties if c[3] == sf])
    x[0] = x[[state_node[sf] for sf in state_fips_list]].sum()

    # per-node label info (node id -> geoid/name/kind) for a readable archive
    node_info = {0: {"geoid": "0100000US", "name": "United States", "kind": "nation"}}
    seen_state = {}
    for geoid, name, pop, sf, sn in counties:
        seen_state.setdefault(sf, sn)
        node_info[county_node[geoid]] = {"geoid": geoid, "name": name, "kind": "county"}
    for sf, sn in seen_state.items():
        node_info[state_node[sf]] = {"geoid": f"0400000US{sf}", "name": sn, "kind": "state"}

    meta = {
        "n_counties": len(counties),
        "n_states": n_states,
        "total_population": int(x[0]),
        "state_names": {state_node[c[3]]: c[4] for c in counties},
        "node_info": node_info,
        "edges": [[int(u), int(v)] for u, v in edges],
    }
    return h, x, meta


def save_hierarchy(h, x, meta, out_dir: str = CENSUS_HIER_DIR) -> str:
    """Persist the treelized census data (structure + proxy ground-truth x +
    readable state/county labels) so the exact tree used in the experiments is
    reproducible and self-contained, without shipping the raw census CSV.
    Does NOT store noised z (a per-seed intermediate, regenerable). UTF-8 with
    ensure_ascii=False so names like 'Dona Ana County' are stored literally."""
    os.makedirs(out_dir, exist_ok=True)
    record = {
        "source": "2020 Census PL94-171, table P1 (P1_001N), county level",
        "note": "Published PL94 counts used as ground-truth PROXY (already "
                "TopDown/DP-processed); state/nation values are bottom-up sums.",
        "num_nodes": int(h.num_nodes),
        "n_states": meta["n_states"],
        "n_counties": meta["n_counties"],
        "height": int(h.height),
        "sensitivity": int(h.sensitivity),
        "total_population": meta["total_population"],
        "edges": meta["edges"],
        "levels": [int(v) for v in h.levels],
        "level_sizes": [int(h.num_nodes_at_level(l)) for l in range(h.depth)],
        "x": [float(v) for v in x],
        "node_info": {str(k): v for k, v in meta["node_info"].items()},
    }
    path = os.path.join(out_dir, "census_county_2020.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=1)
    return path


def load_hierarchy_archive(path: str):
    """Rebuild (Hierarchy, x) from a saved census_county_2020.json archive,
    without touching the original CSV."""
    with open(path, encoding="utf-8") as f:
        rec = json.load(f)
    h = create_unbalanced_tree([tuple(e) for e in rec["edges"]], root=0)
    x = np.array(rec["x"], dtype=float)
    return h, x, rec


def run_census(csv_path: str, with_spectral: bool = True) -> None:
    """Full census experiment: load -> add noise -> run all methods -> results JSON.
    Same output structure as the synthetic runner so plotting code is shared."""
    h, x, meta = load_census(csv_path)
    arch = save_hierarchy(h, x, meta)
    print(f"Saved treelized census archive -> {arch}")
    print(f"Census tree: {h.num_nodes} nodes "
          f"({meta['n_states']} states, {meta['n_counties']} counties), "
          f"height {h.height}, sensitivity {h.sensitivity}")
    print(f"Total population (proxy ground truth): {meta['total_population']:,}")
    print(f"Per-level node counts: "
          f"{[h.num_nodes_at_level(l) for l in range(h.depth)]}")

    methods = dict(METHODS)
    if not with_spectral:
        methods = {k: v for k, v in methods.items() if "Spectral" not in k}
    methods_with_oracle = dict(methods)
    if with_spectral:
        methods_with_oracle["Spectral(oracle)"] = make_spectral_oracle_method(x)

    os.makedirs(CENSUS_RESULTS_DIR, exist_ok=True)
    results = []
    t0 = time.time()
    for eps in EPSILONS:
        sigma2 = noise_variance(h, eps)
        acc = {name: [] for name in methods_with_oracle}
        for trial in range(NUM_TRIALS):
            rng = np.random.default_rng(900000 + trial)   # distinct seed space
            z = add_laplace_noise(x, h, eps, rng)
            for name, fn in methods_with_oracle.items():
                acc[name].append(np.mean((fn(z, h, sigma2) - x) ** 2))
        hay_mean = float(np.mean(acc["Hay"]))
        methods_block = {}
        for name, errs in acc.items():
            m, s = float(np.mean(errs)), float(np.std(errs))
            methods_block[name] = {
                "mse_mean": m, "mse_std": s,
                "improve_over_hay_pct": (100*(hay_mean-m)/hay_mean if hay_mean>0 else 0.0),
            }
        results.append({
            "spec": {"kind": "census", "name": "us_county_2020"},
            "spec_name": "census_county",
            "signal_type": "real", "leaf_mean": None,
            "epsilon": eps, "num_nodes": int(h.num_nodes),
            "num_trials": NUM_TRIALS, "sigma2": sigma2,
            "methods": methods_block,
        })
        print(f"  [{time.time()-t0:6.1f}s] eps={eps:>5}: "
              f"Hay={hay_mean:.1f}, Ours={methods_block['Ours']['mse_mean']:.1f} "
              f"(improve {methods_block['Ours']['improve_over_hay_pct']:+.1f}%)")

    out = os.path.join(CENSUS_RESULTS_DIR, "census_results.json")
    with open(out, "w") as f:
        json.dump({"config": {"methods": list(methods_with_oracle),
                              "epsilons": EPSILONS, "num_trials": NUM_TRIALS,
                              "source": "2020 PL94-171 P1, county level, proxy GT"},
                   "results": results}, f, indent=1)
    print(f"\nwrote {len(results)} configs -> {out}")


if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else CENSUS_CSV
    no_spec = "--no-spectral" in sys.argv
    run_census(csv_path, with_spectral=not no_spec)