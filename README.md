# Scalar, Level, or Node? Choosing Shrinkage Granularity for Differentially Private Hierarchical Counts

# Statement: The repository code has been optimized and debugged using Claude.

Code, data archives, and experiment results for the paper. Everything in the
paper — every number, figure, and table — is regenerated from the archived
results by the scripts in `core/`.

**What this is.** Differentially private releases of hierarchical counts
(e.g., nation → state → county populations) are routinely post-processed with
consistency projection (Hay et al., 2010). A biased _shrinkage_ step can
reduce error further at no privacy cost, but the shrinkage coefficient can be
estimated at three granularities: one per tree (**scalar**), one per level
(**level-wise**), or one per cell (**per-node**). This repository contains the
full experimental study of _which granularity to use_, on synthetic trees and
on two 2020 U.S. Census releases, plus a zero-cost diagnostic that selects the
granularity from the released data alone.

## Requirements

- Python ≥ 3.10 (tested on 3.12)
- `numpy`, `networkx`, `matplotlib`

```bash
pip install numpy networkx matplotlib
```

## Repository layout

```
core/                       all code (run every script FROM this directory)
  hierarchy_data_generator.py   tree structures + golden-master reference archive
  signal_generator.py           synthetic ground-truth signals (+ archive check)
  laplace_noise_injection.py    calibrated Laplace noise -> DP release archives
  projection.py                 Hay two-pass consistency projection + exact LS
  shrinkage.py                  level-wise shrinkage (the LEVEL granularity)
  ablations.py                  SCALAR and NODE granularities + ablations
  spectral.py                   graph-spectral (GFT/Tikhonov + SURE) reference
  baseline.py                   unified method registry
  runner.py                     synthetic experiment grid (504 configurations)
  P1_census_loader.py           PL94 P1 county populations (high-SNR control)
  T01001_census_loader.py       Detailed DHC-A per-group counts (sparse regime)
  stratified_eval.py            evaluation suite on the archived draws: cell-type
                                error split, threshold variants, paired error bars
  diagnostic_check.py           granularity-selection rule + held-out validation
  make_paper_assets.py          regenerates every paper figure and table
synthetic_data/             archived signals and DP noise draws (golden-master)
census_data/
  original_data/P1/             2020 Census PL94-171 Table P1 CSV
  original_data/T01001/         2020 Census Detailed DHC-A Table T01001 CSV
  census_hierarchy_format/      tree-structured archives built from the CSVs
results/
  synthetic_data_results/       results.json (synthetic grid)
  P1_census_results/            PL94 control results
  census_ddhca_results/         per-group DDHC-A results + stratified split
  diagnostic_validation.json    diagnostic rule validation
  paper_assets/                 all paper figures (.pdf/.png) and tables (.tex)
```

## Reproducing the paper

All commands are run from `core/`. Steps 1–3 verify the shipped archives
byte-for-byte (golden-master checks) rather than regenerating them; the
experiment scripts (4–8) recompute results from the archives; step 9 rebuilds
every figure and table from the result JSONs.

```bash
cd core

# 1-3. verify (or rebuild from scratch) the data archives
python hierarchy_data_generator.py      # tree structures
python signal_generator.py              # ground-truth signals
python laplace_noise_injection.py       # DP noise draws

# 4. synthetic grid: 9 trees x 4 signal classes x 2 scales x 7 epsilons x 50 trials
python runner.py                        # hours (includes spectral baselines)

# 5. PL94 P1 county populations (high-SNR control)
python P1_census_loader.py              # add --no-spectral for a fast pass

# 6. Detailed DHC-A, five population groups spanning the sparsity range
python T01001_census_loader.py          # ~1h with spectral; --no-spectral ~minutes

# 7. evaluation suite (reuses the exact draws of steps 5-6; cross-checks its
#    aggregates against step 6's outputs): cell-type-stratified errors,
#    universal-threshold per-node variants, paired error bars
python stratified_eval.py               # ~7 min

# 8. granularity diagnostic: tuned on balanced synthetic configs only,
#    validated on held-out unbalanced + census configs
python diagnostic_check.py              # seconds

# 9. regenerate every paper figure and table into results/paper_assets/
python make_paper_assets.py             # seconds
```

Every module in `core/` also runs a self-test when executed directly
(`python projection.py`, `python shrinkage.py`, ...), asserting the key
invariants (Hay two-pass == exact least squares on balanced trees, projection
non-expansiveness, coefficient calibration, archive reproducibility).

## Conference (CCECE) version: exact projection + PL94 P1 sparse columns

The conference version uses the exact least-squares projection everywhere (so
the projection lemmas apply verbatim; balanced trees are unchanged, only the two
unbalanced synthetic trees move) and replaces the Detailed DHC-A groups by six
sparse multi-race-combination columns of the PL94 P1 table, which are published
for every county with no threshold (see the note in `T01001_census_loader.py`
on why the DDHC-A zeros are imputed). Everything runs in minutes:

```bash
cd core
python runner.py --exact-ls --no-spectral       # synthetic grid -> results_exactls.json (~10 s)
python P1_sparse_columns.py --exact-ls           # six P1 columns -> P1_sparse_results/ (~10 min)
python P1_sparse_columns.py --columns P1_001N --exact-ls   # total-population control row
python diagnostic_check.py --exact-ls --p1       # diagnostic -> diagnostic_validation_exactls.json
python make_ccece_assets.py                      # IEEE tables/figures -> results/ccece_assets/
python sensitivity_absent_counties.py            # (arXiv version) DDHC-A absent-county sensitivity
```

`python P1_sparse_columns.py --list` ranks all 71 P1 columns by the expected
leaf-level coefficient to choose a different regime-spanning set. The projection
is a run-wide switch (`projection.set_projection("hay"|"ls")`); the WPES
commands above keep their Hay two-pass default and reproduce the archived
results byte for byte.

## Reproducibility notes

- **Determinism.** All noise draws use fixed, collision-free seed schedules;
  the synthetic draws are additionally shipped as archives under
  `synthetic_data/`, and re-running the generators verifies them with a
  golden-master check. `stratified_eval.py` regenerates the census draws from
  the same seeds as the main experiment and asserts that the recomputed
  aggregate MSEs match the published result files exactly.
- **Census data.** The two CSVs are public 2020 Census products downloaded
  from [data.census.gov](https://data.census.gov): Redistricting Data
  (P.L. 94-171) Table P1, and Detailed DHC-A Table T01001 (all population
  groups, county level). Published counts are used as a proxy ground truth
  (they are themselves DP-protected), re-aggregated for consistency, with
  fresh calibrated noise injected on top — the standard evaluation protocol
  for census post-processing research.
- **Methods registry.** All estimators share one signature
  `method(z, hierarchy, sigma2) -> x_hat` and are registered in
  `baseline.METHODS`: raw release, zero, Hay projection, the three shrinkage
  granularities, ablations (no projection / no small-level guard), and the
  spectral references (SURE-tuned and oracle-tuned Tikhonov filters).

## License

See [LICENSE](LICENSE).
