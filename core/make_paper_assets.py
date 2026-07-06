"""
Generate every figure and LaTeX table used in the paper, straight from the
result JSONs, into ../results/paper_assets/ with self-describing names:

  fig_synthetic_granularity_vs_eps.pdf/.png   (paper fig:synthetic-grid)
      Improvement over Hay vs epsilon, one facet per signal class, one curve
      per granularity. Source: synthetic_data_results/results.json.
  fig_ddhca_envelope.pdf/.png                 (optional companion to tab:ddhca)
      Same quantity on the five DDHC-A groups (Level per group; Node per group).
  tab_ddhca_improvement.tex                   (paper tab:ddhca)
      Improvement over Hay (%), groups x epsilons, Level and Node rows,
      plus the PL94 P1 high-SNR control row.
  tab_stratified_main.tex                     (paper tab:stratified)
      Present/empty-cell MSE for Hay/Level/Node, two representative groups
      at four epsilons. Source: census_ddhca_results/stratified_results.json.
  tab_stratified_full.tex                     (appendix version: all groups/eps)
  tab_diagnostic_validation.tex               (paper Section sec:diagnostic)
      Accuracy/regret of the penalized three-way rule (and the two-way
      ablation) on tuning vs held-out splits. Source: diagnostic_validation.json.

Regenerate everything with:  python make_paper_assets.py     (from core/)
No experiment is re-run; this script only reads the archived results.
"""
from __future__ import annotations
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SYN_FILE = "../results/synthetic_data_results/results.json"
DDHCA_DIR = "../results/census_ddhca_results"
STRAT_FILE = os.path.join(DDHCA_DIR, "stratified_results.json")
DIAG_FILE = "../results/diagnostic_validation.json"
P1_CANDIDATES = ["../results/P1_census_results/P1_census_results.json",
                 "../results/census_results/census_results.json"]
OUT_DIR = "../results/paper_assets"

# granularity -> (name in result files, display label, color, linestyle)
GRANS = [
    ("Scalar",  "Scalar",     "#009E73", ":"),
    ("Ours",    "Level-wise", "#0072B2", "-"),
    ("PerNode", "Per-node",   "#D55E00", "--"),
]
SIGNAL_ORDER = ["uniform", "smooth", "random", "sparse"]
EPS_ORDER = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 110, "savefig.bbox": "tight",
})


def _save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"{stem}.{ext}"),
                    dpi=220 if ext == "png" else None)
    plt.close(fig)
    print(f"  wrote {stem}.pdf/.png")


def fmt_mse(v: float) -> str:
    if v >= 1000:
        return f"{v:,.0f}"
    if v >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


def fmt_pct(v: float) -> str:
    """One-decimal percentage, avoiding the ugly '-0.0'."""
    return "0.0" if abs(v) < 0.05 else f"{v:.1f}"


# ---------------------------------------------------------------- synthetic
def fig_synthetic():
    with open(SYN_FILE) as f:
        results = json.load(f)["results"]
    # mean improvement over Hay per (signal, eps, method), across trees & scales
    agg = defaultdict(list)
    for r in results:
        for key, *_ in GRANS:
            agg[(r["signal_type"], r["epsilon"], key)].append(
                r["methods"][key]["improve_over_hay_pct"])

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6), sharex=True, sharey=True)
    for ax, st in zip(axes.flat, SIGNAL_ORDER):
        for key, label, color, ls in GRANS:
            y = [np.mean(agg[(st, e, key)]) for e in EPS_ORDER]
            ax.plot(EPS_ORDER, y, ls, color=color, marker="o", ms=3.5,
                    label=label)
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xscale("log")
        ax.set_title(f"{st} signals")
        ax.set_ylim(-15, 100)
        ax.grid(True, which="both", alpha=0.25, lw=0.4)
    for ax in axes[1]:
        ax.set_xlabel(r"privacy budget $\varepsilon$")
    for ax in axes[:, 0]:
        ax.set_ylabel("improvement over Hay (%)")
    axes[0, 0].legend(loc="lower left", frameon=False)
    fig.suptitle("Shrinkage granularity vs. within-level structure "
                 "(mean over 9 trees × 2 scales, 50 trials)", y=1.02)
    _save(fig, "fig_synthetic_granularity_vs_eps")


# ------------------------------------------------------------------ DDHC-A
def _load_ddhca():
    groups = []
    for fn in sorted(os.listdir(DDHCA_DIR)):
        if fn.startswith("ddhca_results_pop") and fn.endswith(".json"):
            with open(os.path.join(DDHCA_DIR, fn)) as f:
                groups.append(json.load(f))
    # sort sparsest first (by total population implied via config label order)
    order = {"1190": 0, "1016": 1, "2031": 2, "3657": 3, "1009": 4}
    groups.sort(key=lambda g: order.get(g["config"]["popgroup"], 99))
    return groups


def _short_label(label: str) -> str:
    return label.replace(" alone or in any combination", " (comb.)") \
                .replace(" alone", "")


def fig_ddhca(groups):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(groups)))
    for ax, key, title in [(axes[0], "Ours", "Level-wise"),
                           (axes[1], "PerNode", "Per-node")]:
        for g, color in zip(groups, cmap):
            eps = [r["epsilon"] for r in g["results"]]
            y = [r["methods"][key]["improve_over_hay_pct"] for r in g["results"]]
            ax.plot(eps, y, "-o", ms=3, color=color,
                    label=_short_label(g["config"]["popgroup_label"]))
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xscale("log")
        ax.set_title(f"{title} shrinkage")
        ax.set_xlabel(r"privacy budget $\varepsilon$")
        ax.grid(True, which="both", alpha=0.25, lw=0.4)
    axes[0].set_ylabel("improvement over Hay (%)")
    axes[0].legend(frameon=False, fontsize=7, loc="lower left")
    fig.suptitle("Detailed DHC-A groups, sparsest (dark) to densest (light)",
                 y=1.04)
    _save(fig, "fig_ddhca_envelope")


def tab_ddhca(groups):
    p1 = None
    for cand in P1_CANDIDATES:
        if os.path.exists(cand):
            with open(cand) as f:
                p1 = json.load(f)
            break
    lines = [
        "% Auto-generated by make_paper_assets.py -- do not edit by hand.",
        "% Improvement over Hay (%), per group and epsilon. Paper tab:ddhca.",
        r"\begin{table*}[t]",
        r"\centering\small",
        r"\caption{Improvement in per-cell MSE over consistency projection"
        r" (\%). Groups ordered sparsest to densest; PL94 total population is"
        r" the high-SNR control.}",
        r"\label{tab:ddhca}",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Release & Granularity & " +
        " & ".join(rf"$\varepsilon{{=}}{e:g}$" for e in EPS_ORDER) + r" \\",
        r"\midrule",
    ]
    for g in groups:
        label = _short_label(g["config"]["popgroup_label"])
        per_eps = {r["epsilon"]: r["methods"] for r in g["results"]}
        for key, disp in [("Ours", "Level"), ("PerNode", "Node")]:
            vals = " & ".join(fmt_pct(per_eps[e][key]["improve_over_hay_pct"])
                              for e in EPS_ORDER)
            head = label if key == "Ours" else ""
            lines.append(rf"{head} & {disp} & {vals} \\")
        lines.append(r"\addlinespace[2pt]")
    if p1 is not None:
        per_eps = {r["epsilon"]: r["methods"] for r in p1["results"]}
        for key, disp in [("Ours", "Level"), ("PerNode", "Node")]:
            vals = " & ".join(fmt_pct(per_eps[e][key]["improve_over_hay_pct"])
                              for e in EPS_ORDER)
            head = "PL94 P1 (control)" if key == "Ours" else ""
            lines.append(rf"{head} & {disp} & {vals} \\")
    else:
        print("  !! P1 results not found; control row omitted")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    path = os.path.join(OUT_DIR, "tab_ddhca_improvement.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("  wrote tab_ddhca_improvement.tex")


# -------------------------------------------------------------- stratified
def tab_stratified():
    with open(STRAT_FILE) as f:
        payload = json.load(f)
    rows = payload["results"]
    by_group = defaultdict(dict)
    labels = {}
    for r in rows:
        by_group[r["popgroup"]][r["epsilon"]] = r
        labels[r["popgroup"]] = _short_label(r["popgroup_label"])

    def table(group_eps_pairs, stem, caption, label, star=False):
        env = "table*" if star else "table"
        lines = [
            "% Auto-generated by make_paper_assets.py -- do not edit by hand.",
            rf"\begin{{{env}}}[t]",
            r"\centering\small",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\begin{tabular}{llrrrrrr}",
            r"\toprule",
            r" & & \multicolumn{3}{c}{populated cells}"
            r" & \multicolumn{3}{c}{empty cells} \\",
            r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
            r"Group & $\varepsilon$ & Hay & Level & Node"
            r" & Hay & Level & Node \\",
            r"\midrule",
        ]
        last_pg = None
        for pg, eps in group_eps_pairs:
            r = by_group[pg][eps]
            m = r["methods"]
            head = labels[pg] if pg != last_pg else ""
            last_pg = pg
            cells = [fmt_mse(m[k]["mse_present"]) for k in ("Hay", "Level", "Node")]
            cells += [fmt_mse(m[k]["mse_empty"]) for k in ("Hay", "Level", "Node")]
            lines.append(rf"{head} & {eps:g} & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", rf"\end{{{env}}}"]
        with open(os.path.join(OUT_DIR, stem + ".tex"), "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"  wrote {stem}.tex")

    main_pairs = [(pg, e) for pg in ("1190", "3657")
                  for e in (0.1, 0.5, 1.0, 2.0)]
    table(main_pairs, "tab_stratified_main",
          "Cell-type-stratified MSE on two Detailed DHC-A groups (same draws "
          "as Table~\\ref{tab:ddhca}). Level-wise trades the populated cells "
          "for the empty ones; per-node cleans the empty cells while staying "
          "near the unbiased baseline on the populated ones.",
          "tab:stratified")
    full_pairs = [(pg, e) for pg in ("1190", "1016", "2031", "3657", "1009")
                  for e in EPS_ORDER]
    table(full_pairs, "tab_stratified_full",
          "Cell-type-stratified MSE, all Detailed DHC-A groups and privacy "
          "budgets (full version of Table~\\ref{tab:stratified}).",
          "tab:stratified-full", star=True)


# -------------------------------------------------------------- diagnostic
def tab_diagnostic():
    with open(DIAG_FILE) as f:
        d = json.load(f)

    def row(name, ev):
        return (rf"{name} & {ev['n_decisive']}/{ev['n_configs']}"
                rf" & {100*ev['accuracy_decisive']:.1f}"
                rf" & {ev['mean_regret_pct']:.1f}"
                rf" & {ev['median_regret_pct']:.1f} \\")

    b = d["rule_b"]; a = d["rule_a"]
    lines = [
        "% Auto-generated by make_paper_assets.py -- do not edit by hand.",
        r"\begin{table}[t]",
        r"\centering\small",
        r"\caption{Validation of the granularity diagnostic"
        rf" (Eq.~\ref{{eq:diagnostic}}, $\kappa={b['kappa']:g}$ tuned on the"
        r" balanced split only). Accuracy is over decisive configurations;"
        r" regret is MSE(chosen)/MSE(best)$-$1 over all configurations. The"
        rf" two-way variant ($\tau={a['tau']:g}$) cannot select"
        r" \textsc{scalar}, which costs it the unbalanced trees.}",
        r"\label{tab:diagnostic}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Split & decisive & acc.\ (\%) & \multicolumn{2}{c}{regret (\%)} \\",
        r" & & & mean & median \\",
        r"\midrule",
        r"\multicolumn{5}{l}{\emph{Three-way penalized rule (ours)}} \\",
        row("~~balanced (tuning)", b["balanced"]),
        row("~~unbalanced (held out)", b["unbalanced_heldout"]),
        row("~~census DDHC-A (held out)", b["census_heldout"]),
        r"\addlinespace[2pt]",
        r"\multicolumn{5}{l}{\emph{Two-way rule (ablation)}} \\",
        row("~~balanced (tuning)", a["balanced"]),
        row("~~unbalanced (held out)", a["unbalanced_heldout"]),
        row("~~census DDHC-A (held out)", a["census_heldout"]),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    with open(os.path.join(OUT_DIR, "tab_diagnostic_validation.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("  wrote tab_diagnostic_validation.tex")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"paper assets -> {OUT_DIR}")
    fig_synthetic()
    groups = _load_ddhca()
    fig_ddhca(groups)
    tab_ddhca(groups)
    tab_stratified()
    tab_diagnostic()
    print("done.")
