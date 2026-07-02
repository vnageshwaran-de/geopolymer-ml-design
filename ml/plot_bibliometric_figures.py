#!/usr/bin/env python3
"""
Regenerate the bibliometric figures as two upright, portrait-friendly figures:

  * figure1_trends.png       — publication trends by sub-stream (was Fig 1a)
  * figure2_composition.png  — ML methods, precursors, and functional themes
                               (was Fig 1b-d), stacked vertically

Trends are computed directly from data/geopolymer_ml_corpus.csv (year, is_ml,
is_onepart). The composition counts are the tabulated bibliometric facet counts
from the screening pass (same source as the original Figure 1); they are kept
here with the figure so the panels are reproducible.

Usage:
    python plot_bibliometric_figures.py \
        --corpus ../data/geopolymer_ml_corpus.csv \
        --outdir ../manuscript/figures
"""
from __future__ import annotations
import argparse, collections, csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_ALL, C_ML, C_OP, C_BOTH = "#7f7f7f", "#1f77b4", "#d62728", "#9467bd"

# --- tabulated bibliometric counts (read from the screening pass) -------------
ML_METHODS = [  # of 346 ML-tagged papers
    ("ANN / ML", 75), ("SHAP / interpret.", 40), ("Random forest", 33),
    ("SVM / SVR", 27), ("XGBoost / GBM", 18), ("Gene expr. prog.", 13),
    ("Deep learning", 11), ("Decision tree", 11),
]
PRECURSORS = [  # of 1850 on-topic papers
    ("Fly ash", 790), ("GGBS / slag", 682), ("Metakaolin", 177),
    ("Natural pozzolan", 45), ("Rice husk ash", 33), ("Waste glass", 20),
    ("Red mud", 18),
]
FUNCTIONS = [  # of 1850 on-topic papers
    ("Fibre-reinforced / ECC", 226), ("Thermal / fire", 131),
    ("Lightweight / foam", 92), ("Blast / ballistic / armour", 21),
    ("EM shielding / conductive", 21), ("3D printing", 19),
    ("Heavy-metal / immobilization", 19), ("Self-healing", 1),
]
# defense-relevant functional themes to highlight
HIGHLIGHT = {"Blast / ballistic / armour", "EM shielding / conductive"}


def load_trends(path):
    by = collections.defaultdict(lambda: [0, 0, 0, 0])
    tb = lambda v: str(v).strip().lower() in ("true", "1", "yes")
    for r in csv.DictReader(open(path)):
        try:
            y = int(float(r["year"]))
        except (ValueError, KeyError):
            continue
        ml, op = tb(r["is_ml"]), tb(r["is_onepart"])
        by[y][0] += 1; by[y][1] += ml; by[y][2] += op; by[y][3] += ml and op
    yrs = sorted(by)
    return yrs, {k: [by[y][i] for y in yrs] for i, k in
                 enumerate(["all", "ml", "op", "both"])}


def fig_trends(path, out):
    yrs, s = load_trends(path)
    fig, ax = plt.subplots(figsize=(7.4, 5.2), dpi=200)
    partial = max(yrs)  # 2026 retrieval is mid-year
    for key, col, lab, mk in [("all", C_ALL, "All geopolymer", "o"),
                              ("ml", C_ML, "ML-assisted", "s"),
                              ("op", C_OP, "One-part", "^"),
                              ("both", C_BOTH, "One-part × ML", "D")]:
        ax.plot(yrs, s[key], color=col, marker=mk, markersize=4.5,
                linewidth=2, label=lab)
    ax.axvspan(partial - 0.5, partial + 0.5, color="#000000", alpha=0.06)
    ax.text(partial, ax.get_ylim()[1] * 0.97, "2026\npartial", ha="center",
            va="top", fontsize=8, color="#666666")
    ax.set_xlabel("Year"); ax.set_ylabel("Publications / year")
    ax.set_title("Geopolymer research is accelerating; ML and one-part streams "
                 "rise fastest,\ntheir overlap only just emerging",
                 fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks(yrs); ax.tick_params(labelsize=8)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


def _hbar(ax, items, unit, title, base_color="#7f7f7f"):
    labels = [k for k, _ in items][::-1]
    vals = [v for _, v in items][::-1]
    colors = ["#e6820e" if k in HIGHLIGHT else base_color for k in labels]
    y = range(len(labels))
    ax.barh(list(y), vals, color=colors)
    ax.set_yticks(list(y)); ax.set_yticklabels(labels, fontsize=9)
    for i, v in zip(y, vals):
        ax.text(v + max(vals) * 0.01, i, str(v), va="center", fontsize=8.5)
    ax.set_xlabel(unit, fontsize=9)
    ax.set_title(title, fontsize=10.5, fontweight="bold", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, max(vals) * 1.12); ax.tick_params(labelsize=8)


def fig_composition(out):
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 11.4), dpi=200)
    _hbar(axes[0], ML_METHODS, "Papers mentioning method (of 346 ML papers)",
          "(a) ML methods: neural nets dominate; tree ensembles and SHAP the "
          "fast-growing second wave", base_color="#1f77b4")
    _hbar(axes[1], PRECURSORS, "Papers (of 1850)",
          "(b) Precursors: fly ash and slag are the workhorses; secondary "
          "streams still niche", base_color="#8a8a2f")
    _hbar(axes[2], FUNCTIONS, "Papers (of 1850)",
          "(c) Functional themes: fibre-reinforcement and thermal lead; "
          "defense-relevant functions (orange) remain under-studied",
          base_color="#7f7f7f")
    fig.tight_layout(h_pad=2.2); fig.savefig(out, bbox_inches="tight")
    plt.close(fig); print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="../data/geopolymer_ml_corpus.csv")
    ap.add_argument("--outdir", default="../manuscript/figures")
    a = ap.parse_args()
    fig_trends(a.corpus, os.path.join(a.outdir, "figure1_trends.png"))
    fig_composition(os.path.join(a.outdir, "figure2_composition.png"))


if __name__ == "__main__":
    main()
