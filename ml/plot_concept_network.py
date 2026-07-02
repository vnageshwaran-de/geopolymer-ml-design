#!/usr/bin/env python3
"""
Regenerate Figure 2 — the concept co-occurrence network — from the archived
network data (data/concept_network_nodes.csv, data/concept_network_edges.csv).

The corpus stores title-level metadata (abstracts were not retained), so the
co-occurrence is title-based; the figure title reflects this ("titles", not
"abstracts").

Usage:
    python plot_concept_network.py \
        --nodes ../data/concept_network_nodes.csv \
        --edges ../data/concept_network_edges.csv \
        --out ../manuscript/figures/figure2_concept_network.png
"""
from __future__ import annotations
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import pandas as pd

SEED = 42

# Facet -> colour (matches the main-text description: ML methods in blue)
FACET_COLORS = {
    "Precursor":   "#8a8a2f",  # olive
    "Chemistry":   "#ff7f0e",  # orange
    "ML":          "#1f77b4",  # blue
    "Property":    "#2ca02c",  # green
    "Application": "#9467bd",  # purple
}
FACET_ORDER = ["Precursor", "Chemistry", "ML", "Property", "Application"]


def build_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    for _, r in nodes.iterrows():
        G.add_node(r["concept"], facet=r["facet"],
                   paper_freq=float(r["paper_freq"]))
    for _, r in edges.iterrows():
        if r["source"] in G and r["target"] in G:
            G.add_edge(r["source"], r["target"], weight=float(r["jaccard"]))
    return G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", default="../data/concept_network_nodes.csv")
    ap.add_argument("--edges", default="../data/concept_network_edges.csv")
    ap.add_argument("--out", default="../manuscript/figures/figure2_concept_network.png")
    args = ap.parse_args()

    nodes = pd.read_csv(args.nodes)
    edges = pd.read_csv(args.edges)
    G = build_graph(nodes, edges)

    # Spring layout weighted by Jaccard co-occurrence; fixed seed for reproducibility
    # Unweighted spring layout spreads the strongly-connected hub apart more
    # evenly than a Jaccard-weighted one (which collapses the core).
    pos = nx.spring_layout(G, weight=None, k=2.2, iterations=500, seed=SEED)

    freqs = nx.get_node_attributes(G, "paper_freq")
    facets = nx.get_node_attributes(G, "facet")
    # node area proportional to paper frequency
    fmax = max(freqs.values())
    sizes = [180 + 2600 * (freqs[n] / fmax) for n in G.nodes()]
    colors = [FACET_COLORS.get(facets[n], "#888888") for n in G.nodes()]

    weights = [G[u][v]["weight"] for u, v in G.edges()]
    wmax = max(weights)
    ewidths = [0.3 + 5.0 * (w / wmax) for w in weights]

    fig, ax = plt.subplots(figsize=(13.0, 10.4), dpi=170)
    nx.draw_networkx_edges(G, pos, width=ewidths, edge_color="#9a9a9a",
                           alpha=0.55, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=colors,
                           edgecolors="white", linewidths=1.2, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9.5, font_color="#111111", ax=ax)

    # No in-figure title/subtitle: the description is carried by the LaTeX
    # \caption in the manuscript (avoids duplication and edge truncation).

    # legend: facet colours + node-size note
    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=11,
                      markerfacecolor=FACET_COLORS[f], markeredgecolor="white",
                      label=f) for f in FACET_ORDER]
    handles.append(Line2D([0], [0], marker="o", linestyle="", markersize=6,
                          markerfacecolor="#888888", markeredgecolor="white",
                          label="Node size ∝ paper frequency"))
    ax.legend(handles=handles, title="Concept facet", loc="lower left",
              frameon=False, fontsize=10, title_fontsize=11)

    ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.out, dpi=170, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
