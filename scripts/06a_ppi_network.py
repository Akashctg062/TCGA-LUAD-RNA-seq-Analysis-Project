"""
06a_ppi_network.py
===================
PPI network analysis using STRING database API.
Identifies final hub genes from WGCNA + DEG intersection.

Usage:
    python scripts/06a_ppi_network.py

Output:
    results/final_hub_genes.csv
    results/ppi_interactions.csv
    figures/ppi_network.png
    figures/hub_gene_ranking.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import requests
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"
FIG_DIR = PROJECT_DIR / "figures"

STRING_API = "https://version-12-0.string-db.org/api"
SPECIES_ID = 9606  # Homo sapiens

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.size": 11, "axes.titlesize": 13,
})


def select_candidate_hub_genes():
    """Combine WGCNA hub genes with DEG results to create final candidates."""
    print("=" * 60)
    print("Step 1: Selecting candidate hub genes...")
    print("=" * 60)

    hub_df = pd.read_csv(RESULTS_DIR / "wgcna_hub_genes.csv")
    deg_df = pd.read_csv(RESULTS_DIR / "deg_all_results.csv", index_col=0)

    # Filter: must be DEG, high MM (>0.8), and reasonable GS (>0.2)
    candidates = hub_df[
        (hub_df["is_DEG"] == True) &
        (hub_df["MM"] > 0.8) &
        (hub_df["GS"] > 0.2)
    ].copy()

    # Add DEG info (log2FC, padj)
    deg_info = deg_df[["log2FoldChange", "padj", "gene_name"]].copy()
    deg_info = deg_info.reset_index().rename(columns={"index": "gene_id"})

    candidates = candidates.merge(
        deg_info[["gene_id", "log2FoldChange", "padj"]],
        on="gene_id", how="left"
    )

    # Score: combine MM, GS, and significance
    candidates["hub_score"] = (
        candidates["MM"] * 0.4 +
        candidates["GS"] * 0.4 +
        (-np.log10(candidates["padj"].clip(lower=1e-300)) / 300) * 0.2
    )
    candidates = candidates.sort_values("hub_score", ascending=False)

    # Take top 50 for PPI analysis
    top_candidates = candidates.head(50)

    print(f"  Total WGCNA hubs: {len(hub_df)}")
    print(f"  After filtering (DEG + MM>0.8 + GS>0.2): {len(candidates)}")
    print(f"  Top 50 selected for PPI analysis")
    print(f"\n  Top 10 candidates:")
    for _, row in top_candidates.head(10).iterrows():
        print(f"    {str(row['gene_name']):15s}  M{int(row['module'])}  "
              f"MM={row['MM']:.3f}  GS={row['GS']:.3f}  "
              f"log2FC={row['log2FoldChange']:+.2f}")

    return top_candidates, candidates


def query_string_ppi(gene_list):
    """Query STRING database for PPI interactions."""
    print("\n" + "=" * 60)
    print("Step 2: Querying STRING database for PPI...")
    print("=" * 60)

    gene_names = [str(g) for g in gene_list if pd.notna(g)]

    # Get STRING identifiers
    print(f"  Querying {len(gene_names)} genes...")

    try:
        # Get interactions
        params = {
            "identifiers": "%0d".join(gene_names),
            "species": SPECIES_ID,
            "required_score": 400,  # Medium confidence (0.4)
            "network_type": "physical",
            "caller_identity": "tcga_luad_analysis",
        }

        url = f"{STRING_API}/json/network"
        response = requests.post(url, data=params, timeout=60)
        response.raise_for_status()
        interactions = response.json()

        print(f"  Retrieved {len(interactions)} interactions")

        # Parse interactions
        edges = []
        for inter in interactions:
            edges.append({
                "gene_a": inter.get("preferredName_A", inter.get("stringId_A", "")),
                "gene_b": inter.get("preferredName_B", inter.get("stringId_B", "")),
                "score": inter.get("score", 0),
            })

        return pd.DataFrame(edges)

    except Exception as e:
        print(f"  STRING API error: {e}")
        print("  Falling back to correlation-based network...")
        return None


def build_correlation_network(candidates):
    """Fallback: build network from gene co-expression correlations."""
    print("  Building correlation-based PPI proxy network...")

    data_dir = PROJECT_DIR / "data"
    counts = pd.read_csv(data_dir / "filtered_counts.csv", index_col=0)

    gene_ids = candidates["gene_id"].tolist()
    available = [g for g in gene_ids if g in counts.index]

    # log2 CPM
    lib_sizes = counts.sum(axis=0)
    cpm = counts.div(lib_sizes, axis=1) * 1e6
    log_cpm = np.log2(cpm + 1)

    expr = log_cpm.loc[available].T

    # Correlation matrix
    cor = expr.corr()

    # Gene ID to name mapping
    gene_map = dict(zip(candidates["gene_id"], candidates["gene_name"]))

    # Create edges for strong correlations
    edges = []
    gene_list = cor.columns.tolist()
    for i in range(len(gene_list)):
        for j in range(i+1, len(gene_list)):
            r = cor.iloc[i, j]
            if abs(r) > 0.5:  # Strong correlation
                edges.append({
                    "gene_a": gene_map.get(gene_list[i], gene_list[i]),
                    "gene_b": gene_map.get(gene_list[j], gene_list[j]),
                    "score": abs(r),
                })

    return pd.DataFrame(edges)


def analyze_network(edge_df, candidates):
    """Build networkx graph and compute centrality metrics."""
    print("\n" + "=" * 60)
    print("Step 3: Analyzing network topology...")
    print("=" * 60)

    G = nx.Graph()

    # Add nodes (all candidates)
    gene_names = candidates["gene_name"].dropna().unique()
    for g in gene_names:
        G.add_node(g)

    # Add edges
    if edge_df is not None and not edge_df.empty:
        for _, row in edge_df.iterrows():
            if row["gene_a"] in G.nodes and row["gene_b"] in G.nodes:
                G.add_edge(row["gene_a"], row["gene_b"], weight=row["score"])

    # Remove isolated nodes for cleaner visualization
    isolated = list(nx.isolates(G))
    G.remove_nodes_from(isolated)

    if len(G.nodes) == 0:
        print("  WARNING: No connected nodes in network")
        return None, pd.DataFrame()

    print(f"  Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  Isolated nodes removed: {len(isolated)}")

    # Centrality metrics
    degree = dict(G.degree())
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)

    # PageRank
    try:
        pagerank = nx.pagerank(G)
    except:
        pagerank = {n: 0 for n in G.nodes}

    centrality_df = pd.DataFrame({
        "gene_name": list(G.nodes),
        "degree": [degree[n] for n in G.nodes],
        "betweenness": [betweenness[n] for n in G.nodes],
        "closeness": [closeness[n] for n in G.nodes],
        "pagerank": [pagerank[n] for n in G.nodes],
    })

    # Rank by each metric, compute combined rank
    for col in ["degree", "betweenness", "closeness", "pagerank"]:
        centrality_df[f"{col}_rank"] = centrality_df[col].rank(ascending=False)

    centrality_df["avg_rank"] = centrality_df[
        ["degree_rank", "betweenness_rank", "closeness_rank", "pagerank_rank"]
    ].mean(axis=1)

    centrality_df = centrality_df.sort_values("avg_rank")

    # Top hub genes
    print(f"\n  Top 10 network hub genes:")
    for _, row in centrality_df.head(10).iterrows():
        print(f"    {row['gene_name']:15s}  degree={int(row['degree']):3d}  "
              f"betweenness={row['betweenness']:.3f}  pagerank={row['pagerank']:.4f}")

    return G, centrality_df


def plot_ppi_network(G, centrality_df, candidates):
    """Visualize PPI network with hub genes highlighted."""
    print("\n  Plotting PPI network...")

    if G is None or len(G.nodes) < 2:
        print("    Skipped: insufficient network data")
        return

    fig, ax = plt.subplots(figsize=(14, 12))

    # Layout
    if len(G.nodes) > 50:
        pos = nx.spring_layout(G, k=2.0, iterations=100, seed=42)
    else:
        pos = nx.spring_layout(G, k=3.0, iterations=150, seed=42)

    # Module colors for nodes
    module_map = dict(zip(candidates["gene_name"], candidates["module"]))
    module_colors_palette = {
        1: "#FF6B6B", 2: "#FFA07A", 3: "#98D8C8",
        4: "#87CEEB", 5: "#DDA0DD", 6: "#778899", 7: "#FF69B4",
    }

    node_colors = []
    for node in G.nodes:
        mod = module_map.get(node, 0)
        node_colors.append(module_colors_palette.get(mod, "#cccccc"))

    # Node sizes based on degree
    degrees = dict(G.degree())
    max_deg = max(degrees.values()) if degrees else 1
    node_sizes = [300 + (degrees[n] / max_deg) * 1500 for n in G.nodes]

    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.2, width=0.5, edge_color="#999999")

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, alpha=0.85, edgecolors="white", linewidths=1.5)

    # Labels for top hub genes
    top_hubs = set(centrality_df.head(15)["gene_name"])
    labels = {n: n for n in G.nodes if n in top_hubs}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8, font_weight="bold")

    # Legend for modules
    legend_handles = []
    for mod, color in sorted(module_colors_palette.items()):
        mod_genes = [n for n in G.nodes if module_map.get(n, 0) == mod]
        if mod_genes:
            legend_handles.append(mpatches.Patch(color=color, label=f"Module {mod} ({len(mod_genes)})"))

    ax.legend(handles=legend_handles, loc="upper left", fontsize=9, title="WGCNA Module")
    ax.set_title(f"Protein-Protein Interaction Network\n({G.number_of_nodes()} genes, {G.number_of_edges()} interactions)",
                 fontsize=14)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "ppi_network.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: figures/ppi_network.png")


def plot_hub_ranking(centrality_df):
    """Bar plot of top hub genes by network centrality."""
    print("  Plotting hub gene ranking...")

    top = centrality_df.head(20).copy()
    top = top.sort_values("degree", ascending=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 8))

    for i, (metric, color) in enumerate([
        ("degree", "#e74c3c"),
        ("betweenness", "#3498db"),
        ("pagerank", "#2ecc71")
    ]):
        axes[i].barh(top["gene_name"], top[metric], color=color, alpha=0.8, edgecolor="white")
        axes[i].set_xlabel(metric.capitalize())
        axes[i].set_title(f"Top 20 by {metric.capitalize()}")
        axes[i].spines[["top", "right"]].set_visible(False)

    plt.suptitle("Hub Gene Network Centrality Ranking", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "hub_gene_ranking.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: figures/hub_gene_ranking.png")


def main():
    print("\n" + "=" * 60)
    print("  Phase 6A: PPI Network Analysis")
    print("=" * 60 + "\n")

    # Select candidates
    top_candidates, all_candidates = select_candidate_hub_genes()

    # Query STRING
    gene_list = top_candidates["gene_name"].dropna().tolist()
    edge_df = query_string_ppi(gene_list)

    # Fallback to correlation network if STRING fails
    if edge_df is None or edge_df.empty:
        edge_df = build_correlation_network(top_candidates)

    # Analyze network
    G, centrality_df = analyze_network(edge_df, top_candidates)

    # Save PPI interactions
    if edge_df is not None and not edge_df.empty:
        edge_df.to_csv(RESULTS_DIR / "ppi_interactions.csv", index=False)

    # Merge centrality with candidate info
    if not centrality_df.empty:
        final_hubs = centrality_df.merge(
            top_candidates[["gene_name", "gene_id", "module", "MM", "GS",
                           "log2FoldChange", "padj", "hub_score"]],
            on="gene_name", how="left"
        )
        final_hubs.to_csv(RESULTS_DIR / "final_hub_genes.csv", index=False)
        print(f"\n  Saved: results/final_hub_genes.csv ({len(final_hubs)} genes)")

        # Plots
        plot_ppi_network(G, centrality_df, top_candidates)
        plot_hub_ranking(centrality_df)
    else:
        # If no network, just save the top candidates as hub genes
        top_candidates.to_csv(RESULTS_DIR / "final_hub_genes.csv", index=False)
        print(f"\n  Saved: results/final_hub_genes.csv ({len(top_candidates)} genes)")

    print(f"\n  Phase 6A complete!")


if __name__ == "__main__":
    main()
