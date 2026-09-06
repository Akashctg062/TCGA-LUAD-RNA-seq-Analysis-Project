"""
06c_immune_infiltration.py
===========================
Immune microenvironment analysis using ssGSEA-based deconvolution.
Correlates hub gene expression with immune cell infiltration.

Usage:
    python scripts/06c_immune_infiltration.py

Output:
    results/immune_scores.csv
    results/immune_hub_correlation.csv
    figures/immune_tumor_vs_normal.png
    figures/immune_hub_correlation_heatmap.png
    figures/immune_boxplot_by_gene.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr, mannwhitneyu
import warnings
warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
FIG_DIR = PROJECT_DIR / "figures"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.size": 10, "axes.titlesize": 12,
})

# ──────────────────────────────────────────────
# Immune cell gene sets (curated from literature)
# Bindea et al., Immunity 2013 + additional markers
# ──────────────────────────────────────────────
IMMUNE_GENE_SETS = {
    "B cells": ["BLK", "CD19", "CD79A", "CD79B", "FCRL2", "MS4A1", "TNFRSF17", "TCL1A", "SPIB", "PNOC"],
    "CD8+ T cells": ["CD8A", "CD8B", "GZMK", "GZMA", "PRF1", "GZMB", "NKG7", "IFNG"],
    "CD4+ T cells": ["CD4", "IL7R", "MAL", "LCK", "CD40LG"],
    "Tregs": ["FOXP3", "CCR8", "CTLA4", "IL2RA", "IKZF2", "TNFRSF18"],
    "NK cells": ["KIR2DL1", "KIR2DL3", "KIR2DL4", "KIR3DL1", "KIR3DL2", "KLRC1", "KLRD1", "NCR1", "NKG7"],
    "Monocytes": ["CD14", "CD163", "CSF1R", "FCGR1A", "FCGR3A", "LYZ"],
    "Macrophages M1": ["IRF5", "NOS2", "IL12A", "IL12B", "TNF", "PTGS2", "IL23A"],
    "Macrophages M2": ["CD163", "MRC1", "MSR1", "TGFB1", "IL10", "CCL18", "ARG1"],
    "Dendritic cells": ["BATF3", "IRF8", "ITGAX", "NRP1", "CD1C", "FLT3", "CCL17"],
    "Neutrophils": ["CEACAM8", "FPR1", "FCGR3B", "CSF3R", "S100A12", "CXCR1", "CXCR2"],
    "Mast cells": ["TPSAB1", "TPSB2", "CMA1", "MS4A2", "HDC", "CTSG"],
    "Eosinophils": ["CCR3", "SIGLEC8", "PRG2", "EPX", "RNASE2"],
    "T helper cells": ["TBX21", "STAT4", "GATA3", "STAT6", "RORC", "IL17A", "BCL6"],
    "Cytotoxic cells": ["GZMA", "GZMB", "GZMH", "PRF1", "GNLY", "NKG7", "KLRK1"],
    "Immune checkpoint": ["PDCD1", "CD274", "PDCD1LG2", "CTLA4", "LAG3", "HAVCR2", "TIGIT", "IDO1", "SIGLEC15"],
}


def ssgsea_score(expr_df, gene_set):
    """Compute ssGSEA-like enrichment score for a gene set.
    Simple implementation: mean z-score of gene set members."""

    # Find genes present in expression data
    available = [g for g in gene_set if g in expr_df.columns]
    if len(available) < 3:
        return pd.Series(np.nan, index=expr_df.index)

    # Z-score normalize each gene across samples
    subset = expr_df[available]
    z_scores = (subset - subset.mean()) / (subset.std() + 1e-10)

    # Score = mean z-score across gene set
    return z_scores.mean(axis=1)


def compute_immune_scores(expr):
    """Compute immune cell scores for all samples."""
    print("=" * 60)
    print("Step 1: Computing immune cell infiltration scores...")
    print("=" * 60)

    scores = {}
    for cell_type, gene_set in IMMUNE_GENE_SETS.items():
        score = ssgsea_score(expr, gene_set)
        available = [g for g in gene_set if g in expr.columns]
        scores[cell_type] = score
        print(f"  {cell_type:25s}: {len(available)}/{len(gene_set)} markers found")

    score_df = pd.DataFrame(scores, index=expr.index)
    return score_df


def plot_tumor_vs_normal(score_df, metadata):
    """Compare immune infiltration between tumor and normal."""
    print("\n" + "=" * 60)
    print("Step 2: Comparing immune infiltration: Tumor vs Normal...")
    print("=" * 60)

    meta = metadata.set_index("sample_submitter_id")
    common = score_df.index.intersection(meta.index)
    score_df = score_df.loc[common]
    conditions = meta.loc[common, "condition"]

    # Statistical tests
    results = []
    for cell_type in score_df.columns:
        tumor_scores = score_df.loc[conditions == "Tumor", cell_type].dropna()
        normal_scores = score_df.loc[conditions == "Normal", cell_type].dropna()

        if len(tumor_scores) < 5 or len(normal_scores) < 5:
            continue

        stat, pval = mannwhitneyu(tumor_scores, normal_scores, alternative="two-sided")
        results.append({
            "cell_type": cell_type,
            "tumor_mean": tumor_scores.mean(),
            "normal_mean": normal_scores.mean(),
            "diff": tumor_scores.mean() - normal_scores.mean(),
            "p_value": pval,
        })

    result_df = pd.DataFrame(results).sort_values("p_value")

    print(f"\n  Immune cell comparisons (Tumor vs Normal):")
    for _, row in result_df.iterrows():
        direction = "Higher in Tumor" if row["diff"] > 0 else "Higher in Normal"
        sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else ""
        print(f"    {row['cell_type']:25s}  {direction:20s}  p={row['p_value']:.2e} {sig}")

    # Plot
    plot_data = []
    for cell_type in score_df.columns:
        for sample in common:
            cond = conditions[sample]
            val = score_df.loc[sample, cell_type]
            if pd.notna(val):
                plot_data.append({"Cell Type": cell_type, "Score": val, "Condition": cond})

    plot_df = pd.DataFrame(plot_data)

    fig, ax = plt.subplots(figsize=(16, 7))

    cell_order = result_df["cell_type"].tolist()
    colors = {"Tumor": "#e74c3c", "Normal": "#3498db"}

    sns.boxplot(
        data=plot_df, x="Cell Type", y="Score", hue="Condition",
        order=cell_order, palette=colors, ax=ax,
        fliersize=2, linewidth=0.8, width=0.7,
    )

    # Add significance markers
    for i, ct in enumerate(cell_order):
        row = result_df[result_df["cell_type"] == ct].iloc[0]
        if row["p_value"] < 0.05:
            star = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*"
            y_max = plot_df[plot_df["Cell Type"] == ct]["Score"].max()
            ax.text(i, y_max + 0.1, star, ha="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("")
    ax.set_ylabel("Immune Infiltration Score (ssGSEA)")
    ax.set_title("Immune Cell Infiltration: Tumor vs Normal (TCGA-LUAD)")
    plt.xticks(rotation=45, ha="right")
    ax.legend(title="Condition", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "immune_tumor_vs_normal.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: figures/immune_tumor_vs_normal.png")

    return result_df


def correlate_hub_genes_with_immune(expr, score_df, metadata):
    """Correlate hub gene expression with immune cell scores in tumor samples."""
    print("\n" + "=" * 60)
    print("Step 3: Correlating hub genes with immune infiltration...")
    print("=" * 60)

    # Load hub genes
    hub_path = RESULTS_DIR / "final_hub_genes.csv"
    if hub_path.exists():
        hub_df = pd.read_csv(hub_path)
    else:
        hub_df = pd.read_csv(RESULTS_DIR / "wgcna_hub_genes.csv")

    gene_mapping = None
    if (DATA_DIR / "gene_id_mapping.csv").exists():
        gene_mapping = pd.read_csv(DATA_DIR / "gene_id_mapping.csv", index_col=0)
        name_to_id = dict(zip(gene_mapping["gene_name"], gene_mapping.index))
    else:
        name_to_id = {}

    # Filter to tumor samples only
    meta = metadata.set_index("sample_submitter_id")
    tumor_samples = meta[meta["condition"] == "Tumor"].index
    tumor_samples = [s for s in tumor_samples if s in expr.index and s in score_df.index]

    expr_tumor = expr.loc[tumor_samples]
    score_tumor = score_df.loc[tumor_samples]

    # Select top hub genes (by hub_score or MM)
    sort_col = "hub_score" if "hub_score" in hub_df.columns else "MM"
    top_hubs = hub_df.sort_values(sort_col, ascending=False).head(20)

    # Compute correlations
    cor_matrix = pd.DataFrame(index=[], columns=score_tumor.columns, dtype=float)
    pval_matrix = pd.DataFrame(index=[], columns=score_tumor.columns, dtype=float)

    valid_genes = []
    for _, row in top_hubs.iterrows():
        gene_name = row.get("gene_name", "")
        gene_id = row.get("gene_id", "")

        if gene_id in expr_tumor.columns:
            gene_expr = expr_tumor[gene_id]
        elif gene_name in name_to_id and name_to_id[gene_name] in expr_tumor.columns:
            gene_expr = expr_tumor[name_to_id[gene_name]]
        else:
            continue

        valid_genes.append(gene_name)
        cors = {}
        pvals = {}
        for cell_type in score_tumor.columns:
            valid = gene_expr.notna() & score_tumor[cell_type].notna()
            if valid.sum() > 10:
                r, p = pearsonr(gene_expr[valid], score_tumor.loc[valid, cell_type])
                cors[cell_type] = r
                pvals[cell_type] = p
            else:
                cors[cell_type] = np.nan
                pvals[cell_type] = np.nan

        cor_matrix.loc[gene_name] = cors
        pval_matrix.loc[gene_name] = pvals

    cor_matrix = cor_matrix.astype(float)
    pval_matrix = pval_matrix.astype(float)

    print(f"  Computed correlations for {len(valid_genes)} hub genes × {len(score_tumor.columns)} immune cell types")

    # Plot heatmap
    print("  Plotting correlation heatmap...")
    fig, ax = plt.subplots(figsize=(14, max(6, len(valid_genes) * 0.4)))

    # Annotation with significance
    annot = cor_matrix.copy().astype(str)
    for gene in cor_matrix.index:
        for ct in cor_matrix.columns:
            r = cor_matrix.loc[gene, ct]
            p = pval_matrix.loc[gene, ct]
            if pd.notna(r):
                star = ""
                if p < 0.001: star = "***"
                elif p < 0.01: star = "**"
                elif p < 0.05: star = "*"
                annot.loc[gene, ct] = f"{r:.2f}{star}" if star else ""
            else:
                annot.loc[gene, ct] = ""

    sns.heatmap(
        cor_matrix.astype(float),
        annot=annot, fmt="",
        cmap="RdBu_r", vmin=-0.5, vmax=0.5, center=0,
        xticklabels=True, yticklabels=True,
        linewidths=0.5, ax=ax,
        cbar_kws={"label": "Pearson Correlation", "shrink": 0.7},
        annot_kws={"fontsize": 7},
    )
    ax.set_title("Hub Gene — Immune Cell Infiltration Correlation (Tumor Samples)", fontsize=13)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "immune_hub_correlation_heatmap.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: figures/immune_hub_correlation_heatmap.png")

    # Save
    cor_matrix.to_csv(RESULTS_DIR / "immune_hub_correlation.csv")

    return cor_matrix, pval_matrix


def plot_checkpoint_analysis(expr, metadata):
    """Analyze immune checkpoint gene expression."""
    print("\n  Plotting immune checkpoint analysis...")

    checkpoint_genes = ["PDCD1", "CD274", "CTLA4", "LAG3", "HAVCR2", "TIGIT", "IDO1"]
    checkpoint_names = ["PD-1", "PD-L1", "CTLA-4", "LAG-3", "TIM-3", "TIGIT", "IDO1"]

    gene_mapping = None
    if (DATA_DIR / "gene_id_mapping.csv").exists():
        gene_mapping = pd.read_csv(DATA_DIR / "gene_id_mapping.csv", index_col=0)
        name_to_id = dict(zip(gene_mapping["gene_name"], gene_mapping.index))
    else:
        name_to_id = {}

    meta = metadata.set_index("sample_submitter_id")
    common = expr.index.intersection(meta.index)

    plot_data = []
    for gene_symbol, display_name in zip(checkpoint_genes, checkpoint_names):
        gene_id = name_to_id.get(gene_symbol, gene_symbol)
        if gene_id not in expr.columns:
            continue
        for sample in common:
            plot_data.append({
                "Gene": display_name,
                "Expression": expr.loc[sample, gene_id],
                "Condition": meta.loc[sample, "condition"],
            })

    if not plot_data:
        print("    Skipped: checkpoint genes not found")
        return

    plot_df = pd.DataFrame(plot_data)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {"Tumor": "#e74c3c", "Normal": "#3498db"}

    sns.boxplot(data=plot_df, x="Gene", y="Expression", hue="Condition",
                palette=colors, ax=ax, fliersize=2, width=0.6)

    ax.set_xlabel("")
    ax.set_ylabel("Expression (log2 CPM)")
    ax.set_title("Immune Checkpoint Gene Expression: Tumor vs Normal")
    ax.legend(title="Condition")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "immune_checkpoint_expression.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: figures/immune_checkpoint_expression.png")


def main():
    print("\n" + "=" * 60)
    print("  Phase 6C: Immune Infiltration Analysis")
    print("=" * 60 + "\n")

    # Load data
    print("Loading data...")
    counts = pd.read_csv(DATA_DIR / "filtered_counts.csv", index_col=0)
    metadata = pd.read_csv(DATA_DIR / "metadata.csv")

    gene_mapping = None
    if (DATA_DIR / "gene_id_mapping.csv").exists():
        gene_mapping = pd.read_csv(DATA_DIR / "gene_id_mapping.csv", index_col=0)

    # Normalize
    lib_sizes = counts.sum(axis=0)
    cpm = counts.div(lib_sizes, axis=1) * 1e6
    log_cpm = np.log2(cpm + 1)

    # Map Ensembl IDs to gene names for immune gene matching
    if gene_mapping is not None:
        id_to_name = gene_mapping["gene_name"].to_dict()
        log_cpm_named = log_cpm.copy()
        log_cpm_named.index = [id_to_name.get(g, g) for g in log_cpm_named.index]
        # Remove duplicate gene names
        log_cpm_named = log_cpm_named[~log_cpm_named.index.duplicated(keep="first")]
        expr = log_cpm_named.T  # samples × genes (by name)
    else:
        expr = log_cpm.T

    # Compute immune scores
    score_df = compute_immune_scores(expr)

    # Save scores
    score_df.to_csv(RESULTS_DIR / "immune_scores.csv")

    # Tumor vs Normal comparison
    plot_tumor_vs_normal(score_df, metadata)

    # Hub gene-immune correlation (use Ensembl-indexed expr for gene_id matching)
    expr_ensembl = log_cpm.T
    correlate_hub_genes_with_immune(expr_ensembl, score_df, metadata)

    # Checkpoint analysis
    plot_checkpoint_analysis(expr, metadata)

    print(f"\n  Phase 6C complete!")


if __name__ == "__main__":
    main()
