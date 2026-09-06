"""
03_differential_expression.py
==============================
Differential expression analysis using PyDESeq2.
Compares Tumor vs Normal in TCGA-LUAD.

Usage:
    python scripts/03_differential_expression.py

Input:
    data/filtered_counts.csv
    data/metadata.csv
    data/gene_id_mapping.csv

Output:
    results/deg_all_results.csv         - Full DESeq2 results for all genes
    results/deg_significant.csv         - Significant DEGs only
    results/deg_summary.txt             - Summary statistics
    figures/volcano_plot.png
    figures/heatmap_top_degs.png
    figures/deg_counts_barplot.png
    figures/ma_plot.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
FIG_DIR = PROJECT_DIR / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

# DEG thresholds
LOG2FC_THRESHOLD = 1.0      # |log2FoldChange| > 1
PADJ_THRESHOLD = 0.05       # adjusted p-value < 0.05

# Figure settings
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 500,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.figsize": (8, 6),
})


# ──────────────────────────────────────────────
# Step 1: Load and prepare data for PyDESeq2
# ──────────────────────────────────────────────
def load_and_prepare():
    print("=" * 60)
    print("Step 1: Loading data for PyDESeq2...")
    print("=" * 60)

    # Load filtered counts (genes × samples)
    counts = pd.read_csv(DATA_DIR / "filtered_counts.csv", index_col=0)
    metadata = pd.read_csv(DATA_DIR / "metadata.csv")

    # Load gene mapping
    gene_mapping = None
    if (DATA_DIR / "gene_id_mapping.csv").exists():
        gene_mapping = pd.read_csv(DATA_DIR / "gene_id_mapping.csv", index_col=0)

    # Ensure metadata is indexed by sample_submitter_id
    metadata = metadata.set_index("sample_submitter_id")

    # Align: keep only samples present in both counts and metadata
    common = counts.columns.intersection(metadata.index)
    counts = counts[common]
    metadata = metadata.loc[common]

    # PyDESeq2 expects samples × genes (transposed)
    counts_t = counts.T.astype(int)

    # Ensure condition is a categorical with "Normal" as reference
    metadata["condition"] = pd.Categorical(
        metadata["condition"], categories=["Normal", "Tumor"]
    )

    print(f"  Count matrix: {counts_t.shape[0]} samples × {counts_t.shape[1]} genes")
    print(f"  Conditions: {metadata['condition'].value_counts().to_dict()}")

    return counts_t, metadata, gene_mapping


# ──────────────────────────────────────────────
# Step 2: Run PyDESeq2
# ──────────────────────────────────────────────
def run_deseq2(counts_t, metadata):
    print("\n" + "=" * 60)
    print("Step 2: Running PyDESeq2 (this may take a few minutes)...")
    print("=" * 60)

    # Create DESeq2 dataset
    dds = DeseqDataSet(
        counts=counts_t,
        metadata=metadata[["condition"]],
        design="~condition",
    )

    # Run DESeq2 pipeline (size factors, dispersion, fitting)
    print("  Estimating size factors...")
    dds.fit_size_factors()

    print("  Estimating dispersions...")
    dds.fit_genewise_dispersions()

    print("  Fitting dispersion trend...")
    dds.fit_dispersion_trend()

    print("  Fitting dispersion prior...")
    dds.fit_dispersion_prior()

    print("  Shrinking dispersions...")
    dds.fit_MAP_dispersions()

    print("  Fitting LFC...")
    dds.fit_LFC()

    print("  Calculating Cook's distances...")
    dds.calculate_cooks()

    # Refit outliers flagged by Cook's distance
    print("  Refitting Cook's outliers...")
    if dds.refit_cooks:
        dds.refit()

    # Statistical testing
    print("  Running Wald test...")
    stat_res = DeseqStats(dds, contrast=["condition", "Tumor", "Normal"])
    stat_res.summary()

    # LFC shrinkage for better visualization
    print("  Applying LFC shrinkage (apeglm)...")
    try:
        stat_res.lfc_shrink(coeff="condition_Tumor_vs_Normal")
        print("  LFC shrinkage applied successfully")
    except Exception as e:
        print(f"  LFC shrinkage skipped: {e}")

    # Extract results
    results_df = stat_res.results_df.copy()

    print(f"\n  Results: {len(results_df)} genes tested")
    print(f"  Non-NA padj: {results_df['padj'].notna().sum()}")

    return results_df, dds


# ──────────────────────────────────────────────
# Step 3: Annotate and filter results
# ──────────────────────────────────────────────
def annotate_results(results_df, gene_mapping):
    print("\n" + "=" * 60)
    print("Step 3: Annotating results...")
    print("=" * 60)

    # Add gene names if mapping available
    if gene_mapping is not None:
        results_df = results_df.join(gene_mapping, how="left")
        n_mapped = results_df["gene_name"].notna().sum()
        print(f"  Mapped {n_mapped}/{len(results_df)} genes to gene names")
    else:
        results_df["gene_name"] = results_df.index
        results_df["gene_type"] = "unknown"

    # Classification
    results_df["regulation"] = "Not Significant"
    sig_up = (results_df["padj"] < PADJ_THRESHOLD) & (results_df["log2FoldChange"] > LOG2FC_THRESHOLD)
    sig_down = (results_df["padj"] < PADJ_THRESHOLD) & (results_df["log2FoldChange"] < -LOG2FC_THRESHOLD)
    results_df.loc[sig_up, "regulation"] = "Up"
    results_df.loc[sig_down, "regulation"] = "Down"

    # Sort by padj
    results_df = results_df.sort_values("padj")

    # Summary
    n_up = (results_df["regulation"] == "Up").sum()
    n_down = (results_df["regulation"] == "Down").sum()
    n_sig = n_up + n_down

    print(f"\n  DEG Summary (|log2FC| > {LOG2FC_THRESHOLD}, padj < {PADJ_THRESHOLD}):")
    print(f"    Upregulated:     {n_up:,}")
    print(f"    Downregulated:   {n_down:,}")
    print(f"    Total DEGs:      {n_sig:,}")
    print(f"    Not significant: {len(results_df) - n_sig:,}")

    # Show top 10 up and down
    print(f"\n  Top 10 upregulated genes:")
    top_up = results_df[results_df["regulation"] == "Up"].head(10)
    for _, row in top_up.iterrows():
        name = row.get("gene_name", row.name)
        print(f"    {name:15s}  log2FC={row['log2FoldChange']:+.2f}  padj={row['padj']:.2e}")

    print(f"\n  Top 10 downregulated genes:")
    top_down = results_df[results_df["regulation"] == "Down"].sort_values("log2FoldChange").head(10)
    for _, row in top_down.iterrows():
        name = row.get("gene_name", row.name)
        print(f"    {name:15s}  log2FC={row['log2FoldChange']:+.2f}  padj={row['padj']:.2e}")

    return results_df


# ──────────────────────────────────────────────
# Step 4: Volcano plot
# ──────────────────────────────────────────────
def plot_volcano(results_df):
    print("\n" + "=" * 60)
    print("Step 4: Generating volcano plot...")
    print("=" * 60)

    df = results_df.dropna(subset=["padj", "log2FoldChange"]).copy()
    df["-log10padj"] = -np.log10(df["padj"].clip(lower=1e-300))

    fig, ax = plt.subplots(figsize=(10, 8))

    # Color mapping
    colors = {"Not Significant": "#cccccc", "Up": "#e74c3c", "Down": "#3498db"}

    for reg, color in colors.items():
        mask = df["regulation"] == reg
        ax.scatter(
            df.loc[mask, "log2FoldChange"],
            df.loc[mask, "-log10padj"],
            c=color, s=8, alpha=0.5, label=reg, edgecolors="none",
        )

    # Threshold lines
    ax.axhline(-np.log10(PADJ_THRESHOLD), color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axvline(LOG2FC_THRESHOLD, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axvline(-LOG2FC_THRESHOLD, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)

    # Label top genes
    n_label = 15
    top_genes = pd.concat([
        df[df["regulation"] == "Up"].nlargest(n_label, "-log10padj"),
        df[df["regulation"] == "Down"].nlargest(n_label, "-log10padj"),
    ])

    try:
        from adjustText import adjust_text
        texts = []
        for idx, row in top_genes.iterrows():
            name = row.get("gene_name", idx)
            if pd.isna(name) or name == "":
                name = idx
            t = ax.text(
                row["log2FoldChange"], row["-log10padj"],
                str(name), fontsize=7, ha="center", va="bottom",
            )
            texts.append(t)
        adjust_text(texts, arrowprops=dict(arrowstyle="-", color="grey", lw=0.5), ax=ax)
    except ImportError:
        for idx, row in top_genes.iterrows():
            name = row.get("gene_name", idx)
            if pd.isna(name) or name == "":
                name = idx
            ax.annotate(str(name), (row["log2FoldChange"], row["-log10padj"]),
                       fontsize=6, ha="center", va="bottom")

    # Counts in legend — use handles from scatter calls to keep colors correct
    n_up = (df["regulation"] == "Up").sum()
    n_down = (df["regulation"] == "Down").sum()
    n_ns = (df["regulation"] == "Not Significant").sum()

    handles, _ = ax.get_legend_handles_labels()
    # scatter order was: Not Significant, Up, Down
    ax.legend(
        handles=handles,
        labels=[f"NS ({n_ns:,})", f"Up ({n_up:,})", f"Down ({n_down:,})"],
        title="Regulation",
        loc="upper right", fontsize=9,
    )

    ax.set_xlabel("log2 Fold Change (Tumor vs Normal)")
    ax.set_ylabel("-log10(adjusted p-value)")
    ax.set_title("Differential Gene Expression: Tumor vs Normal (TCGA-LUAD)")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "volcano_plot.png")
    plt.close()
    print(f"  Saved: figures/volcano_plot.png")


# ──────────────────────────────────────────────
# Step 5: MA plot
# ──────────────────────────────────────────────
def plot_ma(results_df):
    print("  Generating MA plot...")

    df = results_df.dropna(subset=["padj", "log2FoldChange", "baseMean"]).copy()
    df["log10_baseMean"] = np.log10(df["baseMean"] + 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"Not Significant": "#cccccc", "Up": "#e74c3c", "Down": "#3498db"}
    for reg in ["Not Significant", "Up", "Down"]:
        mask = df["regulation"] == reg
        ax.scatter(
            df.loc[mask, "log10_baseMean"],
            df.loc[mask, "log2FoldChange"],
            c=colors[reg], s=6, alpha=0.4, label=reg, edgecolors="none",
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(LOG2FC_THRESHOLD, color="grey", linestyle="--", linewidth=0.7)
    ax.axhline(-LOG2FC_THRESHOLD, color="grey", linestyle="--", linewidth=0.7)

    ax.set_xlabel("log10(Mean Expression)")
    ax.set_ylabel("log2 Fold Change")
    ax.set_title("MA Plot: Tumor vs Normal")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "ma_plot.png")
    plt.close()
    print(f"  Saved: figures/ma_plot.png")


# ──────────────────────────────────────────────
# Step 6: Heatmap of top DEGs
# ──────────────────────────────────────────────
def plot_heatmap(results_df, counts_t, metadata):
    print("  Generating heatmap of top DEGs...")

    # Select top 25 up + top 25 down by padj
    sig_df = results_df[results_df["regulation"] != "Not Significant"].copy()
    top_up = sig_df[sig_df["regulation"] == "Up"].head(25)
    top_down = sig_df[sig_df["regulation"] == "Down"].sort_values("log2FoldChange").head(25)
    top_genes = pd.concat([top_up, top_down])

    gene_ids = top_genes.index.tolist()

    # Get normalized expression (log2 CPM)
    counts_raw = counts_t[gene_ids].T  # genes × samples
    lib_sizes = counts_t.sum(axis=1)
    cpm = counts_raw.div(lib_sizes, axis=1) * 1e6
    log_cpm = np.log2(cpm + 1)

    # Z-score per gene
    z_scores = log_cpm.subtract(log_cpm.mean(axis=1), axis=0).div(log_cpm.std(axis=1), axis=0)

    # Sort samples by condition
    sample_order = metadata.sort_values("condition").index
    sample_order = [s for s in sample_order if s in z_scores.columns]
    z_scores = z_scores[sample_order]

    # Use gene names if available
    if "gene_name" in top_genes.columns:
        name_map = top_genes["gene_name"].to_dict()
        z_scores.index = [name_map.get(g, g) for g in z_scores.index]

    # Condition color bar
    conditions = metadata.loc[sample_order, "condition"]
    col_colors = conditions.map({"Tumor": "#e74c3c", "Normal": "#3498db"}).values

    # Clamp z-scores for visualization
    z_scores = z_scores.clip(-3, 3)

    # Plot
    fig_height = max(8, len(z_scores) * 0.3)
    g = sns.clustermap(
        z_scores,
        cmap="RdBu_r",
        vmin=-3, vmax=3,
        col_cluster=False,
        row_cluster=True,
        col_colors=col_colors,
        figsize=(14, fig_height),
        xticklabels=False,
        yticklabels=True,
        linewidths=0,
        cbar_pos=(0.02, 0.82, 0.03, 0.12),
        cbar_kws={"label": "Z-score"},
        dendrogram_ratio=(0.08, 0.02),
    )

    g.ax_heatmap.set_xlabel("Samples")
    g.ax_heatmap.set_ylabel("")
    g.fig.suptitle("Top 50 Differentially Expressed Genes (Tumor vs Normal)", y=1.01, fontsize=14)

    # Add legend for condition colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e74c3c", label="Tumor"),
        Patch(facecolor="#3498db", label="Normal"),
    ]
    g.ax_heatmap.legend(handles=legend_elements, loc="upper left",
                         bbox_to_anchor=(1.05, 1.0), title="Condition", fontsize=9)

    plt.savefig(FIG_DIR / "heatmap_top_degs.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: figures/heatmap_top_degs.png")


# ──────────────────────────────────────────────
# Step 7: DEG counts barplot
# ──────────────────────────────────────────────
def plot_deg_barplot(results_df):
    print("  Generating DEG summary barplot...")

    n_up = (results_df["regulation"] == "Up").sum()
    n_down = (results_df["regulation"] == "Down").sum()

    fig, ax = plt.subplots(figsize=(6, 5))

    bars = ax.bar(
        ["Upregulated", "Downregulated"],
        [n_up, n_down],
        color=["#e74c3c", "#3498db"],
        width=0.5,
        edgecolor="white",
    )

    for bar, val in zip(bars, [n_up, n_down]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f"{val:,}", ha="center", fontsize=12, fontweight="bold")

    ax.set_ylabel("Number of DEGs")
    ax.set_title(f"Differentially Expressed Genes\n(|log2FC| > {LOG2FC_THRESHOLD}, padj < {PADJ_THRESHOLD})")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "deg_counts_barplot.png")
    plt.close()
    print(f"  Saved: figures/deg_counts_barplot.png")


# ──────────────────────────────────────────────
# Step 8: Save results
# ──────────────────────────────────────────────
def save_results(results_df):
    print("\n" + "=" * 60)
    print("Step 5: Saving results...")
    print("=" * 60)

    # All results
    results_df.to_csv(RESULTS_DIR / "deg_all_results.csv")
    print(f"  Saved: results/deg_all_results.csv ({len(results_df):,} genes)")

    # Significant DEGs only
    sig_df = results_df[results_df["regulation"] != "Not Significant"].copy()
    sig_df.to_csv(RESULTS_DIR / "deg_significant.csv")
    print(f"  Saved: results/deg_significant.csv ({len(sig_df):,} DEGs)")

    # Up and down separately
    up_df = results_df[results_df["regulation"] == "Up"]
    down_df = results_df[results_df["regulation"] == "Down"]
    up_df.to_csv(RESULTS_DIR / "deg_upregulated.csv")
    down_df.to_csv(RESULTS_DIR / "deg_downregulated.csv")

    # Summary text
    summary = f"""TCGA-LUAD Differential Expression Analysis Summary
====================================================
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
Comparison: Tumor vs Normal
Method: PyDESeq2 (Wald test)

Thresholds:
  |log2FoldChange| > {LOG2FC_THRESHOLD}
  Adjusted p-value  < {PADJ_THRESHOLD}

Results:
  Total genes tested:  {len(results_df):,}
  Upregulated (Tumor): {len(up_df):,}
  Downregulated:       {len(down_df):,}
  Total DEGs:          {len(sig_df):,}
  Not significant:     {len(results_df) - len(sig_df):,}

Top 20 Upregulated Genes:
{up_df.head(20)[['gene_name', 'log2FoldChange', 'padj']].to_string()}

Top 20 Downregulated Genes:
{down_df.sort_values('log2FoldChange').head(20)[['gene_name', 'log2FoldChange', 'padj']].to_string()}
"""

    with open(RESULTS_DIR / "deg_summary.txt", "w") as f:
        f.write(summary)
    print(f"  Saved: results/deg_summary.txt")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("  TCGA-LUAD Differential Expression Analysis (PyDESeq2)")
    print("=" * 60 + "\n")

    # Load data
    counts_t, metadata, gene_mapping = load_and_prepare()

    # Run DESeq2
    results_df, dds = run_deseq2(counts_t, metadata)

    # Annotate
    results_df = annotate_results(results_df, gene_mapping)

    # Plots
    print("\n" + "=" * 60)
    print("Step 4: Generating figures...")
    print("=" * 60)

    plot_volcano(results_df)
    plot_ma(results_df)
    plot_heatmap(results_df, counts_t, metadata)
    plot_deg_barplot(results_df)

    # Save
    save_results(results_df)

    print("\n" + "=" * 60)
    print("  Phase 3 complete!")
    print("  Next: python scripts/04_enrichment_analysis.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
