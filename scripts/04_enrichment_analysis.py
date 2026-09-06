"""
04_enrichment_analysis.py
==========================
Functional enrichment analysis of DEGs from TCGA-LUAD.
  - Gene Ontology (BP, MF, CC) overrepresentation analysis
  - KEGG pathway enrichment
  - GSEA with MSigDB Hallmark gene sets

Usage:
    python scripts/04_enrichment_analysis.py

Input:
    results/deg_all_results.csv
    results/deg_significant.csv

Output:
    results/go_bp_up.csv, go_bp_down.csv
    results/go_mf_up.csv, go_mf_down.csv
    results/go_cc_up.csv, go_cc_down.csv
    results/kegg_up.csv, kegg_down.csv
    results/gsea_hallmark.csv
    figures/go_bp_dotplot.png
    figures/go_combined_barplot.png
    figures/kegg_dotplot.png
    figures/gsea_hallmark_dotplot.png
    figures/gsea_top_enrichment_plots.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path
import gseapy as gp
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

# Enrichr library names
GO_BP = "GO_Biological_Process_2023"
GO_MF = "GO_Molecular_Function_2023"
GO_CC = "GO_Cellular_Component_2023"
KEGG = "KEGG_2021_Human"
HALLMARK = "MSigDB_Hallmark_2020"

# Enrichment thresholds
PADJ_CUTOFF = 0.05
TOP_N = 15  # Number of terms to display in plots

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
})


# ──────────────────────────────────────────────
# Step 1: Load DEG results
# ──────────────────────────────────────────────
def load_deg_results():
    print("=" * 60)
    print("Step 1: Loading DEG results...")
    print("=" * 60)

    all_results = pd.read_csv(RESULTS_DIR / "deg_all_results.csv", index_col=0)
    sig_results = pd.read_csv(RESULTS_DIR / "deg_significant.csv", index_col=0)

    # Separate up and down gene lists (gene names)
    up_genes = sig_results[sig_results["regulation"] == "Up"]["gene_name"].dropna().tolist()
    down_genes = sig_results[sig_results["regulation"] == "Down"]["gene_name"].dropna().tolist()

    # For GSEA: build ranked gene list (all genes, ranked by signed significance)
    ranked = all_results.dropna(subset=["pvalue", "log2FoldChange"]).copy()
    ranked["rank_metric"] = -np.log10(ranked["pvalue"].clip(lower=1e-300)) * np.sign(ranked["log2FoldChange"])
    ranked = ranked.dropna(subset=["gene_name"])
    ranked = ranked[ranked["gene_name"] != ""]
    # Remove duplicates, keep most significant
    ranked = ranked.sort_values("pvalue").drop_duplicates(subset="gene_name", keep="first")
    ranked_series = ranked.set_index("gene_name")["rank_metric"].sort_values(ascending=False)

    print(f"  Upregulated genes:   {len(up_genes):,}")
    print(f"  Downregulated genes: {len(down_genes):,}")
    print(f"  Ranked gene list:    {len(ranked_series):,} genes")

    return up_genes, down_genes, ranked_series


# ──────────────────────────────────────────────
# Step 2: Run Enrichr ORA (GO + KEGG)
# ──────────────────────────────────────────────
def run_enrichr(gene_list, gene_sets, label="", description=""):
    """Run Enrichr overrepresentation analysis."""
    try:
        enr = gp.enrichr(
            gene_list=gene_list,
            gene_sets=gene_sets,
            organism="human",
            outdir=None,  # Don't save to disk automatically
            no_plot=True,
            cutoff=0.5,   # Relaxed cutoff for initial query; filter later
        )
        results = enr.results
        results = results[results["Adjusted P-value"] < PADJ_CUTOFF]
        print(f"    {label}: {len(results)} significant terms (padj < {PADJ_CUTOFF})")
        return results
    except Exception as e:
        print(f"    WARNING: {label} failed: {e}")
        return pd.DataFrame()


def run_all_enrichr(up_genes, down_genes):
    print("\n" + "=" * 60)
    print("Step 2: Running GO and KEGG enrichment (Enrichr)...")
    print("=" * 60)

    enrichment_results = {}

    libraries = {
        "GO_BP": GO_BP,
        "GO_MF": GO_MF,
        "GO_CC": GO_CC,
        "KEGG": KEGG,
    }

    for lib_key, lib_name in libraries.items():
        print(f"\n  {lib_key}:")

        # Upregulated
        up_res = run_enrichr(up_genes, lib_name, label=f"{lib_key}_up")
        if not up_res.empty:
            up_res["direction"] = "Up"
            up_res.to_csv(RESULTS_DIR / f"{lib_key.lower()}_up.csv", index=False)
            enrichment_results[f"{lib_key}_up"] = up_res

        # Downregulated
        down_res = run_enrichr(down_genes, lib_name, label=f"{lib_key}_down")
        if not down_res.empty:
            down_res["direction"] = "Down"
            down_res.to_csv(RESULTS_DIR / f"{lib_key.lower()}_down.csv", index=False)
            enrichment_results[f"{lib_key}_down"] = down_res

    return enrichment_results


# ──────────────────────────────────────────────
# Step 3: Run GSEA with Hallmark gene sets
# ──────────────────────────────────────────────
def run_gsea_hallmark(ranked_series):
    print("\n" + "=" * 60)
    print("Step 3: Running GSEA with MSigDB Hallmark gene sets...")
    print("=" * 60)

    try:
        pre_res = gp.prerank(
            rnk=ranked_series,
            gene_sets=HALLMARK,
            min_size=15,
            max_size=500,
            permutation_num=1000,
            outdir=None,
            seed=42,
            no_plot=True,
        )

        gsea_results = pre_res.res2d
        gsea_results = gsea_results.sort_values("NES", ascending=False)

        # Classify
        gsea_results["direction"] = gsea_results["NES"].apply(
            lambda x: "Activated" if x > 0 else "Suppressed"
        )

        sig = gsea_results[gsea_results["FDR q-val"].astype(float) < PADJ_CUTOFF]
        print(f"  Total Hallmark pathways tested: {len(gsea_results)}")
        print(f"  Significant (FDR < {PADJ_CUTOFF}): {len(sig)}")
        print(f"    Activated: {(sig['direction'] == 'Activated').sum()}")
        print(f"    Suppressed: {(sig['direction'] == 'Suppressed').sum()}")

        # Save
        gsea_results.to_csv(RESULTS_DIR / "gsea_hallmark.csv", index=False)

        # Print top results
        print(f"\n  Top activated pathways:")
        for _, row in sig[sig["direction"] == "Activated"].head(5).iterrows():
            name = row["Term"].replace("HALLMARK_", "").replace("_", " ")
            print(f"    {name:40s}  NES={float(row['NES']):+.2f}  FDR={float(row['FDR q-val']):.2e}")

        print(f"\n  Top suppressed pathways:")
        for _, row in sig[sig["direction"] == "Suppressed"].tail(5).iloc[::-1].iterrows():
            name = row["Term"].replace("HALLMARK_", "").replace("_", " ")
            print(f"    {name:40s}  NES={float(row['NES']):+.2f}  FDR={float(row['FDR q-val']):.2e}")

        return gsea_results, pre_res

    except Exception as e:
        print(f"  ERROR: GSEA failed: {e}")
        return pd.DataFrame(), None


# ──────────────────────────────────────────────
# Step 4: Plotting functions
# ──────────────────────────────────────────────
def clean_term(term, max_len=50):
    """Clean and truncate enrichment term names for display."""
    # Remove GO ID suffix like (GO:0006955)
    if "(" in term:
        term = term.rsplit("(", 1)[0].strip()
    # Remove common prefixes
    term = term.replace("HALLMARK_", "").replace("_", " ")
    # Truncate
    if len(term) > max_len:
        term = term[:max_len-3] + "..."
    return term


def plot_enrichr_dotplot(results_df, title, filename, top_n=TOP_N):
    """Dotplot for Enrichr results (single direction)."""
    if results_df.empty:
        return

    df = results_df.head(top_n).copy()
    df["Term_clean"] = df["Term"].apply(clean_term)
    df["-log10(padj)"] = -np.log10(df["Adjusted P-value"].clip(lower=1e-300))

    # Gene ratio: Overlap / gene set size
    if "Overlap" in df.columns:
        df["overlap_count"] = df["Overlap"].apply(lambda x: int(str(x).split("/")[0]))
        df["set_size"] = df["Overlap"].apply(lambda x: int(str(x).split("/")[1]))
        df["Gene Ratio"] = df["overlap_count"] / df["set_size"]
    else:
        df["overlap_count"] = df.get("Genes", "").apply(lambda x: len(str(x).split(";")))
        df["Gene Ratio"] = 0.1

    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.4)))

    scatter = ax.scatter(
        df["Gene Ratio"],
        range(len(df)),
        s=df["overlap_count"] * 8,
        c=df["-log10(padj)"],
        cmap="RdYlBu_r",
        edgecolors="black",
        linewidth=0.5,
        alpha=0.8,
    )

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["Term_clean"].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Gene Ratio")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("-log10(adj. p-value)", fontsize=9)

    # Size legend
    sizes = [10, 30, 60]
    for s in sizes:
        ax.scatter([], [], s=s*8, c="grey", alpha=0.5, edgecolors="black",
                   linewidth=0.5, label=f"{s} genes")
    ax.legend(title="Overlap", loc="lower right", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, bbox_inches="tight")
    plt.close()
    print(f"  Saved: figures/{filename}")


def plot_go_combined_barplot(enrichment_results):
    """Combined barplot showing top GO BP terms for up and down genes."""
    print("  Generating GO BP combined barplot...")

    up_df = enrichment_results.get("GO_BP_up", pd.DataFrame())
    down_df = enrichment_results.get("GO_BP_down", pd.DataFrame())

    if up_df.empty and down_df.empty:
        print("    Skipped: no GO BP results")
        return

    n = 10

    fig, ax = plt.subplots(figsize=(12, 8))

    plot_data = []

    if not up_df.empty:
        top_up = up_df.head(n).copy()
        top_up["Term_clean"] = top_up["Term"].apply(clean_term)
        top_up["-log10(padj)"] = -np.log10(top_up["Adjusted P-value"].clip(lower=1e-300))
        for _, row in top_up.iterrows():
            plot_data.append({
                "term": row["Term_clean"],
                "value": row["-log10(padj)"],
                "direction": "Upregulated in Tumor"
            })

    if not down_df.empty:
        top_down = down_df.head(n).copy()
        top_down["Term_clean"] = top_down["Term"].apply(clean_term)
        top_down["-log10(padj)"] = -np.log10(top_down["Adjusted P-value"].clip(lower=1e-300))
        for _, row in top_down.iterrows():
            plot_data.append({
                "term": row["Term_clean"],
                "value": -row["-log10(padj)"],  # Negative for left side
                "direction": "Downregulated in Tumor"
            })

    plot_df = pd.DataFrame(plot_data)
    plot_df = plot_df.sort_values("value")

    colors = ["#3498db" if v < 0 else "#e74c3c" for v in plot_df["value"]]

    ax.barh(range(len(plot_df)), plot_df["value"], color=colors, edgecolor="white", height=0.7)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["term"], fontsize=9)
    ax.set_xlabel("-log10(adj. p-value)")
    ax.set_title("GO Biological Process Enrichment\n(Tumor vs Normal)")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)

    # Add direction labels
    xlim = ax.get_xlim()
    ax.text(xlim[1] * 0.7, len(plot_df) + 0.5, "Upregulated\nin Tumor",
            ha="center", fontsize=10, color="#e74c3c", fontweight="bold")
    ax.text(xlim[0] * 0.7, len(plot_df) + 0.5, "Downregulated\nin Tumor",
            ha="center", fontsize=10, color="#3498db", fontweight="bold")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "go_combined_barplot.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: figures/go_combined_barplot.png")


def plot_gsea_hallmark(gsea_results):
    """Dotplot for GSEA Hallmark results."""
    print("  Generating GSEA Hallmark dotplot...")

    if gsea_results.empty:
        print("    Skipped: no GSEA results")
        return

    df = gsea_results.copy()
    df["FDR"] = df["FDR q-val"].astype(float)
    df["NES"] = df["NES"].astype(float)

    # Show all significant + top non-significant
    sig = df[df["FDR"] < PADJ_CUTOFF].copy()
    if len(sig) < 5:
        sig = df.head(20).copy()

    sig["Term_clean"] = sig["Term"].apply(
        lambda x: x.replace("HALLMARK_", "").replace("_", " ").title()
    )
    sig = sig.sort_values("NES")
    sig["-log10(FDR)"] = -np.log10(sig["FDR"].clip(lower=1e-300))

    fig, ax = plt.subplots(figsize=(10, max(6, len(sig) * 0.35)))

    colors = ["#e74c3c" if nes > 0 else "#3498db" for nes in sig["NES"]]

    bars = ax.barh(
        range(len(sig)), sig["NES"],
        color=colors, edgecolor="white", height=0.7, alpha=0.85,
    )

    ax.set_yticks(range(len(sig)))
    ax.set_yticklabels(sig["Term_clean"], fontsize=9)
    ax.set_xlabel("Normalized Enrichment Score (NES)")
    ax.set_title("GSEA: MSigDB Hallmark Pathways (TCGA-LUAD)")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)

    # Add FDR annotation
    for i, (_, row) in enumerate(sig.iterrows()):
        fdr = row["FDR"]
        star = "***" if fdr < 0.001 else "**" if fdr < 0.01 else "*" if fdr < 0.05 else ""
        if star:
            x_pos = row["NES"] + (0.05 if row["NES"] > 0 else -0.05)
            ax.text(x_pos, i, star, va="center",
                    ha="left" if row["NES"] > 0 else "right",
                    fontsize=10, fontweight="bold", color="black")

    # Legend
    ax.text(0.98, 0.02, "* FDR<0.05  ** FDR<0.01  *** FDR<0.001",
            transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
            color="grey")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "gsea_hallmark_dotplot.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: figures/gsea_hallmark_dotplot.png")


def plot_gsea_enrichment_curves(pre_res):
    """Plot GSEA enrichment curves for top activated and suppressed pathways."""
    print("  Generating GSEA enrichment score curves...")

    if pre_res is None:
        print("    Skipped: no GSEA results")
        return

    df = pre_res.res2d.copy()
    df["NES"] = df["NES"].astype(float)
    df["FDR q-val"] = df["FDR q-val"].astype(float)
    sig = df[df["FDR q-val"] < PADJ_CUTOFF]

    # Top 3 activated + top 3 suppressed
    top_activated = sig[sig["NES"] > 0].nlargest(3, "NES")["Term"].tolist()
    top_suppressed = sig[sig["NES"] < 0].nsmallest(3, "NES")["Term"].tolist()
    terms_to_plot = top_activated + top_suppressed

    if len(terms_to_plot) == 0:
        print("    Skipped: no significant pathways for curves")
        return

    n_plots = len(terms_to_plot)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, term in enumerate(terms_to_plot):
        ax = axes[i]

        # Get enrichment data from gseapy results
        term_data = df[df["Term"] == term].iloc[0]
        nes = float(term_data["NES"])
        fdr = float(term_data["FDR q-val"])

        # Get the running enrichment score
        try:
            # Access the ranking and enrichment score from the prerank object
            gsea_term = pre_res.results[term]
            es_profile = gsea_term["RES"]

            color = "#e74c3c" if nes > 0 else "#3498db"
            ax.plot(es_profile, color=color, linewidth=1.5)
            ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
            ax.fill_between(range(len(es_profile)), es_profile, alpha=0.15, color=color)

            title = term.replace("HALLMARK_", "").replace("_", " ").title()
            ax.set_title(f"{title}\nNES={nes:+.2f} FDR={fdr:.1e}", fontsize=9)
            ax.set_xlabel("Rank", fontsize=8)
            ax.set_ylabel("ES", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.spines[["top", "right"]].set_visible(False)
        except Exception:
            # Fallback: just show NES bar
            color = "#e74c3c" if nes > 0 else "#3498db"
            ax.barh([0], [nes], color=color)
            title = term.replace("HALLMARK_", "").replace("_", " ").title()
            ax.set_title(f"{title}\nNES={nes:+.2f}", fontsize=9)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("GSEA Enrichment Score Profiles", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "gsea_top_enrichment_plots.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: figures/gsea_top_enrichment_plots.png")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("  TCGA-LUAD Functional Enrichment Analysis")
    print("=" * 60 + "\n")

    # Load DEGs
    up_genes, down_genes, ranked_series = load_deg_results()

    # Run Enrichr (GO + KEGG)
    enrichment_results = run_all_enrichr(up_genes, down_genes)

    # Run GSEA Hallmark
    gsea_results, pre_res = run_gsea_hallmark(ranked_series)

    # ── Generate figures ──
    print("\n" + "=" * 60)
    print("Step 4: Generating figures...")
    print("=" * 60)

    # GO BP dotplots (up and down separately)
    if "GO_BP_up" in enrichment_results:
        plot_enrichr_dotplot(
            enrichment_results["GO_BP_up"],
            "GO Biological Process — Upregulated in Tumor",
            "go_bp_up_dotplot.png"
        )
    if "GO_BP_down" in enrichment_results:
        plot_enrichr_dotplot(
            enrichment_results["GO_BP_down"],
            "GO Biological Process — Downregulated in Tumor",
            "go_bp_down_dotplot.png"
        )

    # GO combined barplot
    plot_go_combined_barplot(enrichment_results)

    # KEGG dotplots
    if "KEGG_up" in enrichment_results:
        plot_enrichr_dotplot(
            enrichment_results["KEGG_up"],
            "KEGG Pathways — Upregulated in Tumor",
            "kegg_up_dotplot.png"
        )
    if "KEGG_down" in enrichment_results:
        plot_enrichr_dotplot(
            enrichment_results["KEGG_down"],
            "KEGG Pathways — Downregulated in Tumor",
            "kegg_down_dotplot.png"
        )

    # GSEA Hallmark
    plot_gsea_hallmark(gsea_results)

    # GSEA enrichment score curves
    plot_gsea_enrichment_curves(pre_res)

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  Phase 4 complete!")
    print("=" * 60)
    print(f"\n  Results saved to: results/")
    print(f"  Figures saved to: figures/")
    print(f"\n  Next: python scripts/05_wgcna.py")


if __name__ == "__main__":
    main()
