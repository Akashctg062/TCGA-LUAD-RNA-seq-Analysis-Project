"""
08_drug_target_analysis.py
===========================
Drug sensitivity and therapeutic target analysis.
Correlates hub gene expression with predicted drug response using
GDSC (Genomics of Drug Sensitivity in Cancer) IC50 data via CellMiner.

Since direct GDSC API access may be limited, this script uses a
knowledge-based approach with known drug-gene associations from
literature and computes expression-based drug sensitivity signatures.

Usage:
    python scripts/08_drug_target_analysis.py

Output:
    results/drug_gene_associations.csv
    results/ceRNA_network.csv
    figures/drug_target_heatmap.png
    figures/drug_sensitivity_correlation.png
    figures/ceRNA_regulatory_network.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import networkx as nx
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
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

# Hub genes from Phase 6
TOP_GENES = ["BIRC5", "CCNB1", "CCNB2", "CDC20", "KIF2C"]

# Known drug-gene associations (curated from literature / DrugBank / CIViC)
DRUG_GENE_DB = {
    "BIRC5": {
        "drugs": ["YM155 (Sepantronium)", "LY2181308", "Shepherdin"],
        "pathways": ["Apoptosis evasion", "IAP family"],
        "mechanism": "Survivin inhibitor — restores apoptosis in tumor cells",
        "clinical_stage": "Phase II (YM155 in NSCLC)",
        "references": ["Giaccone et al., JCO 2009", "Kelly et al., JCO 2013"],
    },
    "CCNB1": {
        "drugs": ["Paclitaxel", "Docetaxel", "Vinorelbine", "RO-3306"],
        "pathways": ["Cell cycle (G2/M)", "Mitotic spindle"],
        "mechanism": "Cyclin B1 drives G2/M; taxanes trap cells in mitosis",
        "clinical_stage": "FDA-approved (taxanes for NSCLC)",
        "references": ["Soria et al., NEJM 2018"],
    },
    "CCNB2": {
        "drugs": ["Paclitaxel", "Palbociclib", "Dinaciclib"],
        "pathways": ["Cell cycle (G2/M)", "CDK1 activation"],
        "mechanism": "CDK inhibitors block cyclin B2-dependent mitotic entry",
        "clinical_stage": "Phase III (Palbociclib combinations)",
        "references": ["Asghar et al., Nat Rev Drug Discov 2015"],
    },
    "CDC20": {
        "drugs": ["Apcin", "pro-TAME", "CFM-4.16"],
        "pathways": ["APC/C complex", "Spindle assembly checkpoint"],
        "mechanism": "CDC20 inhibitors block APC/C, arrest mitosis",
        "clinical_stage": "Preclinical",
        "references": ["Zeng et al., Cancer Lett 2016"],
    },
    "KIF2C": {
        "drugs": ["Ispinesib", "GSK923295", "ARRY-520"],
        "pathways": ["Kinesin motor proteins", "Mitotic spindle"],
        "mechanism": "Kinesin inhibitors disrupt spindle formation",
        "clinical_stage": "Phase II (Ispinesib in solid tumors)",
        "references": ["Rath & Bhatt, J Mol Med 2015"],
    },
}

# Chemotherapy sensitivity gene signatures (from literature)
CHEMO_SIGNATURES = {
    "Cisplatin_resistance": {
        "up": ["ERCC1", "BRCA1", "RRM1", "ABCC2", "MSH2"],
        "down": ["TP53", "BAX", "CASP3"],
    },
    "Paclitaxel_sensitivity": {
        "up": ["TUBB3", "CCNB1", "CCNB2", "CDC20", "MAD2L1", "BUB1"],
        "down": ["ABCB1", "CYP2C8"],
    },
    "Pemetrexed_sensitivity": {
        "up": ["TYMS", "DHFR", "GART"],
        "down": ["SLC19A1"],
    },
    "Immunotherapy_response": {
        "up": ["CD274", "PDCD1", "CTLA4", "IFNG", "GZMB", "PRF1", "CXCL9", "CXCL10"],
        "down": ["TGFB1", "VEGFA", "IL10"],
    },
}


def drug_gene_association_analysis():
    """Summarize known drug-gene associations for hub genes."""
    print("\n" + "=" * 60)
    print("Step 1: Drug-Gene Association Analysis")
    print("=" * 60)

    rows = []
    for gene, info in DRUG_GENE_DB.items():
        for drug in info["drugs"]:
            rows.append({
                "gene": gene,
                "drug": drug,
                "pathway": "; ".join(info["pathways"]),
                "mechanism": info["mechanism"],
                "clinical_stage": info["clinical_stage"],
                "references": "; ".join(info["references"]),
            })
    drug_df = pd.DataFrame(rows)
    drug_df.to_csv(RESULTS_DIR / "drug_gene_associations.csv", index=False)

    print(f"  Found {len(drug_df)} drug-gene associations for {len(DRUG_GENE_DB)} hub genes:\n")
    for gene, info in DRUG_GENE_DB.items():
        print(f"  {gene}:")
        print(f"    Drugs: {', '.join(info['drugs'])}")
        print(f"    Stage: {info['clinical_stage']}")
        print(f"    Mechanism: {info['mechanism']}")
        print()

    return drug_df


def compute_chemo_sensitivity_scores(log_cpm, metadata, gene_mapping):
    """Compute chemotherapy sensitivity signature scores."""
    print("=" * 60)
    print("Step 2: Chemotherapy Sensitivity Signature Scoring")
    print("=" * 60)

    if gene_mapping is not None:
        name_to_id = dict(zip(gene_mapping["gene_name"], gene_mapping.index))
    else:
        name_to_id = {}

    meta = metadata.set_index("sample_submitter_id")
    tumor_samples = [s for s in log_cpm.columns if s in meta.index and meta.loc[s, "condition"] == "Tumor"]

    scores = {}
    for sig_name, sig_genes in CHEMO_SIGNATURES.items():
        up_ids = [name_to_id.get(g, g) for g in sig_genes["up"]]
        down_ids = [name_to_id.get(g, g) for g in sig_genes["down"]]

        up_found = [g for g in up_ids if g in log_cpm.index]
        down_found = [g for g in down_ids if g in log_cpm.index]

        print(f"  {sig_name}: {len(up_found)}/{len(sig_genes['up'])} up, "
              f"{len(down_found)}/{len(sig_genes['down'])} down markers found")

        if len(up_found) + len(down_found) < 2:
            continue

        # Signature score = mean(up z-scores) - mean(down z-scores)
        expr_tumor = log_cpm[tumor_samples]

        up_score = 0
        if up_found:
            up_z = (expr_tumor.loc[up_found] - expr_tumor.loc[up_found].mean(axis=1).values.reshape(-1,1)) / \
                   (expr_tumor.loc[up_found].std(axis=1).values.reshape(-1,1) + 1e-10)
            up_score = up_z.mean(axis=0)

        down_score = 0
        if down_found:
            down_z = (expr_tumor.loc[down_found] - expr_tumor.loc[down_found].mean(axis=1).values.reshape(-1,1)) / \
                     (expr_tumor.loc[down_found].std(axis=1).values.reshape(-1,1) + 1e-10)
            down_score = down_z.mean(axis=0)

        scores[sig_name] = up_score - down_score

    score_df = pd.DataFrame(scores, index=tumor_samples)
    return score_df


def correlate_hubs_with_drug_sensitivity(log_cpm, score_df, gene_mapping):
    """Correlate hub gene expression with drug sensitivity scores."""
    print("\n" + "=" * 60)
    print("Step 3: Hub Gene — Drug Sensitivity Correlation")
    print("=" * 60)

    if gene_mapping is not None:
        name_to_id = dict(zip(gene_mapping["gene_name"], gene_mapping.index))
    else:
        name_to_id = {}

    tumor_samples = score_df.index.tolist()

    cor_matrix = pd.DataFrame(index=TOP_GENES, columns=score_df.columns, dtype=float)
    pval_matrix = pd.DataFrame(index=TOP_GENES, columns=score_df.columns, dtype=float)

    for gene in TOP_GENES:
        gene_id = name_to_id.get(gene, gene)
        if gene_id not in log_cpm.index:
            continue

        gene_expr = log_cpm.loc[gene_id, tumor_samples]

        for sig in score_df.columns:
            valid = gene_expr.notna() & score_df[sig].notna()
            if valid.sum() > 10:
                r, p = spearmanr(gene_expr[valid], score_df.loc[valid, sig])
                cor_matrix.loc[gene, sig] = r
                pval_matrix.loc[gene, sig] = p

    cor_matrix = cor_matrix.astype(float)

    print("\n  Correlation results (Spearman):")
    for gene in TOP_GENES:
        for sig in score_df.columns:
            r = cor_matrix.loc[gene, sig]
            p = pval_matrix.loc[gene, sig]
            if pd.notna(r):
                star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                if abs(r) > 0.15:
                    print(f"    {gene:10s} × {sig:25s}  rho={r:+.3f} {star}")

    return cor_matrix, pval_matrix


def plot_drug_target_heatmap(drug_df):
    """Heatmap-style visualization of drug-gene associations."""
    print("\n  Plotting drug-target association map...")

    # Create binary matrix
    all_drugs = drug_df["drug"].unique()
    all_genes = drug_df["gene"].unique()

    mat = pd.DataFrame(0, index=all_genes, columns=all_drugs)
    for _, row in drug_df.iterrows():
        mat.loc[row["gene"], row["drug"]] = 1

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(mat, annot=True, fmt="d", cmap="YlOrRd", ax=ax,
                linewidths=1, cbar=False, annot_kws={"fontsize": 12})
    ax.set_title("Hub Gene — Drug Target Associations", fontsize=14)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=11)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "drug_target_heatmap.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: figures/drug_target_heatmap.png")


def plot_drug_sensitivity_correlation(cor_matrix, pval_matrix):
    """Heatmap of hub gene vs drug sensitivity signature correlations."""
    print("  Plotting drug sensitivity correlation heatmap...")

    # Rename columns for display
    display_names = {
        "Cisplatin_resistance": "Cisplatin\nResistance",
        "Paclitaxel_sensitivity": "Paclitaxel\nSensitivity",
        "Pemetrexed_sensitivity": "Pemetrexed\nSensitivity",
        "Immunotherapy_response": "Immunotherapy\nResponse",
    }

    cor_display = cor_matrix.rename(columns=display_names)

    # Significance annotations
    annot = cor_display.copy().astype(str)
    for gene in cor_matrix.index:
        for sig in cor_matrix.columns:
            r = cor_matrix.loc[gene, sig]
            p = pval_matrix.loc[gene, sig]
            if pd.notna(r):
                star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                annot.loc[gene, display_names.get(sig, sig)] = f"{r:.2f}{star}"
            else:
                annot.loc[gene, display_names.get(sig, sig)] = ""

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        cor_display.astype(float), annot=annot, fmt="",
        cmap="RdBu_r", vmin=-0.5, vmax=0.5, center=0,
        linewidths=1, ax=ax,
        cbar_kws={"label": "Spearman Correlation", "shrink": 0.8},
        annot_kws={"fontsize": 12},
    )
    ax.set_title("Hub Gene Expression vs Drug Sensitivity Signatures\n(TCGA-LUAD Tumor Samples)",
                 fontsize=13)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=11)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "drug_sensitivity_correlation.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: figures/drug_sensitivity_correlation.png")


def build_ceRNA_regulatory_network(log_cpm, metadata, gene_mapping):
    """Build a ceRNA/regulatory network showing hub gene interactions
    with key regulators (transcription factors, miRNA targets)."""
    print("\n" + "=" * 60)
    print("Step 4: Regulatory Network Analysis")
    print("=" * 60)

    if gene_mapping is not None:
        name_to_id = dict(zip(gene_mapping["gene_name"], gene_mapping.index))
    else:
        name_to_id = {}

    meta = metadata.set_index("sample_submitter_id")
    tumor_samples = [s for s in log_cpm.columns if s in meta.index and meta.loc[s, "condition"] == "Tumor"]

    # Known transcription factors regulating hub genes
    TF_TARGETS = {
        "E2F1": ["BIRC5", "CCNB1", "CCNB2", "CDC20"],
        "FOXM1": ["BIRC5", "CCNB1", "CCNB2", "CDC20", "KIF2C"],
        "MYC": ["BIRC5", "CCNB1", "CDC20"],
        "TP53": ["BIRC5", "CCNB1", "CDC20"],
        "STAT3": ["BIRC5", "CCNB1"],
        "NF-kB (RELA)": ["BIRC5"],
    }

    # Build network
    G = nx.Graph()
    edges_data = []

    for tf, targets in TF_TARGETS.items():
        tf_gene = tf.replace(" (RELA)", "")
        if tf_gene == "NF-kB":
            tf_gene = "RELA"
        tf_id = name_to_id.get(tf_gene, tf_gene)

        for target in targets:
            target_id = name_to_id.get(target, target)

            # Compute correlation if both genes exist
            r_val = np.nan
            if tf_id in log_cpm.index and target_id in log_cpm.index:
                tf_expr = log_cpm.loc[tf_id, tumor_samples]
                target_expr = log_cpm.loc[target_id, tumor_samples]
                valid = tf_expr.notna() & target_expr.notna()
                if valid.sum() > 10:
                    r_val, _ = pearsonr(tf_expr[valid], target_expr[valid])

            G.add_node(tf, node_type="TF")
            G.add_node(target, node_type="Hub Gene")
            G.add_edge(tf, target, weight=abs(r_val) if pd.notna(r_val) else 0.5,
                       correlation=r_val)

            edges_data.append({
                "TF": tf, "target": target, "correlation": r_val
            })

    edges_df = pd.DataFrame(edges_data)
    edges_df.to_csv(RESULTS_DIR / "ceRNA_network.csv", index=False)

    print(f"  Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"\n  TF-Target correlations:")
    for _, row in edges_df.iterrows():
        r = row["correlation"]
        if pd.notna(r):
            print(f"    {row['TF']:15s} → {row['target']:10s}  r={r:+.3f}")

    # Plot
    print("\n  Plotting regulatory network...")
    fig, ax = plt.subplots(figsize=(12, 10))

    pos = nx.spring_layout(G, k=3.5, iterations=200, seed=42)

    # Node styling
    node_colors = []
    node_sizes = []
    for node in G.nodes:
        if G.nodes[node].get("node_type") == "TF":
            node_colors.append("#FFD700")  # Gold for TFs
            node_sizes.append(1200)
        else:
            node_colors.append("#FF6B6B")  # Red for hub genes
            node_sizes.append(1800)

    # Edge styling (color by correlation)
    edge_colors = []
    edge_widths = []
    for u, v in G.edges:
        r = G[u][v].get("correlation", 0)
        if pd.notna(r):
            edge_colors.append("#e74c3c" if r > 0 else "#3498db")
            edge_widths.append(abs(r) * 4 + 0.5)
        else:
            edge_colors.append("#cccccc")
            edge_widths.append(1)

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors,
                           width=edge_widths, alpha=0.6)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, alpha=0.9,
                           edgecolors="white", linewidths=2)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight="bold")

    # Legend
    legend_handles = [
        mpatches.Patch(color="#FFD700", label="Transcription Factor"),
        mpatches.Patch(color="#FF6B6B", label="Hub Gene"),
        plt.Line2D([0], [0], color="#e74c3c", lw=2, label="Positive Correlation"),
        plt.Line2D([0], [0], color="#3498db", lw=2, label="Negative Correlation"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=10)
    ax.set_title("Transcription Factor — Hub Gene Regulatory Network\n(TCGA-LUAD)",
                 fontsize=14)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "ceRNA_regulatory_network.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: figures/ceRNA_regulatory_network.png")


def main():
    print("\n" + "=" * 60)
    print("  Phase 8: Drug Target & Regulatory Network Analysis")
    print("=" * 60 + "\n")

    # Load data
    print("Loading data...")
    counts = pd.read_csv(DATA_DIR / "filtered_counts.csv", index_col=0)
    metadata = pd.read_csv(DATA_DIR / "metadata.csv")
    gene_mapping = None
    if (DATA_DIR / "gene_id_mapping.csv").exists():
        gene_mapping = pd.read_csv(DATA_DIR / "gene_id_mapping.csv", index_col=0)

    lib_sizes = counts.sum(axis=0)
    cpm = counts.div(lib_sizes, axis=1) * 1e6
    log_cpm = np.log2(cpm + 1)

    # Step 1: Drug-gene associations
    drug_df = drug_gene_association_analysis()

    # Step 2: Chemo sensitivity scoring
    score_df = compute_chemo_sensitivity_scores(log_cpm, metadata, gene_mapping)

    # Step 3: Correlate hub genes with drug sensitivity
    cor_matrix, pval_matrix = correlate_hubs_with_drug_sensitivity(
        log_cpm, score_df, gene_mapping
    )

    # Step 4: Regulatory network
    build_ceRNA_regulatory_network(log_cpm, metadata, gene_mapping)

    # Plots
    plot_drug_target_heatmap(drug_df)
    plot_drug_sensitivity_correlation(cor_matrix, pval_matrix)

    print(f"\n{'=' * 60}")
    print(f"  Phase 8 complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
