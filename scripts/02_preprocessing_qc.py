"""
02_preprocessing_qc.py
=======================
Preprocessing and quality control of the TCGA-LUAD count matrix.

Usage:
    python scripts/02_preprocessing_qc.py

Input:
    data/raw_counts.csv
    data/clinical_data.csv
    data/sample_map.csv
    data/gene_id_mapping.csv

Output:
    data/filtered_counts.csv       - Filtered count matrix (low-expression genes removed)
    data/metadata.csv              - Merged sample + clinical metadata
    figures/pca_tumor_vs_normal.png
    figures/pca_clinical_variables.png
    figures/sample_count_distribution.png
    figures/gene_filtering_summary.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
FIG_DIR = PROJECT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Publication-quality figure settings
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.figsize": (8, 6),
})


# ──────────────────────────────────────────────
# Step 1: Load data
# ──────────────────────────────────────────────
def load_data():
    print("=" * 60)
    print("Step 1: Loading data...")
    print("=" * 60)

    counts = pd.read_csv(DATA_DIR / "raw_counts.csv", index_col=0)
    clinical = pd.read_csv(DATA_DIR / "clinical_data.csv")
    sample_map = pd.read_csv(DATA_DIR / "sample_map.csv")

    gene_mapping = None
    if (DATA_DIR / "gene_id_mapping.csv").exists():
        gene_mapping = pd.read_csv(DATA_DIR / "gene_id_mapping.csv", index_col=0)

    print(f"  Raw counts: {counts.shape[0]} genes × {counts.shape[1]} samples")
    print(f"  Clinical data: {clinical.shape[0]} cases")
    print(f"  Sample map: {sample_map.shape[0]} entries")

    return counts, clinical, sample_map, gene_mapping


# ──────────────────────────────────────────────
# Step 2: Build metadata (merge sample map + clinical)
# ──────────────────────────────────────────────
def build_metadata(counts, clinical, sample_map):
    print("\n" + "=" * 60)
    print("Step 2: Building sample metadata...")
    print("=" * 60)

    # Sample map has sample_submitter_id as key, clinical has submitter_id (patient-level)
    # Extract patient ID from sample barcode: TCGA-XX-XXXX-01A → TCGA-XX-XXXX
    sample_map = sample_map.copy()
    sample_map["patient_id"] = sample_map["sample_submitter_id"].str[:12]

    # Merge with clinical
    metadata = sample_map.merge(
        clinical, left_on="patient_id", right_on="submitter_id", how="left",
        suffixes=("_sample", "_clinical")
    )

    # Keep only samples present in the count matrix
    sample_cols = set(counts.columns)
    metadata = metadata[metadata["sample_submitter_id"].isin(sample_cols)].copy()

    # Ensure condition column exists
    if "condition" not in metadata.columns:
        metadata["condition"] = metadata["sample_type"].apply(
            lambda x: "Normal" if "Normal" in str(x) else "Tumor"
        )

    print(f"  Metadata shape: {metadata.shape}")
    print(f"  Condition breakdown:")
    print(f"    {metadata['condition'].value_counts().to_string()}")

    return metadata


# ──────────────────────────────────────────────
# Step 3: Handle duplicate samples
# ──────────────────────────────────────────────
def handle_duplicates(counts, metadata):
    print("\n" + "=" * 60)
    print("Step 3: Checking for duplicate samples...")
    print("=" * 60)

    # Check for duplicate patients (same patient, multiple tumor samples)
    tumor_meta = metadata[metadata["condition"] == "Tumor"]
    dup_patients = tumor_meta["patient_id"].value_counts()
    dup_patients = dup_patients[dup_patients > 1]

    if len(dup_patients) > 0:
        print(f"  Found {len(dup_patients)} patients with multiple tumor samples")
        print("  Keeping first sample per patient...")

        # Keep first occurrence per patient for tumor samples
        tumor_keep = tumor_meta.drop_duplicates("patient_id", keep="first")
        normal_meta = metadata[metadata["condition"] == "Normal"]
        metadata = pd.concat([tumor_keep, normal_meta], ignore_index=True)
    else:
        print("  No duplicate patients found")

    # Align count matrix columns with metadata
    valid_samples = metadata["sample_submitter_id"].tolist()
    counts = counts[[c for c in counts.columns if c in valid_samples]]

    print(f"  After dedup: {counts.shape[1]} samples")

    return counts, metadata


# ──────────────────────────────────────────────
# Step 4: Gene filtering
# ──────────────────────────────────────────────
def filter_genes(counts, metadata, min_count=10, min_samples_frac=0.1):
    """Remove lowly expressed genes. Keep genes with >= min_count reads
    in at least min_samples_frac of samples."""
    print("\n" + "=" * 60)
    print("Step 4: Filtering low-expression genes...")
    print("=" * 60)

    n_samples = counts.shape[1]
    min_samples = int(n_samples * min_samples_frac)

    genes_before = counts.shape[0]

    # Count how many samples have >= min_count for each gene
    passing = (counts >= min_count).sum(axis=1)
    keep_mask = passing >= min_samples

    # Also remove non-protein-coding gene types if mapping available
    filtered_counts = counts[keep_mask]

    genes_after = filtered_counts.shape[0]
    removed = genes_before - genes_after

    print(f"  Criteria: >= {min_count} counts in >= {min_samples} samples ({min_samples_frac*100:.0f}%)")
    print(f"  Before: {genes_before} genes")
    print(f"  After:  {genes_after} genes")
    print(f"  Removed: {removed} genes ({removed/genes_before*100:.1f}%)")

    # Plot filtering summary
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: histogram of counts per gene
    log_means = np.log10(counts.mean(axis=1) + 1)
    axes[0].hist(log_means, bins=100, color="#4a90d9", alpha=0.7, edgecolor="white")
    axes[0].axvline(np.log10(min_count), color="red", linestyle="--", label=f"Threshold ({min_count})")
    axes[0].set_xlabel("log10(mean count + 1)")
    axes[0].set_ylabel("Number of genes")
    axes[0].set_title("Gene Expression Distribution")
    axes[0].legend()

    # Right: before/after comparison
    categories = ["Before\nFiltering", "After\nFiltering"]
    values = [genes_before, genes_after]
    bars = axes[1].bar(categories, values, color=["#cccccc", "#4a90d9"], edgecolor="white", width=0.5)
    axes[1].set_ylabel("Number of Genes")
    axes[1].set_title("Gene Filtering Summary")
    for bar, val in zip(bars, values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                     f"{val:,}", ha="center", fontsize=11)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "gene_filtering_summary.png")
    plt.close()
    print(f"  Saved: figures/gene_filtering_summary.png")

    return filtered_counts


# ──────────────────────────────────────────────
# Step 5: Library size distribution
# ──────────────────────────────────────────────
def plot_library_sizes(counts, metadata):
    print("\n" + "=" * 60)
    print("Step 5: Plotting library size distribution...")
    print("=" * 60)

    lib_sizes = counts.sum(axis=0) / 1e6  # Millions

    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        "sample": lib_sizes.index,
        "library_size_M": lib_sizes.values,
    })

    # Map condition
    sample_to_condition = dict(zip(metadata["sample_submitter_id"], metadata["condition"]))
    plot_df["condition"] = plot_df["sample"].map(sample_to_condition)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: boxplot by condition
    colors = {"Tumor": "#e74c3c", "Normal": "#3498db"}
    for cond, color in colors.items():
        subset = plot_df[plot_df["condition"] == cond]
        axes[0].hist(subset["library_size_M"], bins=30, alpha=0.6, color=color, label=cond, edgecolor="white")

    axes[0].set_xlabel("Library Size (millions of reads)")
    axes[0].set_ylabel("Number of Samples")
    axes[0].set_title("Library Size Distribution")
    axes[0].legend()

    # Right: sorted library sizes
    plot_df_sorted = plot_df.sort_values("library_size_M")
    bar_colors = [colors.get(c, "#999999") for c in plot_df_sorted["condition"]]
    axes[1].bar(range(len(plot_df_sorted)), plot_df_sorted["library_size_M"],
                color=bar_colors, width=1.0, edgecolor="none")
    axes[1].set_xlabel("Samples (sorted)")
    axes[1].set_ylabel("Library Size (M reads)")
    axes[1].set_title("Library Sizes (sorted)")

    # Add legend manually
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=l) for l, c in colors.items()]
    axes[1].legend(handles=legend_elements)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "sample_count_distribution.png")
    plt.close()
    print(f"  Saved: figures/sample_count_distribution.png")

    # Flag potential outliers (< 5M or > 100M reads)
    outliers = plot_df[(plot_df["library_size_M"] < 5) | (plot_df["library_size_M"] > 100)]
    if len(outliers) > 0:
        print(f"  WARNING: {len(outliers)} potential outlier samples (library size < 5M or > 100M)")
    else:
        print("  No library size outliers detected")


# ──────────────────────────────────────────────
# Step 6: PCA analysis
# ──────────────────────────────────────────────
def run_pca(counts, metadata):
    print("\n" + "=" * 60)
    print("Step 6: Running PCA...")
    print("=" * 60)

    # Normalize: log2(CPM + 1)
    lib_sizes = counts.sum(axis=0)
    cpm = counts.div(lib_sizes, axis=1) * 1e6
    log_cpm = np.log2(cpm + 1)

    # Transpose: samples × genes
    X = log_cpm.T

    # Keep only the top 5000 most variable genes for PCA
    gene_var = X.var(axis=0)
    top_genes = gene_var.nlargest(5000).index
    X_sub = X[top_genes]

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sub)

    # PCA
    pca = PCA(n_components=10)
    pcs = pca.fit_transform(X_scaled)

    pc_df = pd.DataFrame(
        pcs[:, :10],
        columns=[f"PC{i+1}" for i in range(10)],
        index=X.index
    )

    # Map metadata
    sample_to_condition = dict(zip(metadata["sample_submitter_id"], metadata["condition"]))
    sample_to_stage = dict(zip(metadata["sample_submitter_id"], metadata["tumor_stage"]))
    sample_to_gender = dict(zip(metadata["sample_submitter_id"], metadata["gender"]))

    pc_df["condition"] = pc_df.index.map(sample_to_condition)
    pc_df["stage"] = pc_df.index.map(sample_to_stage)
    pc_df["gender"] = pc_df.index.map(sample_to_gender)

    var_explained = pca.explained_variance_ratio_ * 100

    print(f"  Variance explained:")
    for i in range(5):
        print(f"    PC{i+1}: {var_explained[i]:.1f}%")

    # ── Plot 1: PCA colored by Tumor vs Normal ──
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"Tumor": "#e74c3c", "Normal": "#3498db"}

    for cond, color in colors.items():
        mask = pc_df["condition"] == cond
        ax.scatter(pc_df.loc[mask, "PC1"], pc_df.loc[mask, "PC2"],
                   c=color, label=cond, alpha=0.6, s=30, edgecolors="white", linewidth=0.5)

    ax.set_xlabel(f"PC1 ({var_explained[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1f}%)")
    ax.set_title("PCA: Tumor vs Normal")
    ax.legend(title="Sample Type")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "pca_tumor_vs_normal.png")
    plt.close()
    print(f"  Saved: figures/pca_tumor_vs_normal.png")

    # ── Plot 2: PCA colored by clinical variables ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # By stage (simplify to I, II, III, IV)
    stage_map = {}
    for s in pc_df["stage"].dropna().unique():
        s_str = str(s).upper()
        if "IV" in s_str:
            stage_map[s] = "Stage IV"
        elif "III" in s_str:
            stage_map[s] = "Stage III"
        elif "II" in s_str:
            stage_map[s] = "Stage II"
        elif "I" in s_str:
            stage_map[s] = "Stage I"
        else:
            stage_map[s] = "Other"

    pc_df["stage_simple"] = pc_df["stage"].map(stage_map)

    stage_colors = {"Stage I": "#2ecc71", "Stage II": "#f1c40f",
                    "Stage III": "#e67e22", "Stage IV": "#e74c3c", "Other": "#95a5a6"}

    for stage, color in stage_colors.items():
        mask = pc_df["stage_simple"] == stage
        if mask.sum() > 0:
            axes[0].scatter(pc_df.loc[mask, "PC1"], pc_df.loc[mask, "PC2"],
                           c=color, label=stage, alpha=0.6, s=30, edgecolors="white", linewidth=0.5)

    axes[0].set_xlabel(f"PC1 ({var_explained[0]:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({var_explained[1]:.1f}%)")
    axes[0].set_title("PCA: By Tumor Stage")
    axes[0].legend(title="Stage", fontsize=9)
    axes[0].spines[["top", "right"]].set_visible(False)

    # By gender
    gender_colors = {"female": "#9b59b6", "male": "#3498db"}
    for g, color in gender_colors.items():
        mask = pc_df["gender"] == g
        if mask.sum() > 0:
            axes[1].scatter(pc_df.loc[mask, "PC1"], pc_df.loc[mask, "PC2"],
                           c=color, label=g.capitalize(), alpha=0.6, s=30,
                           edgecolors="white", linewidth=0.5)

    axes[1].set_xlabel(f"PC1 ({var_explained[0]:.1f}%)")
    axes[1].set_ylabel(f"PC2 ({var_explained[1]:.1f}%)")
    axes[1].set_title("PCA: By Gender")
    axes[1].legend(title="Gender", fontsize=9)
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "pca_clinical_variables.png")
    plt.close()
    print(f"  Saved: figures/pca_clinical_variables.png")

    return pc_df, var_explained


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("  TCGA-LUAD Preprocessing & Quality Control")
    print("=" * 60 + "\n")

    # Load
    counts, clinical, sample_map, gene_mapping = load_data()

    # Build metadata
    metadata = build_metadata(counts, clinical, sample_map)

    # Handle duplicates
    counts, metadata = handle_duplicates(counts, metadata)

    # Filter genes
    filtered_counts = filter_genes(counts, metadata)

    # Library size plots
    plot_library_sizes(filtered_counts, metadata)

    # PCA
    pc_df, var_explained = run_pca(filtered_counts, metadata)

    # ── Save outputs ──
    print("\n" + "=" * 60)
    print("Saving outputs...")
    print("=" * 60)

    filtered_counts.to_csv(DATA_DIR / "filtered_counts.csv")
    print(f"  Saved: data/filtered_counts.csv ({filtered_counts.shape[0]} genes × {filtered_counts.shape[1]} samples)")

    metadata.to_csv(DATA_DIR / "metadata.csv", index=False)
    print(f"  Saved: data/metadata.csv ({len(metadata)} samples)")

    pc_df.to_csv(DATA_DIR / "pca_results.csv")
    print(f"  Saved: data/pca_results.csv")

    print(f"\nDone! Check the figures/ folder for QC plots.")
    print(f"Next step: Run python scripts/03_differential_expression.py")


if __name__ == "__main__":
    main()
