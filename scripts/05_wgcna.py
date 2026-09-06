"""
05_wgcna.py
============
Weighted Gene Co-expression Network Analysis (WGCNA) for TCGA-LUAD.
Implements the core WGCNA pipeline using numpy/scipy.

Usage:
    python scripts/05_wgcna.py

Input:
    data/filtered_counts.csv
    data/metadata.csv
    data/gene_id_mapping.csv

Output:
    results/wgcna_modules.csv            - Gene → module assignments
    results/wgcna_module_trait_cor.csv    - Module-trait correlations
    results/wgcna_hub_genes.csv          - Hub genes per module
    figures/wgcna_soft_threshold.png
    figures/wgcna_dendrogram_modules.png
    figures/wgcna_module_trait_heatmap.png
    figures/wgcna_module_sizes.png

Notes:
    - Uses top 5,000 most variable genes (MAD-based)
    - Memory: ~1-2 GB peak for 5000 genes
    - Runtime: ~15-30 minutes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
import sys
import warnings
warnings.filterwarnings("ignore")
sys.setrecursionlimit(10000)  # Needed for dendrogram with many genes

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
FIG_DIR = PROJECT_DIR / "figures"

N_TOP_GENES = 5000           # Number of most variable genes to use
MIN_MODULE_SIZE = 30         # Minimum genes per module
MERGE_CUT_HEIGHT = 0.25     # Merge modules with > 0.75 correlation

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
})


# ──────────────────────────────────────────────
# Step 1: Prepare expression matrix
# ──────────────────────────────────────────────
def prepare_expression():
    print("=" * 60)
    print("Step 1: Preparing expression data...")
    print("=" * 60)

    counts = pd.read_csv(DATA_DIR / "filtered_counts.csv", index_col=0)
    metadata = pd.read_csv(DATA_DIR / "metadata.csv")

    gene_mapping = None
    if (DATA_DIR / "gene_id_mapping.csv").exists():
        gene_mapping = pd.read_csv(DATA_DIR / "gene_id_mapping.csv", index_col=0)

    # Normalize: log2(CPM + 1)
    lib_sizes = counts.sum(axis=0)
    cpm = counts.div(lib_sizes, axis=1) * 1e6
    log_cpm = np.log2(cpm + 1)

    # Select top N most variable genes by MAD (Median Absolute Deviation)
    mad = log_cpm.subtract(log_cpm.median(axis=1), axis=0).abs().median(axis=1)
    top_genes = mad.nlargest(N_TOP_GENES).index
    expr = log_cpm.loc[top_genes].T  # samples × genes

    print(f"  Expression matrix: {expr.shape[0]} samples × {expr.shape[1]} genes")
    print(f"  MAD range: {mad[top_genes].min():.3f} - {mad[top_genes].max():.3f}")

    return expr, metadata, gene_mapping


# ──────────────────────────────────────────────
# Step 2: Soft-threshold power selection
# ──────────────────────────────────────────────
def pick_soft_threshold(expr, powers=None):
    print("\n" + "=" * 60)
    print("Step 2: Picking soft-threshold power...")
    print("=" * 60)

    if powers is None:
        powers = list(range(1, 21))

    # Compute correlation matrix once
    print("  Computing correlation matrix...")
    cor_matrix = np.corrcoef(expr.values.T)  # genes × genes
    np.fill_diagonal(cor_matrix, 0)

    results = []
    for power in powers:
        # Adjacency
        adj = np.abs(cor_matrix) ** power

        # Connectivity (sum of adjacency for each gene)
        k = adj.sum(axis=0)

        # Scale-free topology fit: R² of log(k) vs log(p(k))
        k_nonzero = k[k > 0]
        log_k = np.log10(k_nonzero)

        # Bin connectivity into histogram
        n_bins = 30
        hist, bin_edges = np.histogram(log_k, bins=n_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Keep non-zero bins
        mask = hist > 0
        if mask.sum() < 3:
            results.append({"power": power, "r_squared": 0, "mean_k": k.mean(), "slope": 0})
            continue

        log_p = np.log10(hist[mask] + 1)
        x = bin_centers[mask]

        # Linear regression for R²
        slope, intercept = np.polyfit(x, log_p, 1)
        predicted = slope * x + intercept
        ss_res = np.sum((log_p - predicted) ** 2)
        ss_tot = np.sum((log_p - log_p.mean()) ** 2)
        r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Scale-free R² with sign (negative slope expected)
        signed_r_sq = -r_sq if slope > 0 else r_sq

        results.append({
            "power": power,
            "r_squared": signed_r_sq,
            "mean_k": k.mean(),
            "slope": slope,
        })

    results_df = pd.DataFrame(results)

    # Pick power: first power where R² > 0.85, or the one closest
    good = results_df[results_df["r_squared"] > 0.85]
    if len(good) > 0:
        best_power = good.iloc[0]["power"]
    else:
        best_power = results_df.loc[results_df["r_squared"].idxmax(), "power"]
    best_power = int(best_power)

    print(f"\n  Power selection results:")
    for _, row in results_df.iterrows():
        marker = " <-- selected" if int(row["power"]) == best_power else ""
        print(f"    Power {int(row['power']):2d}: R²={row['r_squared']:.3f}, "
              f"mean(k)={row['mean_k']:.1f}{marker}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(results_df["power"], results_df["r_squared"], "o-", color="#e74c3c", markersize=6)
    axes[0].axhline(0.85, color="grey", linestyle="--", linewidth=0.8, label="R² = 0.85")
    axes[0].axvline(best_power, color="#3498db", linestyle="--", linewidth=0.8, alpha=0.7)
    axes[0].set_xlabel("Soft Threshold Power")
    axes[0].set_ylabel("Scale-Free Topology Fit (R²)")
    axes[0].set_title("Scale-Free Topology Fit")
    axes[0].legend(fontsize=9)
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].plot(results_df["power"], results_df["mean_k"], "o-", color="#3498db", markersize=6)
    axes[1].axvline(best_power, color="#e74c3c", linestyle="--", linewidth=0.8, alpha=0.7)
    axes[1].set_xlabel("Soft Threshold Power")
    axes[1].set_ylabel("Mean Connectivity")
    axes[1].set_title("Mean Connectivity")
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.suptitle(f"Soft Threshold Selection (power = {best_power})", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "wgcna_soft_threshold.png", bbox_inches="tight")
    plt.close()
    print(f"\n  Selected power: {best_power}")
    print(f"  Saved: figures/wgcna_soft_threshold.png")

    return best_power, cor_matrix


# ──────────────────────────────────────────────
# Step 3: Build TOM and cluster genes into modules
# ──────────────────────────────────────────────
def build_network_and_detect_modules(cor_matrix, power, gene_ids):
    print("\n" + "=" * 60)
    print("Step 3: Building network and detecting modules...")
    print("=" * 60)

    n_genes = cor_matrix.shape[0]

    # Adjacency matrix
    print("  Computing adjacency matrix...")
    adj = np.abs(cor_matrix) ** power

    # TOM (Topological Overlap Matrix)
    print("  Computing TOM (this takes a few minutes)...")
    # TOM_ij = (sum_u(a_iu * a_uj) + a_ij) / (min(k_i, k_j) + 1 - a_ij)
    # where k_i = sum_j(a_ij)

    connectivity = adj.sum(axis=0) - np.diag(adj)  # exclude self

    # Numerator: A @ A gives sum_u(a_iu * a_uj) for each pair
    print("    Matrix multiplication (largest step)...")
    numerator = adj @ adj + adj
    np.fill_diagonal(numerator, 0)

    # Denominator
    k_matrix = np.minimum(
        connectivity[:, np.newaxis],
        connectivity[np.newaxis, :]
    )
    denominator = k_matrix + 1 - adj
    denominator[denominator == 0] = 1  # Avoid division by zero

    tom = numerator / denominator
    np.fill_diagonal(tom, 1)

    # Clip to [0, 1]
    tom = np.clip(tom, 0, 1)

    # Distance = 1 - TOM
    print("  Computing distance matrix...")
    dist = 1 - tom
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, None)

    # Free memory
    del adj, numerator, denominator, k_matrix
    import gc; gc.collect()

    # Hierarchical clustering
    print("  Hierarchical clustering...")
    condensed_dist = squareform(dist, checks=False)
    Z = linkage(condensed_dist, method="average")

    # Dynamic tree cut: scan a wide range of cut heights to find the one
    # that yields a reasonable number of modules (target: 8-25)
    print("  Detecting modules (adaptive tree cut)...")
    best_labels = None
    best_n_modules = 0
    best_height = None

    # Scan from low to high — lower height = more modules
    for h in np.arange(0.70, 0.99, 0.005):
        labels = fcluster(Z, t=h, criterion="distance")
        unique, counts = np.unique(labels, return_counts=True)
        valid_modules = unique[counts >= MIN_MODULE_SIZE]
        n_valid = len(valid_modules)

        # Prefer 8-25 modules; accept 5-30
        if 8 <= n_valid <= 25:
            best_n_modules = n_valid
            best_labels = labels.copy()
            best_height = h
            break  # First hit in the sweet spot
        elif 5 <= n_valid <= 30 and n_valid > best_n_modules:
            best_n_modules = n_valid
            best_labels = labels.copy()
            best_height = h

    if best_labels is None:
        # Fallback: use maxclust criterion
        print("  Distance-based cut failed, trying maxclust=15...")
        best_labels = fcluster(Z, t=15, criterion="maxclust")
        best_height = "maxclust=15"

    # Reassign small modules to module 0 (grey/unassigned)
    unique, counts = np.unique(best_labels, return_counts=True)
    small_modules = unique[counts < MIN_MODULE_SIZE]
    for sm in small_modules:
        best_labels[best_labels == sm] = 0

    # Renumber modules consecutively
    unique_modules = sorted(set(best_labels))
    module_map = {old: new for new, old in enumerate(unique_modules)}
    final_labels = np.array([module_map[l] for l in best_labels])

    # Module summary
    unique_final, counts_final = np.unique(final_labels, return_counts=True)
    n_modules = len(unique_final) - (1 if 0 in unique_final else 0)

    print(f"\n  Cut height: {best_height:.2f}")
    print(f"  Modules detected: {n_modules} (+ unassigned)")
    print(f"  Module sizes:")
    for mod, cnt in sorted(zip(unique_final, counts_final), key=lambda x: -x[1]):
        label = "Unassigned" if mod == 0 else f"Module {mod}"
        print(f"    {label:15s}: {cnt:,} genes")

    # Assign colors to modules
    color_palette = sns.color_palette("husl", n_colors=max(n_modules, 1))
    module_colors = {}
    color_idx = 0
    for mod in sorted(unique_final):
        if mod == 0:
            module_colors[mod] = "#cccccc"  # grey for unassigned
        else:
            module_colors[mod] = mcolors.rgb2hex(color_palette[color_idx])
            color_idx += 1

    # Plot dendrogram with module colors
    print("\n  Plotting dendrogram...")
    fig, axes = plt.subplots(2, 1, figsize=(16, 8),
                              gridspec_kw={"height_ratios": [4, 0.5]})

    # Dendrogram (truncated to avoid recursion issues with large gene sets)
    cut_h = best_height if isinstance(best_height, (int, float)) else 0.85
    dendro = dendrogram(
        Z, ax=axes[0], no_labels=True, color_threshold=cut_h,
        above_threshold_color="grey", leaf_font_size=0,
        truncate_mode="lastp", p=200,  # Show last 200 merged clusters
    )
    axes[0].axhline(cut_h, color="red", linestyle="--", linewidth=1, alpha=0.7)
    axes[0].set_title(f"Gene Dendrogram and Module Colors ({n_modules} modules)")
    axes[0].set_ylabel("Distance (1 - TOM)")

    # Module color bar — build from full gene ordering (not truncated leaves)
    # When truncated, leaves contain cluster IDs (>= n_genes), so we use
    # the original linkage order instead
    from scipy.cluster.hierarchy import leaves_list
    full_leaf_order = leaves_list(Z)
    ordered_colors = [module_colors[final_labels[i]] for i in full_leaf_order]

    for i, color in enumerate(ordered_colors):
        axes[1].axvspan(i - 0.5, i + 0.5, color=color)
    axes[1].set_xlim(-0.5, len(full_leaf_order) - 0.5)
    axes[1].set_yticks([])
    axes[1].set_xlabel("Genes")
    axes[1].set_ylabel("Module", fontsize=10)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "wgcna_dendrogram_modules.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: figures/wgcna_dendrogram_modules.png")

    # Create results DataFrame
    module_df = pd.DataFrame({
        "gene_id": gene_ids,
        "module": final_labels,
        "module_color": [module_colors[m] for m in final_labels],
    })

    return module_df, final_labels, module_colors, tom, Z


# ──────────────────────────────────────────────
# Step 4: Module eigengenes and trait correlation
# ──────────────────────────────────────────────
def compute_module_eigengenes(expr, module_df):
    """Compute first principal component of each module."""
    print("\n" + "=" * 60)
    print("Step 4: Computing module eigengenes...")
    print("=" * 60)

    from sklearn.decomposition import PCA

    eigengenes = {}
    modules = sorted(module_df["module"].unique())

    for mod in modules:
        if mod == 0:
            continue  # Skip unassigned

        gene_mask = module_df["module"] == mod
        gene_ids = module_df.loc[gene_mask, "gene_id"].values
        mod_expr = expr[gene_ids].values  # samples × genes

        # Standardize
        mod_expr_std = (mod_expr - mod_expr.mean(axis=0)) / (mod_expr.std(axis=0) + 1e-10)

        # First PC
        pca = PCA(n_components=1)
        me = pca.fit_transform(mod_expr_std).flatten()

        # Ensure ME is positively correlated with mean expression
        mean_expr = mod_expr.mean(axis=1)
        if np.corrcoef(me, mean_expr)[0, 1] < 0:
            me = -me

        eigengenes[f"ME{mod}"] = me
        var_explained = pca.explained_variance_ratio_[0]
        print(f"  Module {mod}: {gene_mask.sum()} genes, ME variance explained: {var_explained:.1%}")

    eigengene_df = pd.DataFrame(eigengenes, index=expr.index)
    return eigengene_df


def correlate_with_traits(eigengene_df, metadata, expr):
    """Correlate module eigengenes with clinical traits."""
    print("\n" + "=" * 60)
    print("Step 5: Correlating modules with clinical traits...")
    print("=" * 60)

    # Prepare trait matrix (numeric encoding)
    meta = metadata.set_index("sample_submitter_id")
    common = eigengene_df.index.intersection(meta.index)
    eigengene_df = eigengene_df.loc[common]
    meta = meta.loc[common]

    traits = pd.DataFrame(index=common)

    # Condition: Tumor=1, Normal=0
    traits["Tumor_status"] = (meta["condition"] == "Tumor").astype(int)

    # Stage (ordinal)
    stage_map = {}
    for s in meta["tumor_stage"].dropna().unique():
        s_str = str(s).upper()
        if "IV" in s_str:
            stage_map[s] = 4
        elif "III" in s_str:
            stage_map[s] = 3
        elif "II" in s_str:
            stage_map[s] = 2
        elif "I" in s_str:
            stage_map[s] = 1
    traits["Stage"] = meta["tumor_stage"].map(stage_map)

    # Survival event
    if "os_event" in meta.columns:
        traits["OS_event"] = meta["os_event"].astype(float)

    # Survival time
    if "os_time_days" in meta.columns:
        traits["OS_time"] = meta["os_time_days"].astype(float)

    # Age
    if "age_years" in meta.columns:
        traits["Age"] = meta["age_years"].astype(float)

    # Compute correlations and p-values
    cor_matrix = pd.DataFrame(
        index=eigengene_df.columns, columns=traits.columns, dtype=float
    )
    pval_matrix = pd.DataFrame(
        index=eigengene_df.columns, columns=traits.columns, dtype=float
    )

    for me_col in eigengene_df.columns:
        for trait_col in traits.columns:
            # Drop NaN pairs
            valid = eigengene_df[me_col].notna() & traits[trait_col].notna()
            if valid.sum() < 10:
                cor_matrix.loc[me_col, trait_col] = np.nan
                pval_matrix.loc[me_col, trait_col] = np.nan
                continue

            r, p = pearsonr(
                eigengene_df.loc[valid, me_col],
                traits.loc[valid, trait_col]
            )
            cor_matrix.loc[me_col, trait_col] = r
            pval_matrix.loc[me_col, trait_col] = p

    # Print significant associations
    print(f"\n  Module-trait correlations (|r| > 0.3):")
    for me_col in cor_matrix.index:
        for trait_col in cor_matrix.columns:
            r = cor_matrix.loc[me_col, trait_col]
            p = pval_matrix.loc[me_col, trait_col]
            if pd.notna(r) and abs(r) > 0.3:
                print(f"    {me_col} ~ {trait_col}: r={r:+.3f}, p={p:.2e}")

    # Plot heatmap
    print("\n  Plotting module-trait heatmap...")
    fig, ax = plt.subplots(figsize=(8, max(4, len(cor_matrix) * 0.5)))

    # Annotation: r (p-value)
    annot = cor_matrix.copy().astype(str)
    for me_col in cor_matrix.index:
        for trait_col in cor_matrix.columns:
            r = cor_matrix.loc[me_col, trait_col]
            p = pval_matrix.loc[me_col, trait_col]
            if pd.notna(r):
                stars = ""
                if p < 0.001:
                    stars = "***"
                elif p < 0.01:
                    stars = "**"
                elif p < 0.05:
                    stars = "*"
                annot.loc[me_col, trait_col] = f"{r:.2f}{stars}"
            else:
                annot.loc[me_col, trait_col] = ""

    # Clean module labels
    row_labels = [f"Module {c.replace('ME', '')}" for c in cor_matrix.index]

    sns.heatmap(
        cor_matrix.astype(float),
        annot=annot, fmt="",
        cmap="RdBu_r", vmin=-1, vmax=1, center=0,
        xticklabels=traits.columns,
        yticklabels=row_labels,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Correlation", "shrink": 0.7},
        annot_kws={"fontsize": 9},
    )
    ax.set_title("Module-Trait Relationships", fontsize=14, pad=15)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "wgcna_module_trait_heatmap.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: figures/wgcna_module_trait_heatmap.png")

    return cor_matrix, pval_matrix, traits


# ──────────────────────────────────────────────
# Step 5: Hub gene identification
# ──────────────────────────────────────────────
def identify_hub_genes(expr, module_df, eigengene_df, cor_matrix, gene_mapping,
                        top_n_per_module=20):
    print("\n" + "=" * 60)
    print("Step 6: Identifying hub genes...")
    print("=" * 60)

    # Load DEG results for overlap
    deg_df = None
    deg_path = RESULTS_DIR / "deg_significant.csv"
    if deg_path.exists():
        deg_df = pd.read_csv(deg_path, index_col=0)

    all_hubs = []
    modules = sorted(module_df["module"].unique())

    for mod in modules:
        if mod == 0:
            continue

        gene_mask = module_df["module"] == mod
        gene_ids = module_df.loc[gene_mask, "gene_id"].values
        me_col = f"ME{mod}"

        if me_col not in eigengene_df.columns:
            continue

        me = eigengene_df[me_col]

        # Module Membership (MM): correlation of gene expression with module eigengene
        mm_values = []
        gs_values = []

        for gid in gene_ids:
            if gid in expr.columns:
                gene_expr = expr[gid]

                # Module membership
                valid = gene_expr.notna() & me.notna()
                if valid.sum() > 10:
                    r_mm, p_mm = pearsonr(gene_expr[valid], me[valid])
                else:
                    r_mm, p_mm = 0, 1

                mm_values.append({"gene_id": gid, "MM": abs(r_mm), "MM_pval": p_mm})

        mm_df = pd.DataFrame(mm_values)

        if mm_df.empty:
            continue

        # Gene Significance (GS): correlation with tumor status
        # Use Tumor_status as the primary trait
        common_samples = expr.index.intersection(eigengene_df.index)
        # We need the metadata
        meta = pd.read_csv(DATA_DIR / "metadata.csv").set_index("sample_submitter_id")
        tumor_status = (meta.loc[common_samples, "condition"] == "Tumor").astype(int)

        for i, row in mm_df.iterrows():
            gid = row["gene_id"]
            if gid in expr.columns:
                gene_expr = expr.loc[common_samples, gid]
                valid = gene_expr.notna() & tumor_status.notna()
                if valid.sum() > 10:
                    r_gs, p_gs = pearsonr(gene_expr[valid], tumor_status[valid])
                    mm_df.loc[i, "GS"] = abs(r_gs)
                    mm_df.loc[i, "GS_pval"] = p_gs
                else:
                    mm_df.loc[i, "GS"] = 0
                    mm_df.loc[i, "GS_pval"] = 1

        # Hub genes: high MM AND high GS
        mm_df["module"] = mod
        mm_df = mm_df.sort_values("MM", ascending=False)

        # Add gene names
        if gene_mapping is not None:
            name_map = gene_mapping["gene_name"].to_dict()
            mm_df["gene_name"] = mm_df["gene_id"].map(name_map)

        # Check if DEG
        if deg_df is not None:
            deg_genes = set(deg_df.index)
            mm_df["is_DEG"] = mm_df["gene_id"].isin(deg_genes)

        top_hubs = mm_df.head(top_n_per_module)
        all_hubs.append(top_hubs)

        n_hub = min(5, len(top_hubs))
        print(f"\n  Module {mod} — top {n_hub} hub genes:")
        for _, row in top_hubs.head(n_hub).iterrows():
            name = row.get("gene_name", row["gene_id"])
            gs = row.get("GS", 0)
            deg = " [DEG]" if row.get("is_DEG", False) else ""
            print(f"    {str(name):15s}  MM={row['MM']:.3f}  GS={gs:.3f}{deg}")

    hub_df = pd.concat(all_hubs, ignore_index=True) if all_hubs else pd.DataFrame()
    return hub_df


# ──────────────────────────────────────────────
# Step 6: Module size barplot
# ──────────────────────────────────────────────
def plot_module_sizes(module_df, module_colors):
    print("\n  Plotting module sizes...")

    counts = module_df["module"].value_counts().sort_index()
    labels = [f"M{m}" if m > 0 else "Unassigned" for m in counts.index]
    colors = [module_colors.get(m, "#cccccc") for m in counts.index]

    fig, ax = plt.subplots(figsize=(max(8, len(counts) * 0.6), 5))
    bars = ax.bar(labels, counts.values, color=colors, edgecolor="white")

    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                str(val), ha="center", fontsize=8)

    ax.set_xlabel("Module")
    ax.set_ylabel("Number of Genes")
    ax.set_title("WGCNA Module Sizes")
    ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "wgcna_module_sizes.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: figures/wgcna_module_sizes.png")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("  TCGA-LUAD Weighted Gene Co-expression Network Analysis")
    print("=" * 60 + "\n")

    # Step 1: Prepare data
    expr, metadata, gene_mapping = prepare_expression()

    # Step 2: Pick soft threshold
    power, cor_matrix = pick_soft_threshold(expr)

    # Step 3: Build network and detect modules
    gene_ids = expr.columns.tolist()
    module_df, labels, module_colors, tom, Z = build_network_and_detect_modules(
        cor_matrix, power, gene_ids
    )

    # Step 4: Module eigengenes
    eigengene_df = compute_module_eigengenes(expr, module_df)

    # Step 5: Module-trait correlation
    trait_cor, trait_pval, traits = correlate_with_traits(eigengene_df, metadata, expr)

    # Step 6: Hub genes
    hub_df = identify_hub_genes(expr, module_df, eigengene_df, trait_cor, gene_mapping)

    # Plot module sizes
    plot_module_sizes(module_df, module_colors)

    # ── Save results ──
    print("\n" + "=" * 60)
    print("Saving results...")
    print("=" * 60)

    # Add gene names to module_df
    if gene_mapping is not None:
        name_map = gene_mapping["gene_name"].to_dict()
        module_df["gene_name"] = module_df["gene_id"].map(name_map)

    module_df.to_csv(RESULTS_DIR / "wgcna_modules.csv", index=False)
    print(f"  Saved: results/wgcna_modules.csv ({len(module_df)} genes)")

    trait_cor.to_csv(RESULTS_DIR / "wgcna_module_trait_cor.csv")
    print(f"  Saved: results/wgcna_module_trait_cor.csv")

    trait_pval.to_csv(RESULTS_DIR / "wgcna_module_trait_pval.csv")

    if not hub_df.empty:
        hub_df.to_csv(RESULTS_DIR / "wgcna_hub_genes.csv", index=False)
        print(f"  Saved: results/wgcna_hub_genes.csv ({len(hub_df)} hub genes)")

    eigengene_df.to_csv(RESULTS_DIR / "wgcna_eigengenes.csv")
    print(f"  Saved: results/wgcna_eigengenes.csv")

    print(f"\n" + "=" * 60)
    print("  Phase 5 complete!")
    print("  Next: python scripts/06a_ppi_network.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
