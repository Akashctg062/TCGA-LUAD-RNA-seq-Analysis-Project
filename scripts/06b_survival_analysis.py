"""
06b_survival_analysis.py
=========================
Survival analysis for hub genes: Kaplan-Meier curves,
log-rank test, univariate and multivariate Cox regression.

Usage:
    python scripts/06b_survival_analysis.py

Output:
    results/survival_km_results.csv
    results/survival_cox_univariate.csv
    results/survival_cox_multivariate.csv
    figures/km_curves_hub_genes.png
    figures/cox_forest_plot.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import warnings
warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
FIG_DIR = PROJECT_DIR / "figures"

TOP_N_KM = 12  # Number of hub genes to plot KM curves for

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.size": 10, "axes.titlesize": 11,
})


def load_data():
    print("=" * 60)
    print("Step 1: Loading expression and clinical data...")
    print("=" * 60)

    counts = pd.read_csv(DATA_DIR / "filtered_counts.csv", index_col=0)
    metadata = pd.read_csv(DATA_DIR / "metadata.csv")
    gene_mapping = None
    if (DATA_DIR / "gene_id_mapping.csv").exists():
        gene_mapping = pd.read_csv(DATA_DIR / "gene_id_mapping.csv", index_col=0)

    # Load hub genes
    hub_path = RESULTS_DIR / "final_hub_genes.csv"
    if hub_path.exists():
        hub_df = pd.read_csv(hub_path)
    else:
        hub_df = pd.read_csv(RESULTS_DIR / "wgcna_hub_genes.csv")

    # Normalize expression
    lib_sizes = counts.sum(axis=0)
    cpm = counts.div(lib_sizes, axis=1) * 1e6
    log_cpm = np.log2(cpm + 1)

    # Filter to tumor samples only (survival analysis on tumor patients)
    meta = metadata.set_index("sample_submitter_id")
    tumor_samples = meta[meta["condition"] == "Tumor"].index
    tumor_samples = [s for s in tumor_samples if s in log_cpm.columns]

    expr = log_cpm[tumor_samples].T  # samples × genes
    meta_tumor = meta.loc[tumor_samples].copy()

    # Prepare survival data
    meta_tumor["os_time"] = pd.to_numeric(meta_tumor["os_time_days"], errors="coerce")
    meta_tumor["os_event"] = pd.to_numeric(meta_tumor["os_event"], errors="coerce")

    # Drop samples without survival data
    valid = meta_tumor["os_time"].notna() & meta_tumor["os_event"].notna() & (meta_tumor["os_time"] > 0)
    meta_tumor = meta_tumor[valid]
    expr = expr.loc[meta_tumor.index]

    print(f"  Tumor samples with survival data: {len(meta_tumor)}")
    print(f"  Events (deaths): {int(meta_tumor['os_event'].sum())}")
    print(f"  Median follow-up: {meta_tumor['os_time'].median():.0f} days")
    print(f"  Hub genes to test: {len(hub_df)}")

    return expr, meta_tumor, hub_df, gene_mapping


def run_km_analysis(expr, meta_tumor, hub_df, gene_mapping):
    """Run Kaplan-Meier analysis for each hub gene."""
    print("\n" + "=" * 60)
    print("Step 2: Kaplan-Meier survival analysis...")
    print("=" * 60)

    # Map gene names to IDs
    if gene_mapping is not None:
        name_to_id = dict(zip(gene_mapping["gene_name"], gene_mapping.index))
    else:
        name_to_id = {}

    km_results = []

    for _, row in hub_df.iterrows():
        gene_name = row.get("gene_name", "")
        gene_id = row.get("gene_id", "")

        # Find gene in expression matrix
        if gene_id in expr.columns:
            gene_expr = expr[gene_id]
        elif gene_name in name_to_id and name_to_id[gene_name] in expr.columns:
            gene_expr = expr[name_to_id[gene_name]]
        else:
            continue

        # Split by median expression
        median_val = gene_expr.median()
        high = gene_expr >= median_val
        low = gene_expr < median_val

        # Ensure both groups have events
        t = meta_tumor["os_time"]
        e = meta_tumor["os_event"]

        if high.sum() < 5 or low.sum() < 5:
            continue

        # Log-rank test
        result = logrank_test(
            t[high], t[low],
            event_observed_A=e[high], event_observed_B=e[low]
        )

        # Median survival for each group
        kmf_high = KaplanMeierFitter()
        kmf_high.fit(t[high], e[high])
        median_high = kmf_high.median_survival_time_

        kmf_low = KaplanMeierFitter()
        kmf_low.fit(t[low], e[low])
        median_low = kmf_low.median_survival_time_

        km_results.append({
            "gene_name": gene_name,
            "gene_id": gene_id,
            "logrank_p": result.p_value,
            "logrank_stat": result.test_statistic,
            "median_os_high": median_high,
            "median_os_low": median_low,
            "n_high": int(high.sum()),
            "n_low": int(low.sum()),
            "module": row.get("module", ""),
        })

    km_df = pd.DataFrame(km_results).sort_values("logrank_p")

    # BH correction
    from statsmodels.stats.multitest import multipletests
    if len(km_df) > 0:
        _, km_df["logrank_padj"], _, _ = multipletests(km_df["logrank_p"], method="fdr_bh")

    sig = km_df[km_df["logrank_p"] < 0.05]
    print(f"\n  Genes tested: {len(km_df)}")
    print(f"  Significant (p < 0.05): {len(sig)}")
    print(f"  Significant (padj < 0.05): {(km_df['logrank_padj'] < 0.05).sum()}")

    print(f"\n  Top prognostic genes:")
    for _, row in km_df.head(10).iterrows():
        print(f"    {row['gene_name']:15s}  p={row['logrank_p']:.4e}  "
              f"padj={row['logrank_padj']:.4e}")

    return km_df


def plot_km_curves(expr, meta_tumor, km_df, gene_mapping):
    """Plot KM curves for top prognostic hub genes."""
    print("\n  Plotting Kaplan-Meier curves...")

    # Select top genes
    top_genes = km_df.head(TOP_N_KM)

    if gene_mapping is not None:
        name_to_id = dict(zip(gene_mapping["gene_name"], gene_mapping.index))
    else:
        name_to_id = {}

    n_genes = len(top_genes)
    n_cols = 4
    n_rows = (n_genes + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    t = meta_tumor["os_time"] / 365.25  # Convert to years
    e = meta_tumor["os_event"]

    for i, (_, row) in enumerate(top_genes.iterrows()):
        ax = axes[i]
        gene_name = row["gene_name"]
        gene_id = row["gene_id"]

        if gene_id in expr.columns:
            gene_expr = expr[gene_id]
        elif gene_name in name_to_id and name_to_id[gene_name] in expr.columns:
            gene_expr = expr[name_to_id[gene_name]]
        else:
            ax.set_visible(False)
            continue

        median_val = gene_expr.median()
        high = gene_expr >= median_val
        low = gene_expr < median_val

        kmf = KaplanMeierFitter()

        kmf.fit(t[high], e[high], label="High expression")
        kmf.plot_survival_function(ax=ax, color="#e74c3c", ci_show=True, linewidth=1.5)

        kmf.fit(t[low], e[low], label="Low expression")
        kmf.plot_survival_function(ax=ax, color="#3498db", ci_show=True, linewidth=1.5)

        p_val = row["logrank_p"]
        p_text = f"p = {p_val:.2e}" if p_val < 0.001 else f"p = {p_val:.4f}"

        ax.set_title(f"{gene_name}\n{p_text}", fontsize=10,
                     fontweight="bold" if p_val < 0.05 else "normal")
        ax.set_xlabel("Time (years)")
        ax.set_ylabel("Survival Probability")
        ax.legend(fontsize=7, loc="lower left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, 1.05)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Kaplan-Meier Survival Curves for Hub Genes (TCGA-LUAD)",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "km_curves_hub_genes.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: figures/km_curves_hub_genes.png")


def run_cox_regression(expr, meta_tumor, km_df, gene_mapping):
    """Univariate and multivariate Cox regression."""
    print("\n" + "=" * 60)
    print("Step 3: Cox proportional hazards regression...")
    print("=" * 60)

    if gene_mapping is not None:
        name_to_id = dict(zip(gene_mapping["gene_name"], gene_mapping.index))
    else:
        name_to_id = {}

    # Prepare clinical covariates
    meta = meta_tumor.copy()
    meta["os_time_years"] = meta["os_time"] / 365.25

    # Stage numeric
    stage_map = {}
    for s in meta["tumor_stage"].dropna().unique():
        s_str = str(s).upper()
        if "IV" in s_str: stage_map[s] = 4
        elif "III" in s_str: stage_map[s] = 3
        elif "II" in s_str: stage_map[s] = 2
        elif "I" in s_str: stage_map[s] = 1
    meta["stage_num"] = meta["tumor_stage"].map(stage_map)
    meta["age_num"] = pd.to_numeric(meta["age_years"], errors="coerce")

    # Significant genes for Cox analysis
    sig_genes = km_df[km_df["logrank_p"] < 0.05].head(20)

    # Univariate Cox
    uni_results = []
    for _, row in sig_genes.iterrows():
        gene_name = row["gene_name"]
        gene_id = row["gene_id"]

        if gene_id in expr.columns:
            gene_expr = expr[gene_id]
        elif gene_name in name_to_id and name_to_id[gene_name] in expr.columns:
            gene_expr = expr[name_to_id[gene_name]]
        else:
            continue

        cox_data = pd.DataFrame({
            "T": meta["os_time_years"],
            "E": meta["os_event"],
            "gene": gene_expr,
        }).dropna()

        if len(cox_data) < 30:
            continue

        try:
            cph = CoxPHFitter()
            cph.fit(cox_data, duration_col="T", event_col="E")
            summary = cph.summary

            uni_results.append({
                "gene_name": gene_name,
                "gene_id": gene_id,
                "HR": np.exp(summary.loc["gene", "coef"]),
                "HR_lower": np.exp(summary.loc["gene", "coef lower 95%"]),
                "HR_upper": np.exp(summary.loc["gene", "coef upper 95%"]),
                "coef": summary.loc["gene", "coef"],
                "p_value": summary.loc["gene", "p"],
                "concordance": cph.concordance_index_,
            })
        except Exception as e:
            continue

    uni_df = pd.DataFrame(uni_results).sort_values("p_value")

    print(f"\n  Univariate Cox results (significant genes):")
    for _, row in uni_df.head(10).iterrows():
        risk = "Risk" if row["HR"] > 1 else "Protective"
        print(f"    {row['gene_name']:15s}  HR={row['HR']:.3f} "
              f"({row['HR_lower']:.2f}-{row['HR_upper']:.2f})  "
              f"p={row['p_value']:.4e}  [{risk}]")

    # Multivariate Cox (top genes + clinical variables)
    multi_results = []
    top_uni = uni_df[uni_df["p_value"] < 0.05].head(10)

    for _, row in top_uni.iterrows():
        gene_name = row["gene_name"]
        gene_id = row["gene_id"]

        if gene_id in expr.columns:
            gene_expr = expr[gene_id]
        elif gene_name in name_to_id and name_to_id[gene_name] in expr.columns:
            gene_expr = expr[name_to_id[gene_name]]
        else:
            continue

        cox_data = pd.DataFrame({
            "T": meta["os_time_years"],
            "E": meta["os_event"],
            "gene": gene_expr,
            "stage": meta["stage_num"],
            "age": meta["age_num"],
        }).dropna()

        if len(cox_data) < 30:
            continue

        try:
            cph = CoxPHFitter()
            cph.fit(cox_data, duration_col="T", event_col="E")
            s = cph.summary

            multi_results.append({
                "gene_name": gene_name,
                "HR": np.exp(s.loc["gene", "coef"]),
                "HR_lower": np.exp(s.loc["gene", "coef lower 95%"]),
                "HR_upper": np.exp(s.loc["gene", "coef upper 95%"]),
                "p_value": s.loc["gene", "p"],
                "stage_p": s.loc["stage", "p"] if "stage" in s.index else np.nan,
                "age_p": s.loc["age", "p"] if "age" in s.index else np.nan,
                "concordance": cph.concordance_index_,
            })
        except Exception:
            continue

    multi_df = pd.DataFrame(multi_results).sort_values("p_value")

    print(f"\n  Multivariate Cox (gene + stage + age):")
    for _, row in multi_df.head(10).iterrows():
        sig = "*" if row["p_value"] < 0.05 else ""
        print(f"    {row['gene_name']:15s}  HR={row['HR']:.3f} "
              f"({row['HR_lower']:.2f}-{row['HR_upper']:.2f})  "
              f"p={row['p_value']:.4e} {sig}")

    return uni_df, multi_df


def plot_forest_plot(uni_df, multi_df):
    """Forest plot of Cox regression hazard ratios."""
    print("\n  Plotting forest plot...")

    # Use multivariate results if available, else univariate
    plot_df = multi_df if not multi_df.empty else uni_df
    plot_df = plot_df.head(15).sort_values("HR", ascending=True).copy()

    if plot_df.empty:
        print("    Skipped: no significant Cox results")
        return

    fig, ax = plt.subplots(figsize=(10, max(5, len(plot_df) * 0.5)))

    y_positions = range(len(plot_df))

    # Plot HR points with CI
    for i, (_, row) in enumerate(plot_df.iterrows()):
        color = "#e74c3c" if row["HR"] > 1 else "#3498db"

        ax.plot([row["HR_lower"], row["HR_upper"]], [i, i],
                color=color, linewidth=2, alpha=0.7)
        ax.scatter(row["HR"], i, color=color, s=80, zorder=5, edgecolors="white", linewidth=1)

    # Reference line at HR=1
    ax.axvline(1, color="black", linestyle="--", linewidth=0.8)

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(plot_df["gene_name"], fontsize=10)
    ax.set_xlabel("Hazard Ratio (95% CI)")
    title_type = "Multivariate" if not multi_df.empty else "Univariate"
    ax.set_title(f"{title_type} Cox Regression — Hub Genes (TCGA-LUAD)")
    ax.spines[["top", "right"]].set_visible(False)

    # Add p-value annotations
    for i, (_, row) in enumerate(plot_df.iterrows()):
        p = row["p_value"]
        p_text = f"p={p:.2e}" if p < 0.001 else f"p={p:.3f}"
        ax.text(ax.get_xlim()[1] * 0.98, i, p_text, va="center", ha="right", fontsize=8)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "cox_forest_plot.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: figures/cox_forest_plot.png")


def main():
    print("\n" + "=" * 60)
    print("  Phase 6B: Survival Analysis")
    print("=" * 60 + "\n")

    expr, meta_tumor, hub_df, gene_mapping = load_data()

    # Kaplan-Meier
    km_df = run_km_analysis(expr, meta_tumor, hub_df, gene_mapping)
    km_df.to_csv(RESULTS_DIR / "survival_km_results.csv", index=False)

    plot_km_curves(expr, meta_tumor, km_df, gene_mapping)

    # Cox regression
    uni_df, multi_df = run_cox_regression(expr, meta_tumor, km_df, gene_mapping)
    uni_df.to_csv(RESULTS_DIR / "survival_cox_univariate.csv", index=False)
    multi_df.to_csv(RESULTS_DIR / "survival_cox_multivariate.csv", index=False)

    plot_forest_plot(uni_df, multi_df)

    print(f"\n  Phase 6B complete!")


if __name__ == "__main__":
    main()
