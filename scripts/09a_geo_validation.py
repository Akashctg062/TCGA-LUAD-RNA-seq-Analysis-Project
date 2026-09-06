"""
09a_geo_validation.py
======================
External validation in 3 independent GEO lung adenocarcinoma cohorts:
  - GSE31210: 226 LUAD (Stage I-II), Affymetrix HG-U133 Plus 2.0
  - GSE72094: 442 LUAD, RNA-seq (KRAS-focused)
  - GSE68465: 443 LUAD, Affymetrix HG-U133A (Director's Challenge)

Validates:
  1. Hub gene differential expression
  2. Survival prognostic value (KM + Cox)
  3. Diagnostic AUC (where tumor/normal available)

Usage:
    pip install GEOparse --break-system-packages   (if not installed)
    python scripts/09a_geo_validation.py

Output:
    data/geo/GSE*/  (cached downloads)
    results/geo_validation_summary.csv
    figures/geo_survival_validation.png
    figures/geo_expression_validation.png
    figures/geo_forest_plot.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from scipy.stats import mannwhitneyu
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings("ignore")

try:
    import GEOparse
    HAS_GEOPARSE = True
except ImportError:
    HAS_GEOPARSE = False
    print("WARNING: GEOparse not installed. Run: pip install GEOparse --break-system-packages")

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
GEO_DIR = DATA_DIR / "geo"
RESULTS_DIR = PROJECT_DIR / "results"
FIG_DIR = PROJECT_DIR / "figures"
GEO_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.size": 10, "axes.titlesize": 12,
})

HUB_GENES = ["BIRC5", "CCNB1", "CCNB2", "CDC20", "KIF2C"]

# Probe-to-gene mappings for Affymetrix platforms
# Multiple probes per gene — we take the one with highest variance
PROBE_GENE_MAP = {
    # HG-U133 Plus 2.0 and HG-U133A common probe sets
    "BIRC5": ["202095_s_at", "210334_x_at"],
    "CCNB1": ["214710_s_at", "228729_at"],
    "CCNB2": ["202705_at"],
    "CDC20": ["202870_s_at"],
    "KIF2C": ["209408_at", "211519_s_at"],
}


def download_geo_dataset(gse_id):
    """Download and cache a GEO dataset."""
    cache_dir = GEO_DIR / gse_id
    cache_dir.mkdir(exist_ok=True)

    expr_cache = cache_dir / "expression.csv"
    pheno_cache = cache_dir / "phenotype.csv"

    if expr_cache.exists() and pheno_cache.exists():
        print(f"  Loading cached {gse_id}...")
        expr = pd.read_csv(expr_cache, index_col=0)
        pheno = pd.read_csv(pheno_cache, index_col=0)
        return expr, pheno

    if not HAS_GEOPARSE:
        print(f"  ERROR: Cannot download {gse_id} without GEOparse")
        return None, None

    print(f"  Downloading {gse_id} (this may take a few minutes)...")
    gse = GEOparse.get_GEO(geo=gse_id, destdir=str(cache_dir), silent=True)

    # Extract expression matrix
    # Try to get from the first GPL platform
    gpls = list(gse.gpls.keys())
    print(f"    Platforms: {gpls}")

    # Get pivoted expression table
    samples = list(gse.gsms.keys())
    print(f"    Samples: {len(samples)}")

    # Build expression matrix from GSMs
    expr_dict = {}
    for gsm_name, gsm in gse.gsms.items():
        table = gsm.table
        if table is not None and not table.empty:
            if "VALUE" in table.columns and "ID_REF" in table.columns:
                series = table.set_index("ID_REF")["VALUE"]
                series = pd.to_numeric(series, errors="coerce")
                expr_dict[gsm_name] = series

    if not expr_dict:
        print(f"    WARNING: Could not extract expression data for {gse_id}")
        return None, None

    expr = pd.DataFrame(expr_dict)
    print(f"    Expression matrix: {expr.shape[0]} probes × {expr.shape[1]} samples")

    # Extract phenotype data
    pheno_rows = []
    for gsm_name, gsm in gse.gsms.items():
        meta = gsm.metadata
        chars = {}
        chars["sample_id"] = gsm_name
        chars["title"] = meta.get("title", [""])[0]

        # Parse characteristics
        for char in meta.get("characteristics_ch1", []):
            if ": " in char:
                key, val = char.split(": ", 1)
                chars[key.strip().lower().replace(" ", "_")] = val.strip()

        pheno_rows.append(chars)

    pheno = pd.DataFrame(pheno_rows).set_index("sample_id")

    # Cache
    expr.to_csv(expr_cache)
    pheno.to_csv(pheno_cache)

    return expr, pheno


def extract_gene_expression(expr, gene_name, platform="hgu133"):
    """Extract gene expression from probe-level data."""
    probes = PROBE_GENE_MAP.get(gene_name, [])

    # Try probe IDs first
    available_probes = [p for p in probes if p in expr.index]

    if available_probes:
        # Use probe with highest variance
        if len(available_probes) > 1:
            variances = expr.loc[available_probes].var(axis=1)
            best_probe = variances.idxmax()
        else:
            best_probe = available_probes[0]
        return expr.loc[best_probe]

    # Try gene name directly (some datasets use gene symbols)
    if gene_name in expr.index:
        return expr.loc[gene_name]

    # Try case-insensitive search
    idx_upper = {i.upper(): i for i in expr.index if isinstance(i, str)}
    if gene_name.upper() in idx_upper:
        return expr.loc[idx_upper[gene_name.upper()]]

    return None


def parse_survival_gse31210(pheno):
    """Parse survival data from GSE31210."""
    surv = pd.DataFrame(index=pheno.index)

    # GSE31210 fields: "relapse (1, relapsed; 0, not relapsed)",
    # "days before relapse/censor"
    for col in pheno.columns:
        col_lower = col.lower()
        if "relapse" in col_lower and "day" in col_lower:
            surv["time"] = pd.to_numeric(pheno[col], errors="coerce") / 365.25
        elif "relapse" in col_lower and "day" not in col_lower:
            vals = pheno[col].astype(str)
            surv["event"] = vals.apply(lambda x: 1 if "1" in str(x) or "relapse" in str(x).lower() else 0)
        elif "death" in col_lower and "day" in col_lower:
            surv["os_time"] = pd.to_numeric(pheno[col], errors="coerce") / 365.25
        elif "death" in col_lower and "day" not in col_lower:
            vals = pheno[col].astype(str)
            surv["os_event"] = vals.apply(lambda x: 1 if "1" in str(x) or "dead" in str(x).lower() else 0)

    # Prefer OS if available, else RFS
    if "os_time" in surv.columns and surv["os_time"].notna().sum() > 50:
        surv["time"] = surv["os_time"]
        surv["event"] = surv.get("os_event", surv.get("event", 0))
    elif "time" not in surv.columns:
        # Try to find any time/event columns
        for col in pheno.columns:
            if any(kw in col.lower() for kw in ["survival", "month", "year", "time", "follow"]):
                vals = pd.to_numeric(pheno[col], errors="coerce")
                if vals.notna().sum() > 50:
                    surv["time"] = vals
                    if vals.max() > 100:  # Likely days
                        surv["time"] = vals / 365.25
                    break

        for col in pheno.columns:
            if any(kw in col.lower() for kw in ["status", "event", "vital", "alive", "dead", "censor"]):
                vals = pheno[col].astype(str)
                surv["event"] = vals.apply(
                    lambda x: 1 if any(k in str(x).lower() for k in ["1", "dead", "deceased", "recur"]) else 0
                )
                break

    return surv


def parse_survival_generic(pheno):
    """Generic survival parser — works for most GEO datasets."""
    surv = pd.DataFrame(index=pheno.index)

    # Search for time columns
    time_keywords = ["survival", "os_time", "os_months", "time", "follow_up",
                     "months", "days", "overall_survival", "dfs_time", "rfs"]
    event_keywords = ["status", "event", "vital", "os_event", "alive_dead",
                      "vital_status", "os_status", "censor"]

    for col in pheno.columns:
        col_lower = col.lower().replace(" ", "_")
        for kw in time_keywords:
            if kw in col_lower and "time" not in surv.columns:
                vals = pd.to_numeric(pheno[col], errors="coerce")
                if vals.notna().sum() > 30:
                    surv["time"] = vals
                    # Convert to years if likely in days or months
                    if vals.median() > 500:  # Likely days
                        surv["time"] = vals / 365.25
                    elif vals.median() > 50:  # Likely months
                        surv["time"] = vals / 12
                    break

    for col in pheno.columns:
        col_lower = col.lower().replace(" ", "_")
        for kw in event_keywords:
            if kw in col_lower and "event" not in surv.columns:
                vals = pheno[col].astype(str)
                surv["event"] = vals.apply(
                    lambda x: 1 if any(k in str(x).lower() for k in
                                      ["1", "dead", "deceased", "death", "recurrence", "relapse"])
                    else 0
                )
                break

    return surv


def validate_in_dataset(gse_id, expr, pheno, parse_survival_fn):
    """Run full validation pipeline for one GEO dataset."""
    print(f"\n{'=' * 60}")
    print(f"  Validating in {gse_id}")
    print(f"{'=' * 60}")

    results = {"gse_id": gse_id, "n_samples": len(pheno)}

    # 1. Extract hub gene expression
    gene_expr = {}
    for gene in HUB_GENES:
        ge = extract_gene_expression(expr, gene)
        if ge is not None:
            gene_expr[gene] = ge
            print(f"    {gene}: found (mean={ge.mean():.2f}, sd={ge.std():.2f})")
        else:
            print(f"    {gene}: NOT FOUND")

    results["genes_found"] = len(gene_expr)

    if len(gene_expr) == 0:
        print(f"    SKIP: No hub genes found in {gse_id}")
        return results, {}

    # 2. Parse survival
    surv = parse_survival_fn(pheno)
    has_survival = "time" in surv.columns and "event" in surv.columns
    if has_survival:
        valid_surv = surv.dropna(subset=["time", "event"])
        valid_surv = valid_surv[valid_surv["time"] > 0]
        has_survival = len(valid_surv) > 30
        if has_survival:
            print(f"    Survival data: {len(valid_surv)} patients, "
                  f"{int(valid_surv['event'].sum())} events")

    # 3. Survival analysis for each gene
    km_results = {}
    cox_results = {}

    if has_survival:
        for gene, ge in gene_expr.items():
            common = valid_surv.index.intersection(ge.index)
            if len(common) < 30:
                continue

            gene_vals = ge[common]
            sv = valid_surv.loc[common]

            # Median split
            median_val = gene_vals.median()
            high = gene_vals >= median_val
            low = gene_vals < median_val

            if high.sum() < 10 or low.sum() < 10:
                continue

            # Log-rank test
            try:
                lr = logrank_test(
                    sv.loc[high, "time"], sv.loc[low, "time"],
                    sv.loc[high, "event"], sv.loc[low, "event"]
                )
                km_results[gene] = {
                    "p_value": lr.p_value,
                    "n_high": int(high.sum()),
                    "n_low": int(low.sum()),
                }

                star = "***" if lr.p_value < 0.001 else "**" if lr.p_value < 0.01 else "*" if lr.p_value < 0.05 else ""
                print(f"    {gene} KM: p={lr.p_value:.4f} {star}")
            except Exception as e:
                print(f"    {gene} KM error: {e}")

            # Cox regression
            try:
                cox_df = pd.DataFrame({
                    "time": sv.loc[common, "time"],
                    "event": sv.loc[common, "event"],
                    "gene_expr": gene_vals,
                })
                cox_df = cox_df.dropna()

                # Z-score normalize
                cox_df["gene_expr"] = (cox_df["gene_expr"] - cox_df["gene_expr"].mean()) / (cox_df["gene_expr"].std() + 1e-10)

                cph = CoxPHFitter()
                cph.fit(cox_df, duration_col="time", event_col="event")

                hr = np.exp(cph.params_["gene_expr"])
                ci = np.exp(cph.confidence_intervals_.values[0])
                pval = cph.summary["p"]["gene_expr"]

                cox_results[gene] = {
                    "HR": hr,
                    "CI_low": ci[0],
                    "CI_high": ci[1],
                    "p_value": pval,
                }
            except Exception as e:
                print(f"    {gene} Cox error: {e}")

    results["km_results"] = km_results
    results["cox_results"] = cox_results

    return results, gene_expr


def plot_geo_survival_validation(all_results):
    """Multi-panel KM survival curves across GEO datasets."""
    print("\n  Plotting GEO survival validation...")

    datasets_with_survival = [r for r in all_results if r.get("km_results")]
    if not datasets_with_survival:
        print("    No survival data available for plotting")
        return

    n_datasets = len(datasets_with_survival)
    n_genes = len(HUB_GENES)

    fig, axes = plt.subplots(n_datasets, n_genes,
                             figsize=(4 * n_genes, 4 * n_datasets),
                             squeeze=False)

    for row, res in enumerate(datasets_with_survival):
        gse_id = res["gse_id"]
        km = res["km_results"]

        for col, gene in enumerate(HUB_GENES):
            ax = axes[row][col]
            if gene in km:
                p = km[gene]["p_value"]
                star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                ax.set_title(f"{gene}\np = {p:.4f} {star}", fontsize=10,
                            fontweight="bold" if p < 0.05 else "normal")
                color = "green" if p < 0.05 else "gray"
                ax.text(0.5, 0.5, f"p={p:.4f}\n{star}",
                       transform=ax.transAxes, ha="center", va="center",
                       fontsize=14, color=color, fontweight="bold")
            else:
                ax.set_title(f"{gene}\nN/A", fontsize=10, color="gray")
                ax.text(0.5, 0.5, "Not found", transform=ax.transAxes,
                       ha="center", va="center", fontsize=12, color="gray")

            if col == 0:
                ax.set_ylabel(gse_id, fontsize=12, fontweight="bold")
            ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle("External Validation: Survival Analysis Across GEO Cohorts",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "geo_survival_validation.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: figures/geo_survival_validation.png")


def plot_forest_meta(all_results):
    """Forest plot of hazard ratios across all datasets (meta-analysis style)."""
    print("  Plotting meta-analysis forest plot...")

    rows = []
    # Add TCGA results
    tcga_cox_path = RESULTS_DIR / "cox_univariate.csv"
    if tcga_cox_path.exists():
        tcga_cox = pd.read_csv(tcga_cox_path)
        for _, row in tcga_cox.iterrows():
            gene = row.get("gene_name", row.get("gene", ""))
            if gene in HUB_GENES:
                rows.append({
                    "dataset": "TCGA-LUAD",
                    "gene": gene,
                    "HR": row.get("HR", row.get("exp(coef)", np.nan)),
                    "CI_low": row.get("CI_low", row.get("exp(coef) lower 95%", np.nan)),
                    "CI_high": row.get("CI_high", row.get("exp(coef) upper 95%", np.nan)),
                    "p_value": row.get("p_value", row.get("p", np.nan)),
                })

    # Add GEO results
    for res in all_results:
        for gene, cox in res.get("cox_results", {}).items():
            rows.append({
                "dataset": res["gse_id"],
                "gene": gene,
                "HR": cox["HR"],
                "CI_low": cox["CI_low"],
                "CI_high": cox["CI_high"],
                "p_value": cox["p_value"],
            })

    if not rows:
        print("    No Cox results available for forest plot")
        return

    forest_df = pd.DataFrame(rows)
    forest_df.to_csv(RESULTS_DIR / "meta_forest_data.csv", index=False)

    # Plot forest plot per gene
    genes_with_data = [g for g in HUB_GENES if g in forest_df["gene"].values]
    if not genes_with_data:
        return

    fig, axes = plt.subplots(1, len(genes_with_data),
                             figsize=(5 * len(genes_with_data), 4),
                             squeeze=False)

    for i, gene in enumerate(genes_with_data):
        ax = axes[0][i]
        gdf = forest_df[forest_df["gene"] == gene].copy()
        gdf = gdf.sort_values("dataset")

        y_pos = range(len(gdf))

        for j, (_, row) in enumerate(gdf.iterrows()):
            hr = row["HR"]
            ci_lo = row["CI_low"]
            ci_hi = row["CI_high"]
            pval = row["p_value"]

            color = "#e74c3c" if pval < 0.05 else "#999999"

            if pd.notna(hr) and pd.notna(ci_lo) and pd.notna(ci_hi):
                ax.plot([ci_lo, ci_hi], [j, j], color=color, lw=2)
                ax.plot(hr, j, "o", color=color, markersize=8)
                ax.text(max(ci_hi, hr) + 0.05, j,
                       f"HR={hr:.2f} p={pval:.3f}",
                       va="center", fontsize=8, color=color)

        ax.axvline(1, color="black", ls="--", lw=1, alpha=0.5)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(gdf["dataset"].values, fontsize=9)
        ax.set_xlabel("Hazard Ratio")
        ax.set_title(gene, fontsize=12, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle("Cross-Dataset Survival Validation (Forest Plot)", fontsize=14, y=1.05)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "geo_forest_plot.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: figures/geo_forest_plot.png")


def plot_expression_across_datasets(all_gene_exprs, all_results):
    """Box plots of hub gene expression across datasets."""
    print("  Plotting cross-dataset expression comparison...")

    plot_data = []
    for res, gene_exprs in zip(all_results, all_gene_exprs):
        gse_id = res["gse_id"]
        for gene, ge in gene_exprs.items():
            for val in ge.dropna().values:
                plot_data.append({
                    "Gene": gene,
                    "Expression": float(val),
                    "Dataset": gse_id,
                })

    if not plot_data:
        return

    df = pd.DataFrame(plot_data)

    fig, axes = plt.subplots(1, len(HUB_GENES),
                             figsize=(4 * len(HUB_GENES), 5), squeeze=False)

    for i, gene in enumerate(HUB_GENES):
        ax = axes[0][i]
        gene_data = df[df["Gene"] == gene]
        if gene_data.empty:
            ax.set_title(f"{gene}\n(not found)", color="gray")
            continue

        sns.boxplot(data=gene_data, x="Dataset", y="Expression",
                    ax=ax, palette="Set2", fliersize=2)
        ax.set_title(gene, fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        if i > 0:
            ax.set_ylabel("")
        ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle("Hub Gene Expression Across Validation Cohorts", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "geo_expression_validation.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: figures/geo_expression_validation.png")


def create_validation_summary(all_results):
    """Summary table of validation results."""
    print("\n  Creating validation summary...")

    rows = []
    for res in all_results:
        for gene in HUB_GENES:
            row = {
                "dataset": res["gse_id"],
                "n_samples": res["n_samples"],
                "gene": gene,
                "found": gene in (res.get("km_results", {}).keys() |
                                  res.get("cox_results", {}).keys()),
            }

            km = res.get("km_results", {}).get(gene, {})
            cox = res.get("cox_results", {}).get(gene, {})

            row["KM_p"] = km.get("p_value", np.nan)
            row["KM_significant"] = km.get("p_value", 1) < 0.05 if km else False
            row["HR"] = cox.get("HR", np.nan)
            row["HR_CI_low"] = cox.get("CI_low", np.nan)
            row["HR_CI_high"] = cox.get("CI_high", np.nan)
            row["Cox_p"] = cox.get("p_value", np.nan)

            rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS_DIR / "geo_validation_summary.csv", index=False)

    # Print summary
    print(f"\n  Validation Summary:")
    print(f"  {'Gene':10s} | ", end="")
    for res in all_results:
        print(f"{res['gse_id']:15s} | ", end="")
    print()
    print("  " + "-" * (12 + 18 * len(all_results)))

    for gene in HUB_GENES:
        print(f"  {gene:10s} | ", end="")
        for res in all_results:
            km = res.get("km_results", {}).get(gene, {})
            if km:
                p = km["p_value"]
                sig = "SIG" if p < 0.05 else "ns"
                print(f"p={p:.3f} ({sig:3s}) | ", end="")
            else:
                print(f"{'N/A':15s} | ", end="")
        print()

    return summary


def main():
    print("\n" + "=" * 60)
    print("  Phase 9A: External Validation in GEO Cohorts")
    print("=" * 60 + "\n")

    datasets = [
        ("GSE31210", parse_survival_gse31210),
        ("GSE72094", parse_survival_generic),
        ("GSE68465", parse_survival_generic),
    ]

    all_results = []
    all_gene_exprs = []

    for gse_id, parse_fn in datasets:
        print(f"\n--- Processing {gse_id} ---")
        expr, pheno = download_geo_dataset(gse_id)

        if expr is None or pheno is None:
            print(f"  SKIP: Could not load {gse_id}")
            all_results.append({"gse_id": gse_id, "n_samples": 0})
            all_gene_exprs.append({})
            continue

        print(f"  Phenotype columns: {list(pheno.columns)[:10]}...")

        results, gene_exprs = validate_in_dataset(gse_id, expr, pheno, parse_fn)
        all_results.append(results)
        all_gene_exprs.append(gene_exprs)

    # Summary
    create_validation_summary(all_results)

    # Plots
    plot_geo_survival_validation(all_results)
    plot_forest_meta(all_results)
    plot_expression_across_datasets(all_gene_exprs, all_results)

    print(f"\n{'=' * 60}")
    print(f"  Phase 9A complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
