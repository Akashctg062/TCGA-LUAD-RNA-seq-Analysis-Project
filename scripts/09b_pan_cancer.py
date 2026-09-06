"""
09b_pan_cancer.py
==================
Pan-cancer analysis of hub genes across 33 TCGA cancer types.
Uses GDC API to query expression quantification for hub genes
across all TCGA projects, plus survival analysis.

Usage:
    python scripts/09b_pan_cancer.py

Output:
    data/pancancer/
    results/pancancer_expression.csv
    results/pancancer_survival.csv
    figures/pancancer_expression_heatmap.png
    figures/pancancer_survival_heatmap.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import mannwhitneyu
from lifelines import CoxPHFitter
import requests
import json
import time
import warnings
warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
PANCAN_DIR = DATA_DIR / "pancancer"
RESULTS_DIR = PROJECT_DIR / "results"
FIG_DIR = PROJECT_DIR / "figures"
PANCAN_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.size": 9, "axes.titlesize": 12,
})

HUB_GENES = ["BIRC5", "CCNB1", "CCNB2", "CDC20", "KIF2C"]

TCGA_CANCERS = [
    "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA",
    "GBM", "HNSC", "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC",
    "LUAD", "LUSC", "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ",
    "SARC", "SKCM", "STAD", "TGCT", "THCA", "THYM", "UCEC", "UCS", "UVM",
]


def get_ensembl_ids_for_hub_genes():
    """Get Ensembl IDs for hub genes from our gene mapping or GDC."""
    gene_map_path = DATA_DIR / "gene_id_mapping.csv"
    if gene_map_path.exists():
        gm = pd.read_csv(gene_map_path, index_col=0)
        name_to_id = dict(zip(gm["gene_name"], gm.index))
        return {g: name_to_id.get(g, g) for g in HUB_GENES}

    # Fallback: well-known Ensembl IDs for these genes
    return {
        "BIRC5": "ENSG00000089685",
        "CCNB1": "ENSG00000134057",
        "CCNB2": "ENSG00000157456",
        "CDC20": "ENSG00000117399",
        "KIF2C": "ENSG00000142945",
    }


def download_pancancer_expression():
    """Download hub gene expression across all TCGA cancers via GDC API.
    Uses the gene expression quantification endpoint."""

    expr_cache = PANCAN_DIR / "pancancer_hub_expression.csv"
    if expr_cache.exists():
        print("  Loading cached pan-cancer expression...")
        return pd.read_csv(expr_cache, index_col=0)

    print("  Downloading pan-cancer expression via GDC API...")
    print("  (Querying STAR-Counts for hub genes across 33 cancer types)")

    ensembl_ids = get_ensembl_ids_for_hub_genes()
    # Strip version suffixes for matching
    ensembl_base = {g: eid.split(".")[0] for g, eid in ensembl_ids.items()}

    all_data = []

    for cancer in TCGA_CANCERS:
        cancer_cache = PANCAN_DIR / f"{cancer}_hub_expr.csv"

        if cancer_cache.exists():
            df = pd.read_csv(cancer_cache, index_col=0)
            all_data.append(df)
            print(f"    TCGA-{cancer}: loaded from cache ({len(df)} samples)")
            continue

        project_id = f"TCGA-{cancer}"
        print(f"    TCGA-{cancer}...", end=" ", flush=True)

        # Step 1: Get file UUIDs for STAR-Counts in this project
        files_url = "https://api.gdc.cancer.gov/files"
        filters = {
            "op": "and",
            "content": [
                {"op": "=", "content": {"field": "cases.project.project_id", "value": project_id}},
                {"op": "=", "content": {"field": "data_type", "value": "Gene Expression Quantification"}},
                {"op": "=", "content": {"field": "analysis.workflow_type", "value": "STAR - Counts"}},
                {"op": "=", "content": {"field": "data_format", "value": "TSV"}},
            ]
        }

        params = {
            "filters": json.dumps(filters),
            "fields": "file_id,cases.submitter_id,cases.samples.sample_type",
            "size": "2000",
            "format": "json",
        }

        try:
            response = requests.get(files_url, params=params, timeout=60)
            if response.status_code != 200:
                print(f"HTTP {response.status_code}")
                continue

            files = response.json()["data"]["hits"]
            if not files:
                print("no files")
                continue

            # Map file_id -> case info
            file_info = []
            for f in files:
                case_id = ""
                sample_type = ""
                for case in f.get("cases", []):
                    case_id = case.get("submitter_id", "")
                    for sample in case.get("samples", []):
                        sample_type = sample.get("sample_type", "")
                        break

                file_info.append({
                    "file_id": f["file_id"],
                    "case_id": case_id,
                    "sample_type": sample_type,
                })

            # Step 2: Download a subset of files (max 30 per cancer for speed)
            # Prioritize: take up to 20 tumor + 10 normal
            info_df = pd.DataFrame(file_info)
            tumor_files = info_df[info_df["sample_type"].str.contains("Tumor", case=False, na=False)].head(20)
            normal_files = info_df[info_df["sample_type"].str.contains("Normal|Solid Tissue Normal", case=False, na=False)].head(10)
            selected = pd.concat([tumor_files, normal_files])

            if len(selected) == 0:
                selected = info_df.head(25)

            # Step 3: Download each file and extract hub gene values
            samples_data = []
            for _, frow in selected.iterrows():
                file_id = frow["file_id"]

                try:
                    dl_url = f"https://api.gdc.cancer.gov/data/{file_id}"
                    resp = requests.get(dl_url, timeout=120)
                    if resp.status_code != 200:
                        continue

                    # Parse TSV
                    lines = resp.text.strip().split("\n")
                    # Find header
                    header_idx = 0
                    for idx, line in enumerate(lines):
                        if line.startswith("gene_id") or line.startswith("gene_name"):
                            header_idx = idx
                            break

                    # Parse expression
                    gene_values = {}
                    for line in lines[header_idx + 1:]:
                        parts = line.split("\t")
                        if len(parts) < 4:
                            continue

                        gene_id_raw = parts[0].split(".")[0]  # Strip version
                        gene_name = parts[1] if len(parts) > 1 else ""

                        # Match by gene name or Ensembl ID
                        for hub_gene, ens_id in ensembl_base.items():
                            if gene_name == hub_gene or gene_id_raw == ens_id:
                                # Use TPM (column index varies; typically: gene_id, gene_name, gene_type, unstranded, stranded_first, stranded_second, tpm_unstranded, fpkm_unstranded, fpkm_uq_unstranded)
                                # TPM is usually column 6 (0-indexed)
                                try:
                                    tpm_val = float(parts[6]) if len(parts) > 6 else float(parts[3])
                                    gene_values[hub_gene] = tpm_val
                                except (ValueError, IndexError):
                                    try:
                                        gene_values[hub_gene] = float(parts[3])
                                    except:
                                        pass
                                break

                    if gene_values:
                        gene_values["case_id"] = frow["case_id"]
                        gene_values["sample_type"] = frow["sample_type"]
                        gene_values["cancer_type"] = cancer
                        samples_data.append(gene_values)

                except Exception:
                    continue

                time.sleep(0.1)  # Rate limiting

            if samples_data:
                cancer_df = pd.DataFrame(samples_data)
                cancer_df.to_csv(cancer_cache)
                all_data.append(cancer_df)
                print(f"{len(cancer_df)} samples ({len(tumor_files)} tumor, {len(normal_files)} normal)")
            else:
                print("no data extracted")

        except Exception as e:
            print(f"error: {e}")

    if not all_data:
        print("  WARNING: No pan-cancer data downloaded")
        return None

    combined = pd.concat(all_data, ignore_index=True)
    combined.to_csv(expr_cache)
    print(f"\n  Total: {len(combined)} samples across {combined['cancer_type'].nunique()} cancer types")
    return combined


def download_pancancer_survival():
    """Download survival data for all TCGA cancer types via GDC API."""
    surv_cache = PANCAN_DIR / "pancancer_survival.csv"
    if surv_cache.exists():
        print("  Loading cached pan-cancer survival...")
        return pd.read_csv(surv_cache, index_col=0)

    print("  Downloading pan-cancer survival data via GDC API...")

    all_surv = []

    for cancer in TCGA_CANCERS:
        clin_cache = PANCAN_DIR / f"{cancer}_clinical.csv"

        if clin_cache.exists():
            df = pd.read_csv(clin_cache, index_col=0)
            all_surv.append(df)
            continue

        url = "https://api.gdc.cancer.gov/cases"
        filters = {
            "op": "and",
            "content": [
                {"op": "=", "content": {"field": "project.project_id", "value": f"TCGA-{cancer}"}},
            ]
        }

        params = {
            "filters": json.dumps(filters),
            "fields": "submitter_id,demographic.vital_status,demographic.days_to_death,diagnoses.days_to_last_follow_up",
            "size": "2000",
            "format": "json",
        }

        try:
            response = requests.get(url, params=params, timeout=60)
            if response.status_code != 200:
                continue

            data = response.json()["data"]["hits"]

            rows = []
            for case in data:
                row = {"case_id": case["submitter_id"], "cancer_type": cancer}

                demo = case.get("demographic", {})
                if isinstance(demo, list):
                    demo = demo[0] if demo else {}
                row["vital_status"] = demo.get("vital_status", "")
                row["days_to_death"] = demo.get("days_to_death", None)

                diag = case.get("diagnoses", [{}])
                if isinstance(diag, list) and diag:
                    row["days_to_follow_up"] = diag[0].get("days_to_last_follow_up", None)
                else:
                    row["days_to_follow_up"] = None

                # Compute OS
                if row["vital_status"] == "Dead" and row["days_to_death"]:
                    row["os_time"] = float(row["days_to_death"]) / 365.25
                    row["os_event"] = 1
                elif row["days_to_follow_up"]:
                    row["os_time"] = float(row["days_to_follow_up"]) / 365.25
                    row["os_event"] = 0
                else:
                    row["os_time"] = None
                    row["os_event"] = None

                rows.append(row)

            df = pd.DataFrame(rows).set_index("case_id")
            df.to_csv(clin_cache)
            all_surv.append(df)
            print(f"    TCGA-{cancer}: {len(df)} cases")

        except Exception as e:
            print(f"    TCGA-{cancer}: error: {e}")

        time.sleep(0.2)

    if all_surv:
        combined = pd.concat(all_surv)
        combined.to_csv(surv_cache)
        return combined

    return None


def compute_pancancer_expression_stats(pancan_expr):
    """Compute expression statistics per cancer type."""
    print("\n" + "=" * 60)
    print("Step 1: Pan-Cancer Expression Analysis")
    print("=" * 60)

    if pancan_expr is None or pancan_expr.empty:
        print("  No pan-cancer expression data available")
        return None

    stats = []
    for cancer in TCGA_CANCERS:
        cancer_data = pancan_expr[pancan_expr["cancer_type"] == cancer]
        if cancer_data.empty:
            continue

        row = {"cancer_type": cancer, "n_samples": len(cancer_data)}
        for gene in HUB_GENES:
            if gene in cancer_data.columns:
                vals = pd.to_numeric(cancer_data[gene], errors="coerce")
                # Log2(TPM+1) for visualization
                vals_log = np.log2(vals + 1)
                row[f"{gene}_mean"] = vals_log.mean()
                row[f"{gene}_median"] = vals_log.median()
                row[f"{gene}_std"] = vals_log.std()

        stats.append(row)

    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(RESULTS_DIR / "pancancer_expression.csv", index=False)

    print(f"  Computed expression across {len(stats_df)} cancer types")

    # Print top/bottom cancers for each gene
    for gene in HUB_GENES:
        col = f"{gene}_mean"
        if col in stats_df.columns:
            sorted_df = stats_df.dropna(subset=[col]).sort_values(col, ascending=False)
            if len(sorted_df) >= 3:
                top3 = sorted_df.head(3)
                print(f"  {gene} highest: {', '.join(top3['cancer_type'].values)}")

    return stats_df


def compute_pancancer_survival(pancan_expr, surv_data):
    """Cox regression for hub genes across all cancer types."""
    print("\n" + "=" * 60)
    print("Step 2: Pan-Cancer Survival Analysis")
    print("=" * 60)

    if pancan_expr is None or surv_data is None:
        print("  Missing data for pan-cancer survival")
        return None

    results = []

    for cancer in TCGA_CANCERS:
        cancer_expr = pancan_expr[pancan_expr["cancer_type"] == cancer]
        cancer_surv = surv_data[surv_data["cancer_type"] == cancer]

        if cancer_expr.empty or cancer_surv.empty:
            continue

        for gene in HUB_GENES:
            if gene not in cancer_expr.columns:
                continue

            try:
                # Match by case_id
                expr_cases = dict(zip(
                    cancer_expr["case_id"].astype(str),
                    pd.to_numeric(cancer_expr[gene], errors="coerce")
                ))

                cox_rows = []
                for case_id in cancer_surv.index:
                    case_str = str(case_id)
                    os_time = cancer_surv.loc[case_id, "os_time"]
                    os_event = cancer_surv.loc[case_id, "os_event"]
                    gene_val = expr_cases.get(case_str, None)

                    if gene_val is not None and pd.notna(os_time) and pd.notna(os_event) and float(os_time) > 0:
                        cox_rows.append({
                            "time": float(os_time),
                            "event": int(float(os_event)),
                            "gene_expr": np.log2(float(gene_val) + 1),
                        })

                cox_df = pd.DataFrame(cox_rows)

                if len(cox_df) < 30 or cox_df["event"].sum() < 5:
                    continue

                # Z-score
                cox_df["gene_expr"] = (cox_df["gene_expr"] - cox_df["gene_expr"].mean()) / (cox_df["gene_expr"].std() + 1e-10)

                cph = CoxPHFitter()
                cph.fit(cox_df, duration_col="time", event_col="event")

                hr = np.exp(cph.params_["gene_expr"])
                pval = cph.summary["p"]["gene_expr"]

                results.append({
                    "cancer_type": cancer,
                    "gene": gene,
                    "HR": hr,
                    "p_value": pval,
                    "log_HR": np.log2(hr),
                    "n_samples": len(cox_df),
                    "n_events": int(cox_df["event"].sum()),
                    "significant": pval < 0.05,
                })

                if pval < 0.05:
                    direction = "Risk" if hr > 1 else "Protective"
                    print(f"    {cancer:6s} × {gene:8s}: HR={hr:.2f} p={pval:.3e} ({direction})")

            except Exception:
                continue

    if results:
        result_df = pd.DataFrame(results)
        result_df.to_csv(RESULTS_DIR / "pancancer_survival.csv", index=False)
        return result_df

    return None


def plot_pancancer_expression_heatmap(stats_df):
    """Heatmap of hub gene expression across cancer types."""
    print("\n  Plotting pan-cancer expression heatmap...")

    if stats_df is None or stats_df.empty:
        return

    mean_cols = [f"{g}_mean" for g in HUB_GENES if f"{g}_mean" in stats_df.columns]
    if not mean_cols:
        return

    mat = stats_df.set_index("cancer_type")[mean_cols].copy()
    mat.columns = [c.replace("_mean", "") for c in mat.columns]
    mat = mat.dropna(how="all")

    # Z-score normalize per gene
    mat_z = (mat - mat.mean()) / (mat.std() + 1e-10)

    # Sort by mean z-score (highest expression on top)
    mat_z["mean_z"] = mat_z.mean(axis=1)
    mat_z = mat_z.sort_values("mean_z", ascending=True)
    mat_z = mat_z.drop(columns=["mean_z"])

    fig, ax = plt.subplots(figsize=(10, max(8, len(mat_z) * 0.3)))

    sns.heatmap(
        mat_z, cmap="RdBu_r", center=0,
        xticklabels=True, yticklabels=True,
        linewidths=0.5, ax=ax,
        cbar_kws={"label": "Z-score (Expression)", "shrink": 0.7},
    )

    # Highlight LUAD row
    if "LUAD" in mat_z.index:
        luad_idx = list(mat_z.index).index("LUAD")
        ax.add_patch(plt.Rectangle((0, luad_idx), len(HUB_GENES), 1,
                                    fill=False, edgecolor="gold", lw=3))
        ax.text(len(HUB_GENES) + 0.2, luad_idx + 0.5, "◀ LUAD",
                va="center", fontsize=10, fontweight="bold", color="gold")

    ax.set_title("Pan-Cancer Hub Gene Expression\n(Z-score normalized, 33 TCGA Cancer Types)", fontsize=13)
    ax.set_xlabel("")
    ax.set_ylabel("")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "pancancer_expression_heatmap.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: figures/pancancer_expression_heatmap.png")


def plot_pancancer_survival_heatmap(surv_df):
    """Heatmap of survival significance across cancer types."""
    print("  Plotting pan-cancer survival heatmap...")

    if surv_df is None or surv_df.empty:
        return

    # Pivot
    pivot = surv_df.pivot_table(index="cancer_type", columns="gene",
                                 values="log_HR", aggfunc="first")

    pval_pivot = surv_df.pivot_table(index="cancer_type", columns="gene",
                                      values="p_value", aggfunc="first")

    # Significance annotations
    annot = pivot.copy().astype(str)
    for cancer in pivot.index:
        for gene in pivot.columns:
            hr = pivot.loc[cancer, gene]
            p = pval_pivot.loc[cancer, gene] if cancer in pval_pivot.index and gene in pval_pivot.columns else 1
            if pd.notna(hr) and pd.notna(p):
                star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                annot.loc[cancer, gene] = star
            else:
                annot.loc[cancer, gene] = ""

    # Sort by number of significant associations
    sig_count = (pval_pivot < 0.05).sum(axis=1).sort_values(ascending=True)
    pivot = pivot.reindex(sig_count.index)
    annot = annot.reindex(sig_count.index)

    fig, ax = plt.subplots(figsize=(10, max(8, len(pivot) * 0.35)))

    sns.heatmap(
        pivot.astype(float), cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        annot=annot, fmt="",
        xticklabels=True, yticklabels=True,
        linewidths=0.5, ax=ax,
        cbar_kws={"label": "log2(Hazard Ratio)", "shrink": 0.7},
        annot_kws={"fontsize": 9, "fontweight": "bold"},
    )

    # Highlight LUAD
    if "LUAD" in pivot.index:
        luad_idx = list(pivot.index).index("LUAD")
        ax.add_patch(plt.Rectangle((0, luad_idx), len(pivot.columns), 1,
                                    fill=False, edgecolor="gold", lw=3))

    ax.set_title("Pan-Cancer Prognostic Significance of Hub Genes\n(* p<0.05  ** p<0.01  *** p<0.001)",
                 fontsize=13)
    ax.set_xlabel("")
    ax.set_ylabel("")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "pancancer_survival_heatmap.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: figures/pancancer_survival_heatmap.png")


def main():
    print("\n" + "=" * 60)
    print("  Phase 9B: Pan-Cancer Analysis (33 TCGA Cancer Types)")
    print("=" * 60 + "\n")

    # Step 1: Download expression
    pancan_expr = download_pancancer_expression()

    # Step 2: Download survival
    surv_data = download_pancancer_survival()

    # Step 3: Expression statistics
    stats_df = compute_pancancer_expression_stats(pancan_expr)

    # Step 4: Survival analysis
    surv_results = compute_pancancer_survival(pancan_expr, surv_data)

    # Plots
    plot_pancancer_expression_heatmap(stats_df)
    plot_pancancer_survival_heatmap(surv_results)

    # Summary
    if surv_results is not None:
        sig = surv_results[surv_results["significant"]]
        print(f"\n  Summary:")
        print(f"    Cancer types analyzed: {surv_results['cancer_type'].nunique()}")
        print(f"    Significant associations: {len(sig)}")
        for gene in HUB_GENES:
            gene_sig = sig[sig["gene"] == gene]
            cancers = ", ".join(gene_sig["cancer_type"].values)
            print(f"    {gene}: significant in {len(gene_sig)} cancers ({cancers})")

    print(f"\n{'=' * 60}")
    print(f"  Phase 9B complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
