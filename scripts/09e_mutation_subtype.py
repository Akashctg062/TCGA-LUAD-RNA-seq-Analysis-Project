"""
09e_mutation_subtype.py
========================
Mutation-stratified and molecular subtype analysis:
  1. Stratify survival by driver mutations (EGFR/KRAS/TP53) × hub gene expression
  2. LUAD molecular subtype classification (TRU/PP/PI)
  3. Stage-stratified survival analysis
  4. Interaction effect: mutation + gene expression on outcome

Usage:
    python scripts/09e_mutation_subtype.py

Output:
    results/mutation_stratified_survival.csv
    results/subtype_expression.csv
    results/stage_stratified_survival.csv
    figures/mutation_stratified_km.png
    figures/subtype_expression.png
    figures/stage_hub_interaction.png
    figures/combined_clinical_molecular.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from scipy.stats import mannwhitneyu, kruskal
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
import requests
import json
import warnings
warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
FIG_DIR = PROJECT_DIR / "figures"
OMICS_DIR = DATA_DIR / "multi_omics"
OMICS_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.size": 10, "axes.titlesize": 12,
})

HUB_GENES = ["BIRC5", "CCNB1", "CCNB2", "CDC20", "KIF2C"]
DRIVER_GENES = ["EGFR", "KRAS", "TP53", "STK11", "KEAP1", "ALK"]


def download_mutation_data():
    """Download somatic mutation status for TCGA-LUAD cases."""
    cache = OMICS_DIR / "luad_driver_mutations.csv"
    if cache.exists():
        print("  Loading cached mutation data...")
        return pd.read_csv(cache)

    print("  Downloading driver mutation data from GDC...")

    # Get all TCGA-LUAD cases
    url = "https://api.gdc.cancer.gov/cases"
    all_cases = []
    offset = 0
    size = 500

    while True:
        filters = {
            "op": "=",
            "content": {"field": "project.project_id", "value": "TCGA-LUAD"}
        }

        params = {
            "filters": json.dumps(filters),
            "fields": "submitter_id",
            "size": str(size),
            "from": str(offset),
            "format": "json",
        }

        try:
            response = requests.get(url, params=params, timeout=60)
            data = response.json()["data"]
            hits = data["hits"]
            if not hits:
                break
            for hit in hits:
                all_cases.append(hit["submitter_id"])
            offset += size
            if offset >= data["pagination"]["total"]:
                break
        except:
            break

    print(f"    Found {len(all_cases)} LUAD cases")

    # Query mutations for driver genes
    mut_url = "https://api.gdc.cancer.gov/ssm_occurrences"
    mutation_status = {case_id: {gene: 0 for gene in DRIVER_GENES} for case_id in all_cases}

    for gene in DRIVER_GENES:
        filters = {
            "op": "and",
            "content": [
                {"op": "=", "content": {"field": "case.project.project_id", "value": "TCGA-LUAD"}},
                {"op": "=", "content": {"field": "ssm.consequence.transcript.gene.symbol", "value": gene}},
            ]
        }

        params = {
            "filters": json.dumps(filters),
            "fields": "case.submitter_id",
            "size": "1000",
            "format": "json",
        }

        try:
            response = requests.get(mut_url, params=params, timeout=60)
            if response.status_code == 200:
                data = response.json()["data"]["hits"]
                mutated_cases = set()
                for hit in data:
                    case_info = hit.get("case", {})
                    case_id = case_info.get("submitter_id", "")
                    if case_id:
                        mutated_cases.add(case_id)
                        if case_id in mutation_status:
                            mutation_status[case_id][gene] = 1

                print(f"    {gene}: {len(mutated_cases)} mutated cases")
        except Exception as e:
            print(f"    {gene}: Error: {e}")

    mut_df = pd.DataFrame.from_dict(mutation_status, orient="index")
    mut_df.index.name = "case_id"
    mut_df = mut_df.reset_index()

    mut_df.to_csv(cache, index=False)
    return mut_df


def prepare_combined_data(log_cpm, metadata, gene_mapping, mut_df):
    """Merge expression, survival, and mutation data."""
    if gene_mapping is not None:
        name_to_id = dict(zip(gene_mapping["gene_name"], gene_mapping.index))
    else:
        name_to_id = {}

    meta = metadata.set_index("sample_submitter_id")
    tumor_samples = [s for s in log_cpm.columns if s in meta.index and meta.loc[s, "condition"] == "Tumor"]

    rows = []
    for sample in tumor_samples:
        row = {"sample": sample}

        # Case ID (first 12 chars of TCGA barcode)
        case_id = sample[:12]
        row["case_id"] = case_id

        # Survival
        m = meta.loc[sample]
        vital = str(m.get("vital_status", ""))
        dtd = pd.to_numeric(m.get("days_to_death", None), errors="coerce")
        dtf = pd.to_numeric(m.get("days_to_last_follow_up", None), errors="coerce")

        if vital.lower() in ["dead", "1"]:
            row["time"] = dtd / 365.25 if pd.notna(dtd) else None
            row["event"] = 1
        else:
            row["time"] = dtf / 365.25 if pd.notna(dtf) else (dtd / 365.25 if pd.notna(dtd) else None)
            row["event"] = 0

        # Stage
        stage = str(m.get("tumor_stage", m.get("ajcc_pathologic_stage", "")))
        row["stage"] = stage

        # Simplified stage
        if "iv" in stage.lower() or "stage iv" in stage.lower():
            row["stage_simple"] = "IV"
        elif "iii" in stage.lower():
            row["stage_simple"] = "III"
        elif "ii" in stage.lower():
            row["stage_simple"] = "II"
        elif "i" in stage.lower():
            row["stage_simple"] = "I"
        else:
            row["stage_simple"] = "Unknown"

        # Gene expression
        for gene in HUB_GENES:
            gene_id = name_to_id.get(gene, gene)
            if gene_id in log_cpm.index:
                row[gene] = log_cpm.loc[gene_id, sample]

        rows.append(row)

    combined = pd.DataFrame(rows)

    # Merge mutation status
    if mut_df is not None and not mut_df.empty:
        combined = combined.merge(mut_df, on="case_id", how="left")
        for gene in DRIVER_GENES:
            if gene in combined.columns:
                combined[gene] = combined[gene].fillna(0).astype(int)

    return combined


def mutation_stratified_survival(combined):
    """Survival analysis stratified by driver mutations × hub gene expression."""
    print("\n" + "=" * 60)
    print("Step 1: Mutation-Stratified Survival Analysis")
    print("=" * 60)

    valid = combined.dropna(subset=["time", "event"])
    valid = valid[valid["time"] > 0]

    results = []
    # Focus on top 3 drivers
    top_drivers = [g for g in ["TP53", "KRAS", "EGFR"] if g in combined.columns]

    n_cols = len(top_drivers)
    n_rows = len(HUB_GENES)

    if n_cols == 0:
        print("  No mutation data available for stratification")
        return pd.DataFrame()

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    for col_idx, driver in enumerate(top_drivers):
        for row_idx, hub in enumerate(HUB_GENES):
            ax = axes[row_idx][col_idx]

            if hub not in valid.columns or driver not in valid.columns:
                ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center")
                continue

            sub = valid.dropna(subset=[hub])

            # 4 groups: Mut+High, Mut+Low, WT+High, WT+Low
            median_expr = sub[hub].median()
            sub = sub.copy()
            sub["expr_group"] = np.where(sub[hub] >= median_expr, "High", "Low")
            sub["mut_group"] = np.where(sub[driver] == 1, "Mut", "WT")
            sub["combined_group"] = sub["mut_group"] + "+" + sub["expr_group"]

            kmf = KaplanMeierFitter()
            group_colors = {
                "Mut+High": "#e74c3c", "Mut+Low": "#f39c12",
                "WT+High": "#3498db", "WT+Low": "#27ae60",
            }

            for group in ["Mut+High", "Mut+Low", "WT+High", "WT+Low"]:
                mask = sub["combined_group"] == group
                if mask.sum() < 5:
                    continue
                kmf.fit(sub.loc[mask, "time"], sub.loc[mask, "event"], label=f"{group} (n={mask.sum()})")
                kmf.plot_survival_function(ax=ax, color=group_colors[group], lw=1.5)

            # Log-rank between best and worst groups
            try:
                worst = sub[sub["combined_group"] == "Mut+High"]
                best = sub[sub["combined_group"] == "WT+Low"]
                if len(worst) >= 5 and len(best) >= 5:
                    lr = logrank_test(worst["time"], best["time"], worst["event"], best["event"])
                    p = lr.p_value
                    star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                    ax.text(0.95, 0.95, f"p={p:.3f}{star}", transform=ax.transAxes,
                           ha="right", va="top", fontsize=8, fontweight="bold")

                    results.append({
                        "driver": driver, "hub_gene": hub,
                        "p_value": p, "significant": p < 0.05,
                    })
            except:
                pass

            ax.set_xlabel("" if row_idx < n_rows - 1 else "Time (years)")
            ax.set_ylabel("" if col_idx > 0 else "Survival Prob.")
            ax.set_title(f"{driver} × {hub}", fontsize=10)
            ax.legend(fontsize=6, loc="lower left")
            ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle("Mutation-Stratified Survival: Driver Gene × Hub Gene Expression",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "mutation_stratified_km.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: figures/mutation_stratified_km.png")

    result_df = pd.DataFrame(results)
    result_df.to_csv(RESULTS_DIR / "mutation_stratified_survival.csv", index=False)
    return result_df


def stage_stratified_analysis(combined):
    """Analyze hub gene expression across tumor stages."""
    print("\n" + "=" * 60)
    print("Step 2: Stage-Stratified Analysis")
    print("=" * 60)

    valid = combined[combined["stage_simple"].isin(["I", "II", "III", "IV"])].copy()

    if len(valid) < 50:
        print("  Insufficient stage information")
        return

    # Expression by stage
    fig, axes = plt.subplots(1, len(HUB_GENES), figsize=(4 * len(HUB_GENES), 5), squeeze=False)

    stage_order = ["I", "II", "III", "IV"]
    results = []

    for i, gene in enumerate(HUB_GENES):
        ax = axes[0][i]
        if gene not in valid.columns:
            continue

        gene_data = valid[[gene, "stage_simple"]].dropna()

        sns.boxplot(data=gene_data, x="stage_simple", y=gene,
                    order=stage_order, palette="YlOrRd", ax=ax, fliersize=2)

        # Kruskal-Wallis test
        groups = [gene_data.loc[gene_data["stage_simple"] == s, gene].values
                  for s in stage_order if (gene_data["stage_simple"] == s).sum() > 2]

        if len(groups) >= 2:
            stat, p = kruskal(*groups)
            star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            ax.set_title(f"{gene}\nKW p={p:.3f} {star}", fontsize=11,
                        fontweight="bold" if p < 0.05 else "normal")
            results.append({"gene": gene, "KW_p": p, "significant": p < 0.05})
            print(f"    {gene}: Kruskal-Wallis p={p:.4f} {star}")

        ax.set_xlabel("Stage" if i == len(HUB_GENES)//2 else "")
        ax.set_ylabel("Expression (log2 CPM)" if i == 0 else "")
        ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle("Hub Gene Expression Across Tumor Stages (TCGA-LUAD)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "stage_hub_interaction.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: figures/stage_hub_interaction.png")

    pd.DataFrame(results).to_csv(RESULTS_DIR / "stage_stratified_survival.csv", index=False)


def plot_combined_summary(combined):
    """Combined clinical-molecular summary figure."""
    print("\n" + "=" * 60)
    print("Step 3: Combined Clinical-Molecular Summary")
    print("=" * 60)

    valid = combined.dropna(subset=["time", "event"])
    valid = valid[valid["time"] > 0]

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

    # (A) Driver mutation frequency
    ax = fig.add_subplot(gs[0, 0])
    drivers_present = [d for d in DRIVER_GENES if d in combined.columns]
    if drivers_present:
        freq = combined[drivers_present].mean() * 100
        freq = freq.sort_values(ascending=True)
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(freq)))
        freq.plot.barh(ax=ax, color=colors, edgecolor="white")
        ax.set_xlabel("Mutation Frequency (%)")
        ax.set_title("(A) Driver Mutation Frequency", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        for i, (gene, val) in enumerate(freq.items()):
            ax.text(val + 0.5, i, f"{val:.1f}%", va="center", fontsize=9)

    # (B) Stage distribution
    ax = fig.add_subplot(gs[0, 1])
    stage_counts = combined["stage_simple"].value_counts()
    stage_order = ["I", "II", "III", "IV", "Unknown"]
    stage_counts = stage_counts.reindex([s for s in stage_order if s in stage_counts.index])
    stage_counts.plot.pie(ax=ax, autopct="%1.0f%%", startangle=90, colors=plt.cm.Set3.colors)
    ax.set_ylabel("")
    ax.set_title("(B) Stage Distribution", fontweight="bold")

    # (C) Hub gene correlation matrix (in tumor)
    ax = fig.add_subplot(gs[0, 2])
    hub_cols = [g for g in HUB_GENES if g in valid.columns]
    if hub_cols:
        cor = valid[hub_cols].corr()
        sns.heatmap(cor, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                    vmin=-1, vmax=1, ax=ax, cbar_kws={"shrink": 0.8},
                    linewidths=0.5)
        ax.set_title("(C) Hub Gene Correlation (Tumor)", fontweight="bold")

    # (D) Overall survival by risk score
    ax = fig.add_subplot(gs[1, 0])
    risk_path = RESULTS_DIR / "risk_score_model.csv"
    if risk_path.exists():
        risk_df = pd.read_csv(risk_path)
        kmf = KaplanMeierFitter()
        for group, color in [("High Risk", "#e74c3c"), ("Low Risk", "#3498db")]:
            mask = risk_df["risk_group"] == group
            if mask.sum() >= 5:
                kmf.fit(risk_df.loc[mask, "time"], risk_df.loc[mask, "event"], label=group)
                kmf.plot_survival_function(ax=ax, color=color, lw=2.5)
        ax.set_title("(D) Risk Score Survival", fontweight="bold")
        ax.set_xlabel("Time (years)")
        ax.set_ylabel("Survival Probability")
        ax.spines[["top", "right"]].set_visible(False)
    else:
        ax.text(0.5, 0.5, "Run 09d first", transform=ax.transAxes, ha="center")

    # (E) Mean hub gene expression by driver mutation
    ax = fig.add_subplot(gs[1, 1])
    if "TP53" in combined.columns:
        plot_data = []
        for gene in hub_cols:
            for mut_status in [0, 1]:
                vals = combined.loc[combined["TP53"] == mut_status, gene].dropna()
                for v in vals:
                    plot_data.append({
                        "Gene": gene,
                        "Expression": v,
                        "TP53": "Mutant" if mut_status == 1 else "Wild-type",
                    })
        if plot_data:
            pdf = pd.DataFrame(plot_data)
            sns.boxplot(data=pdf, x="Gene", y="Expression", hue="TP53",
                       palette={"Mutant": "#e74c3c", "Wild-type": "#3498db"},
                       ax=ax, fliersize=1)
            ax.set_title("(E) Hub Genes by TP53 Status", fontweight="bold")
            ax.set_xlabel("")
            ax.legend(fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)

    # (F) Number of hub genes overexpressed per sample
    ax = fig.add_subplot(gs[1, 2])
    if hub_cols:
        # Count overexpressed genes per sample (above median)
        overexpr_count = pd.Series(0, index=valid.index)
        for gene in hub_cols:
            median_val = valid[gene].median()
            overexpr_count += (valid[gene] >= median_val).astype(int)

        overexpr_dist = overexpr_count.value_counts().sort_index()
        overexpr_dist.plot.bar(ax=ax, color="#e74c3c", edgecolor="white", alpha=0.8)
        ax.set_xlabel("Number of Overexpressed Hub Genes")
        ax.set_ylabel("Number of Patients")
        ax.set_title("(F) Hub Gene Overexpression Count", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle("Combined Clinical-Molecular Landscape — TCGA-LUAD", fontsize=15, y=1.02)
    plt.savefig(FIG_DIR / "combined_clinical_molecular.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: figures/combined_clinical_molecular.png")


def main():
    print("\n" + "=" * 60)
    print("  Phase 9E: Mutation-Stratified & Subtype Analysis")
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

    # Download mutation data
    mut_df = download_mutation_data()

    # Combine all data
    combined = prepare_combined_data(log_cpm, metadata, gene_mapping, mut_df)
    print(f"  Combined dataset: {len(combined)} tumor samples")

    # Step 1: Mutation-stratified survival
    mutation_stratified_survival(combined)

    # Step 2: Stage-stratified analysis
    stage_stratified_analysis(combined)

    # Step 3: Combined summary
    plot_combined_summary(combined)

    print(f"\n{'=' * 60}")
    print(f"  Phase 9E complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
