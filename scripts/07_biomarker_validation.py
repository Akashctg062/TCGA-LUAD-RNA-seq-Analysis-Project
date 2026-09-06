"""
07_biomarker_validation.py
===========================
Biomarker diagnostic validation: ROC curves, AUC, logistic regression,
multi-gene signature, and performance evaluation.

Usage:
    python scripts/07_biomarker_validation.py

Output:
    results/roc_results.csv
    results/multi_gene_model.csv
    results/diagnostic_performance.csv
    figures/roc_individual.png
    figures/roc_multi_gene.png
    figures/diagnostic_nomogram.png
    figures/biomarker_expression_violin.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, accuracy_score
)
from sklearn.preprocessing import StandardScaler
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

# Top hub genes from Phase 6 results (survival-significant + PPI central)
TOP_BIOMARKERS = ["BIRC5", "CCNB1", "CCNB2", "CDC20", "KIF2C"]


def load_data():
    """Load expression data and metadata."""
    print("Loading data...")
    counts = pd.read_csv(DATA_DIR / "filtered_counts.csv", index_col=0)
    metadata = pd.read_csv(DATA_DIR / "metadata.csv")

    gene_mapping = None
    if (DATA_DIR / "gene_id_mapping.csv").exists():
        gene_mapping = pd.read_csv(DATA_DIR / "gene_id_mapping.csv", index_col=0)

    # Normalize to log2 CPM
    lib_sizes = counts.sum(axis=0)
    cpm = counts.div(lib_sizes, axis=1) * 1e6
    log_cpm = np.log2(cpm + 1)

    return log_cpm, metadata, gene_mapping


def get_gene_expression(log_cpm, gene_name, gene_mapping):
    """Get expression vector for a gene by name."""
    if gene_mapping is not None:
        name_to_id = dict(zip(gene_mapping["gene_name"], gene_mapping.index))
        gene_id = name_to_id.get(gene_name, gene_name)
    else:
        gene_id = gene_name

    if gene_id in log_cpm.index:
        return log_cpm.loc[gene_id]
    return None


def individual_roc_analysis(log_cpm, metadata, gene_mapping):
    """ROC analysis for individual biomarker genes."""
    print("\n" + "=" * 60)
    print("Step 1: Individual Gene ROC Analysis (Tumor vs Normal)")
    print("=" * 60)

    meta = metadata.set_index("sample_submitter_id")
    common = [s for s in log_cpm.columns if s in meta.index]
    labels = meta.loc[common, "condition"]
    y = (labels == "Tumor").astype(int).values

    results = []
    gene_exprs = {}

    for gene in TOP_BIOMARKERS:
        expr = get_gene_expression(log_cpm, gene, gene_mapping)
        if expr is None:
            print(f"  {gene}: not found, skipping")
            continue

        x = expr[common].values

        # ROC
        fpr, tpr, thresholds = roc_curve(y, x)
        roc_auc = auc(fpr, tpr)

        # If AUC < 0.5, gene is downregulated in tumor — flip
        if roc_auc < 0.5:
            fpr, tpr, thresholds = roc_curve(y, -x)
            roc_auc = auc(fpr, tpr)
            direction = "Down in Tumor"
        else:
            direction = "Up in Tumor"

        # Optimal threshold (Youden's J)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_sensitivity = tpr[optimal_idx]
        optimal_specificity = 1 - fpr[optimal_idx]

        # Mann-Whitney test
        tumor_expr = x[y == 1]
        normal_expr = x[y == 0]
        stat, pval = mannwhitneyu(tumor_expr, normal_expr, alternative="two-sided")

        results.append({
            "gene": gene,
            "AUC": roc_auc,
            "sensitivity": optimal_sensitivity,
            "specificity": optimal_specificity,
            "direction": direction,
            "p_value": pval,
            "fpr": fpr,
            "tpr": tpr,
        })
        gene_exprs[gene] = x

        print(f"  {gene:10s}  AUC={roc_auc:.3f}  Sens={optimal_sensitivity:.3f}  "
              f"Spec={optimal_specificity:.3f}  {direction}  p={pval:.2e}")

    # Plot individual ROC curves
    print("\n  Plotting individual ROC curves...")
    fig, ax = plt.subplots(figsize=(8, 8))

    colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
    for i, res in enumerate(results):
        ax.plot(res["fpr"], res["tpr"], color=colors[i], lw=2.5,
                label=f'{res["gene"]} (AUC = {res["AUC"]:.3f})')

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC = 0.500)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    ax.set_title("ROC Curves — Individual Hub Genes\n(Tumor vs Normal Classification)", fontsize=13)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "roc_individual.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: figures/roc_individual.png")

    # Save results
    save_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ["fpr", "tpr"]}
                            for r in results])
    save_df.to_csv(RESULTS_DIR / "roc_results.csv", index=False)

    return results, gene_exprs, common, y


def multi_gene_model(log_cpm, metadata, gene_mapping, common, y):
    """Logistic regression multi-gene diagnostic model with cross-validation."""
    print("\n" + "=" * 60)
    print("Step 2: Multi-Gene Diagnostic Model (5-Fold CV)")
    print("=" * 60)

    # Build feature matrix
    X_list = []
    valid_genes = []
    for gene in TOP_BIOMARKERS:
        expr = get_gene_expression(log_cpm, gene, gene_mapping)
        if expr is not None:
            X_list.append(expr[common].values)
            valid_genes.append(gene)

    X = np.column_stack(X_list)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 5-fold stratified cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Get cross-validated predictions
    y_prob_cv = cross_val_predict(
        LogisticRegression(max_iter=1000, random_state=42),
        X_scaled, y, cv=cv, method="predict_proba"
    )[:, 1]

    y_pred_cv = (y_prob_cv >= 0.5).astype(int)

    # Overall CV metrics
    fpr_cv, tpr_cv, _ = roc_curve(y, y_prob_cv)
    auc_cv = auc(fpr_cv, tpr_cv)
    acc_cv = accuracy_score(y, y_pred_cv)

    # Confusion matrix
    cm = confusion_matrix(y, y_pred_cv)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    print(f"\n  Multi-gene model ({', '.join(valid_genes)}):")
    print(f"    5-Fold CV AUC:     {auc_cv:.4f}")
    print(f"    Accuracy:          {acc_cv:.4f}")
    print(f"    Sensitivity:       {sensitivity:.4f}")
    print(f"    Specificity:       {specificity:.4f}")
    print(f"    PPV:               {ppv:.4f}")
    print(f"    NPV:               {npv:.4f}")
    print(f"    Confusion Matrix:  TP={tp} FP={fp} FN={fn} TN={tn}")

    # Per-fold AUCs
    print(f"\n  Per-fold AUCs:")
    fold_aucs = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_scaled, y)):
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_scaled[train_idx], y[train_idx])
        y_prob_fold = lr.predict_proba(X_scaled[test_idx])[:, 1]
        fold_auc = auc(*roc_curve(y[test_idx], y_prob_fold)[:2])
        fold_aucs.append(fold_auc)
        print(f"    Fold {fold_idx+1}: AUC = {fold_auc:.4f}")
    print(f"    Mean ± SD:  {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")

    # Train final model on all data for coefficient analysis
    lr_final = LogisticRegression(max_iter=1000, random_state=42)
    lr_final.fit(X_scaled, y)

    coef_df = pd.DataFrame({
        "gene": valid_genes,
        "coefficient": lr_final.coef_[0],
        "abs_coefficient": np.abs(lr_final.coef_[0]),
    }).sort_values("abs_coefficient", ascending=False)

    print(f"\n  Model coefficients:")
    for _, row in coef_df.iterrows():
        direction = "+" if row["coefficient"] > 0 else "-"
        print(f"    {row['gene']:10s}  {direction}{row['abs_coefficient']:.3f}")

    coef_df.to_csv(RESULTS_DIR / "multi_gene_model.csv", index=False)

    # Save diagnostic performance
    perf_df = pd.DataFrame([{
        "model": "Multi-gene (5 genes)",
        "AUC_CV": auc_cv,
        "accuracy": acc_cv,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "PPV": ppv,
        "NPV": npv,
        "mean_fold_AUC": np.mean(fold_aucs),
        "sd_fold_AUC": np.std(fold_aucs),
    }])
    perf_df.to_csv(RESULTS_DIR / "diagnostic_performance.csv", index=False)

    return fpr_cv, tpr_cv, auc_cv, valid_genes, fold_aucs


def plot_multi_gene_roc(individual_results, fpr_cv, tpr_cv, auc_cv, valid_genes, fold_aucs):
    """Combined ROC plot: individual genes + multi-gene model."""
    print("\n  Plotting multi-gene ROC comparison...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: All ROC curves together
    ax = axes[0]
    colors = plt.cm.Set2(np.linspace(0, 1, len(individual_results)))

    for i, res in enumerate(individual_results):
        ax.plot(res["fpr"], res["tpr"], color=colors[i], lw=1.8, alpha=0.7,
                label=f'{res["gene"]} ({res["AUC"]:.3f})')

    ax.plot(fpr_cv, tpr_cv, color="#e74c3c", lw=3,
            label=f'Multi-gene ({auc_cv:.3f})')
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.3)

    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Comparison: Individual vs Multi-Gene Model", fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)

    # Right: AUC bar comparison
    ax2 = axes[1]
    gene_names = [r["gene"] for r in individual_results] + ["Multi-gene\n(5-gene)"]
    aucs = [r["AUC"] for r in individual_results] + [auc_cv]
    colors_bar = ["#87CEEB"] * len(individual_results) + ["#e74c3c"]

    bars = ax2.barh(gene_names, aucs, color=colors_bar, edgecolor="white", height=0.6)

    for bar, auc_val in zip(bars, aucs):
        ax2.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f"{auc_val:.3f}", va="center", fontsize=10, fontweight="bold")

    ax2.set_xlim(0.5, 1.05)
    ax2.set_xlabel("AUC", fontsize=11)
    ax2.set_title("Diagnostic AUC Comparison", fontsize=12)
    ax2.axvline(x=0.9, color="green", ls="--", alpha=0.4, label="Excellent (0.9)")
    ax2.axvline(x=0.8, color="orange", ls="--", alpha=0.4, label="Good (0.8)")
    ax2.legend(fontsize=8)
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "roc_multi_gene.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: figures/roc_multi_gene.png")


def plot_expression_violin(log_cpm, metadata, gene_mapping):
    """Violin plots of biomarker expression in tumor vs normal."""
    print("\n" + "=" * 60)
    print("Step 3: Biomarker Expression Distribution")
    print("=" * 60)

    meta = metadata.set_index("sample_submitter_id")
    common = [s for s in log_cpm.columns if s in meta.index]

    plot_data = []
    for gene in TOP_BIOMARKERS:
        expr = get_gene_expression(log_cpm, gene, gene_mapping)
        if expr is None:
            continue
        for sample in common:
            plot_data.append({
                "Gene": gene,
                "Expression (log2 CPM)": expr[sample],
                "Condition": meta.loc[sample, "condition"],
            })

    df = pd.DataFrame(plot_data)

    fig, axes = plt.subplots(1, len(TOP_BIOMARKERS), figsize=(3.5 * len(TOP_BIOMARKERS), 6),
                             sharey=False)
    if len(TOP_BIOMARKERS) == 1:
        axes = [axes]

    colors = {"Tumor": "#e74c3c", "Normal": "#3498db"}

    for i, gene in enumerate(TOP_BIOMARKERS):
        gene_data = df[df["Gene"] == gene]
        if gene_data.empty:
            continue

        sns.violinplot(data=gene_data, x="Condition", y="Expression (log2 CPM)",
                       palette=colors, ax=axes[i], inner="quartile", cut=0)

        # Add statistical test
        tumor = gene_data[gene_data["Condition"] == "Tumor"]["Expression (log2 CPM)"]
        normal = gene_data[gene_data["Condition"] == "Normal"]["Expression (log2 CPM)"]
        _, pval = mannwhitneyu(tumor, normal, alternative="two-sided")

        star = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
        y_max = gene_data["Expression (log2 CPM)"].max()
        axes[i].text(0.5, y_max + 0.3, star, ha="center", fontsize=14, fontweight="bold",
                     transform=axes[i].get_xaxis_transform())

        axes[i].set_title(gene, fontsize=13, fontweight="bold")
        axes[i].set_xlabel("")
        if i > 0:
            axes[i].set_ylabel("")
        axes[i].spines[["top", "right"]].set_visible(False)

    plt.suptitle("Hub Gene Expression: Tumor vs Normal (TCGA-LUAD)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "biomarker_expression_violin.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: figures/biomarker_expression_violin.png")


def plot_diagnostic_nomogram(log_cpm, metadata, gene_mapping):
    """Nomogram-style visualization of the multi-gene diagnostic model."""
    print("\n" + "=" * 60)
    print("Step 4: Diagnostic Model Summary Nomogram")
    print("=" * 60)

    meta = metadata.set_index("sample_submitter_id")
    common = [s for s in log_cpm.columns if s in meta.index]
    y = (meta.loc[common, "condition"] == "Tumor").astype(int).values

    # Build feature matrix
    X_list = []
    valid_genes = []
    for gene in TOP_BIOMARKERS:
        expr = get_gene_expression(log_cpm, gene, gene_mapping)
        if expr is not None:
            X_list.append(expr[common].values)
            valid_genes.append(gene)

    X = np.column_stack(X_list)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_scaled, y)

    # Predicted probabilities
    y_prob = lr.predict_proba(X_scaled)[:, 1]

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # (A) Coefficient importance
    ax1 = fig.add_subplot(gs[0, 0])
    coefs = pd.Series(lr.coef_[0], index=valid_genes).sort_values()
    colors_coef = ["#3498db" if c < 0 else "#e74c3c" for c in coefs]
    coefs.plot.barh(ax=ax1, color=colors_coef, edgecolor="white")
    ax1.set_xlabel("Logistic Regression Coefficient")
    ax1.set_title("(A) Gene Contribution to Diagnostic Model", fontweight="bold")
    ax1.axvline(0, color="black", lw=0.8)
    ax1.spines[["top", "right"]].set_visible(False)

    # (B) Predicted probability distribution
    ax2 = fig.add_subplot(gs[0, 1])
    tumor_prob = y_prob[y == 1]
    normal_prob = y_prob[y == 0]
    ax2.hist(normal_prob, bins=30, alpha=0.7, color="#3498db", label="Normal", density=True)
    ax2.hist(tumor_prob, bins=30, alpha=0.7, color="#e74c3c", label="Tumor", density=True)
    ax2.axvline(0.5, color="black", ls="--", lw=1.5, label="Threshold (0.5)")
    ax2.set_xlabel("Predicted Probability (Tumor)")
    ax2.set_ylabel("Density")
    ax2.set_title("(B) Predicted Probability Distribution", fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.spines[["top", "right"]].set_visible(False)

    # (C) Confusion matrix heatmap
    ax3 = fig.add_subplot(gs[1, 0])
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax3,
                xticklabels=["Normal", "Tumor"], yticklabels=["Normal", "Tumor"],
                cbar=False, annot_kws={"fontsize": 16})
    ax3.set_xlabel("Predicted", fontsize=11)
    ax3.set_ylabel("Actual", fontsize=11)
    ax3.set_title("(C) Confusion Matrix", fontweight="bold")

    # (D) Precision-Recall curve
    ax4 = fig.add_subplot(gs[1, 1])
    precision, recall, _ = precision_recall_curve(y, y_prob)
    ap = average_precision_score(y, y_prob)
    ax4.plot(recall, precision, color="#e74c3c", lw=2.5)
    ax4.fill_between(recall, precision, alpha=0.15, color="#e74c3c")
    ax4.set_xlabel("Recall (Sensitivity)", fontsize=11)
    ax4.set_ylabel("Precision (PPV)", fontsize=11)
    ax4.set_title(f"(D) Precision-Recall Curve (AP = {ap:.3f})", fontweight="bold")
    ax4.set_xlim([-0.02, 1.02])
    ax4.set_ylim([-0.02, 1.02])
    ax4.spines[["top", "right"]].set_visible(False)
    ax4.grid(True, alpha=0.15)

    plt.savefig(FIG_DIR / "diagnostic_nomogram.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: figures/diagnostic_nomogram.png")


def main():
    print("\n" + "=" * 60)
    print("  Phase 7: Biomarker Diagnostic Validation")
    print("=" * 60 + "\n")

    log_cpm, metadata, gene_mapping = load_data()

    # Step 1: Individual ROC
    individual_results, gene_exprs, common, y = individual_roc_analysis(
        log_cpm, metadata, gene_mapping
    )

    # Step 2: Multi-gene model
    fpr_cv, tpr_cv, auc_cv, valid_genes, fold_aucs = multi_gene_model(
        log_cpm, metadata, gene_mapping, common, y
    )

    # Step 3: Expression violin plots
    plot_expression_violin(log_cpm, metadata, gene_mapping)

    # Step 4: Diagnostic summary nomogram
    plot_diagnostic_nomogram(log_cpm, metadata, gene_mapping)

    # Plot combined ROC
    plot_multi_gene_roc(individual_results, fpr_cv, tpr_cv, auc_cv, valid_genes, fold_aucs)

    print(f"\n{'=' * 60}")
    print(f"  Phase 7 complete!")
    print(f"  Key results:")
    print(f"    - Individual gene AUCs computed")
    print(f"    - 5-gene diagnostic model: CV AUC = {auc_cv:.3f}")
    print(f"    - Cross-validation: {np.mean(fold_aucs):.3f} ± {np.std(fold_aucs):.3f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
