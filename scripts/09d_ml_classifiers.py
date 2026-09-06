"""
09d_ml_classifiers.py
======================
Advanced machine learning classification & time-dependent ROC analysis.
  1. Compare 5 ML models (LR, RF, SVM, XGBoost, MLP) with nested CV
  2. SHAP feature importance
  3. Time-dependent ROC for 1/3/5-year survival prediction
  4. Risk score model construction

Usage:
    pip install xgboost shap --break-system-packages   (if not installed)
    python scripts/09d_ml_classifiers.py

Output:
    results/ml_comparison.csv
    results/time_dependent_roc.csv
    results/risk_score_model.csv
    figures/ml_model_comparison.png
    figures/shap_importance.png
    figures/time_dependent_roc.png
    figures/risk_score_stratification.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from scipy.stats import mannwhitneyu

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test

import warnings
warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed. Run: pip install xgboost --break-system-packages")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("WARNING: shap not installed. Run: pip install shap --break-system-packages")

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
FIG_DIR = PROJECT_DIR / "figures"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.size": 10, "axes.titlesize": 12,
})

HUB_GENES = ["BIRC5", "CCNB1", "CCNB2", "CDC20", "KIF2C"]


def load_data():
    """Load and prepare data."""
    counts = pd.read_csv(DATA_DIR / "filtered_counts.csv", index_col=0)
    metadata = pd.read_csv(DATA_DIR / "metadata.csv")

    gene_mapping = None
    if (DATA_DIR / "gene_id_mapping.csv").exists():
        gene_mapping = pd.read_csv(DATA_DIR / "gene_id_mapping.csv", index_col=0)

    lib_sizes = counts.sum(axis=0)
    cpm = counts.div(lib_sizes, axis=1) * 1e6
    log_cpm = np.log2(cpm + 1)

    return log_cpm, metadata, gene_mapping


def prepare_classification_data(log_cpm, metadata, gene_mapping):
    """Prepare X, y for tumor vs normal classification."""
    if gene_mapping is not None:
        name_to_id = dict(zip(gene_mapping["gene_name"], gene_mapping.index))
    else:
        name_to_id = {}

    meta = metadata.set_index("sample_submitter_id")
    common = [s for s in log_cpm.columns if s in meta.index]

    X_list = []
    valid_genes = []
    for gene in HUB_GENES:
        gene_id = name_to_id.get(gene, gene)
        if gene_id in log_cpm.index:
            X_list.append(log_cpm.loc[gene_id, common].values)
            valid_genes.append(gene)

    X = np.column_stack(X_list)
    y = (meta.loc[common, "condition"] == "Tumor").astype(int).values

    return X, y, valid_genes, common


def compare_ml_models(X, y, valid_genes):
    """Compare 5 ML models with stratified 5-fold CV."""
    print("\n" + "=" * 60)
    print("Step 1: ML Model Comparison (5-Fold Stratified CV)")
    print("=" * 60)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=42),
        "MLP Neural Net": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                                         random_state=42, early_stopping=True),
    }

    if HAS_XGB:
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            random_state=42, eval_metric="logloss", verbosity=0
        )

    results = []

    for name, model in models.items():
        print(f"\n  {name}:")

        # Cross-validated predictions
        y_prob = cross_val_predict(model, X_scaled, y, cv=cv, method="predict_proba")[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        # Metrics
        fpr, tpr, _ = roc_curve(y, y_prob)
        roc_auc = auc(fpr, tpr)
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        prec = precision_score(y, y_pred)
        rec = recall_score(y, y_pred)

        # Per-fold AUCs
        fold_aucs = cross_val_score(model, X_scaled, y, cv=cv, scoring="roc_auc")

        results.append({
            "model": name,
            "AUC": roc_auc,
            "AUC_mean": fold_aucs.mean(),
            "AUC_std": fold_aucs.std(),
            "accuracy": acc,
            "F1": f1,
            "precision": prec,
            "recall": rec,
            "fpr": fpr,
            "tpr": tpr,
        })

        print(f"    AUC:       {roc_auc:.4f} (mean ± sd: {fold_aucs.mean():.4f} ± {fold_aucs.std():.4f})")
        print(f"    Accuracy:  {acc:.4f}")
        print(f"    F1:        {f1:.4f}")
        print(f"    Precision: {prec:.4f}  Recall: {rec:.4f}")

    # Save results
    save_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ["fpr", "tpr"]}
                            for r in results])
    save_df.to_csv(RESULTS_DIR / "ml_comparison.csv", index=False)

    return results, X_scaled, scaler


def plot_ml_comparison(results):
    """Plot ML model comparison."""
    print("\n  Plotting ML model comparison...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # (A) ROC curves
    ax = axes[0]
    colors = plt.cm.Set1(np.linspace(0, 0.8, len(results)))

    for i, res in enumerate(results):
        ax.plot(res["fpr"], res["tpr"], color=colors[i], lw=2.5,
                label=f'{res["model"]} ({res["AUC"]:.3f})')

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.3)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("(A) ROC Curves — ML Model Comparison", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_aspect("equal")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.15)

    # (B) Metrics comparison bar chart
    ax2 = axes[1]
    metrics = ["AUC", "accuracy", "F1", "precision", "recall"]
    model_names = [r["model"] for r in results]
    x = np.arange(len(metrics))
    width = 0.15

    for i, res in enumerate(results):
        vals = [res[m] for m in metrics]
        offset = (i - len(results)/2 + 0.5) * width
        ax2.bar(x + offset, vals, width, label=res["model"], color=colors[i], alpha=0.85)

    ax2.set_xticks(x)
    ax2.set_xticklabels(["AUC", "Accuracy", "F1", "Precision", "Recall"], fontsize=10)
    ax2.set_ylim(0.85, 1.01)
    ax2.set_title("(B) Performance Metrics Comparison", fontweight="bold")
    ax2.legend(fontsize=8, loc="lower left")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(True, alpha=0.15, axis="y")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "ml_model_comparison.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: figures/ml_model_comparison.png")


def shap_analysis(X_scaled, y, valid_genes):
    """SHAP feature importance analysis."""
    print("\n" + "=" * 60)
    print("Step 2: SHAP Feature Importance")
    print("=" * 60)

    if not HAS_SHAP:
        print("  SHAP not available — skipping")
        return

    # Use Random Forest for SHAP (tree-based SHAP is fast)
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_scaled, y)

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_scaled)

    # Handle different SHAP output formats
    if isinstance(shap_values, list):
        sv = shap_values[1]  # Class 1 SHAP values
    else:
        sv = shap_values

    # Ensure 2D: (n_samples, n_features)
    if sv.ndim > 2:
        # Some SHAP versions return (n_samples, n_features, n_classes, ...)
        # Take class 1 slice and flatten extra dims
        sv = sv[..., 1] if sv.shape[-1] == 2 else sv
        while sv.ndim > 2:
            sv = sv[..., -1]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # (A) SHAP feature importance bar plot
    ax = axes[0]
    mean_abs_shap = np.abs(sv).mean(axis=0).flatten()
    sorted_idx = np.argsort(mean_abs_shap)

    ax.barh(range(len(valid_genes)), mean_abs_shap[sorted_idx],
            color="#e74c3c", edgecolor="white", height=0.6)
    ax.set_yticks(range(len(valid_genes)))
    ax.set_yticklabels([valid_genes[i] for i in sorted_idx], fontsize=11)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
    ax.set_title("(A) SHAP Feature Importance", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    # (B) SHAP dependence for top gene
    ax2 = axes[1]
    top_gene_idx = sorted_idx[-1]
    top_gene = valid_genes[top_gene_idx]

    sv_col = sv[:, top_gene_idx].flatten()
    scatter = ax2.scatter(X_scaled[:, top_gene_idx], sv_col,
                          c=y, cmap="RdBu_r", alpha=0.5, s=15)
    ax2.set_xlabel(f"{top_gene} Expression (scaled)", fontsize=11)
    ax2.set_ylabel(f"SHAP Value for {top_gene}", fontsize=11)
    ax2.set_title(f"(B) SHAP Dependence — {top_gene}", fontweight="bold")
    ax2.axhline(0, color="black", lw=0.5, alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label="Class (0=Normal, 1=Tumor)")
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_importance.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: figures/shap_importance.png")


def time_dependent_roc(log_cpm, metadata, gene_mapping):
    """Time-dependent ROC for 1, 3, 5-year survival prediction."""
    print("\n" + "=" * 60)
    print("Step 3: Time-Dependent ROC (1/3/5-Year Survival)")
    print("=" * 60)

    if gene_mapping is not None:
        name_to_id = dict(zip(gene_mapping["gene_name"], gene_mapping.index))
    else:
        name_to_id = {}

    meta = metadata.set_index("sample_submitter_id")
    tumor_samples = [s for s in log_cpm.columns if s in meta.index and meta.loc[s, "condition"] == "Tumor"]

    # Build survival data
    surv_data = []
    for sample in tumor_samples:
        row = meta.loc[sample]
        vital = str(row.get("vital_status", ""))
        dtd = row.get("days_to_death", None)
        dtf = row.get("days_to_last_follow_up", None)

        if vital.lower() in ["dead", "1"]:
            time = pd.to_numeric(dtd, errors="coerce")
            event = 1
        else:
            time = pd.to_numeric(dtf, errors="coerce")
            if pd.isna(time):
                time = pd.to_numeric(dtd, errors="coerce")
            event = 0

        if pd.notna(time) and time > 0:
            time_years = time / 365.25
            gene_vals = {}
            for gene in HUB_GENES:
                gene_id = name_to_id.get(gene, gene)
                if gene_id in log_cpm.index:
                    gene_vals[gene] = log_cpm.loc[gene_id, sample]

            surv_data.append({
                "sample": sample,
                "time": time_years,
                "event": event,
                **gene_vals,
            })

    surv_df = pd.DataFrame(surv_data)
    valid_genes = [g for g in HUB_GENES if g in surv_df.columns]

    if len(surv_df) < 50:
        print(f"  Too few samples with survival data ({len(surv_df)})")
        return

    print(f"  Samples with survival: {len(surv_df)}, Events: {int(surv_df['event'].sum())}")

    # Compute risk score (weighted sum using Cox coefficients)
    cox_df = surv_df[["time", "event"] + valid_genes].dropna()

    # Z-score normalize
    for gene in valid_genes:
        cox_df[gene] = (cox_df[gene] - cox_df[gene].mean()) / (cox_df[gene].std() + 1e-10)

    # Fit multivariate Cox
    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col="time", event_col="event")

    print(f"\n  Multivariate Cox coefficients:")
    for gene in valid_genes:
        coef = cph.params_[gene]
        hr = np.exp(coef)
        p = cph.summary["p"][gene]
        print(f"    {gene:10s}: coef={coef:+.3f}  HR={hr:.3f}  p={p:.4f}")

    # Risk score = linear predictor
    risk_scores = cph.predict_partial_hazard(cox_df[valid_genes])
    cox_df["risk_score"] = risk_scores.values

    # Time-dependent ROC at 1, 3, 5 years
    time_points = [1, 3, 5]
    roc_results = []

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, t in enumerate(time_points):
        ax = axes[idx]

        # Binary outcome at time t: event before t vs alive at t
        # Exclude censored before t
        eligible = cox_df[(cox_df["time"] >= t) | (cox_df["event"] == 1)]
        eligible = eligible.copy()
        eligible["outcome_t"] = ((eligible["time"] <= t) & (eligible["event"] == 1)).astype(int)

        if eligible["outcome_t"].sum() < 5 or (1 - eligible["outcome_t"]).sum() < 5:
            ax.text(0.5, 0.5, f"Insufficient events\nat {t} years",
                   transform=ax.transAxes, ha="center")
            continue

        # ROC for risk score
        fpr_rs, tpr_rs, _ = roc_curve(eligible["outcome_t"], eligible["risk_score"])
        auc_rs = auc(fpr_rs, tpr_rs)

        ax.plot(fpr_rs, tpr_rs, color="#e74c3c", lw=2.5,
                label=f"Risk Score (AUC={auc_rs:.3f})")

        # Individual gene ROCs
        gene_colors = plt.cm.Set2(np.linspace(0, 1, len(valid_genes)))
        for gi, gene in enumerate(valid_genes):
            fpr_g, tpr_g, _ = roc_curve(eligible["outcome_t"], eligible[gene])
            auc_g = auc(fpr_g, tpr_g)
            if auc_g < 0.5:
                fpr_g, tpr_g, _ = roc_curve(eligible["outcome_t"], -eligible[gene])
                auc_g = auc(fpr_g, tpr_g)
            ax.plot(fpr_g, tpr_g, color=gene_colors[gi], lw=1.5, alpha=0.7,
                    label=f"{gene} ({auc_g:.3f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.3)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{t}-Year Survival Prediction", fontweight="bold")
        ax.legend(loc="lower right", fontsize=8)
        ax.set_aspect("equal")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, alpha=0.15)

        roc_results.append({
            "time_point": f"{t}-year",
            "risk_score_AUC": auc_rs,
            "n_eligible": len(eligible),
            "n_events": int(eligible["outcome_t"].sum()),
        })

    plt.suptitle("Time-Dependent ROC — 5-Gene Risk Score (TCGA-LUAD)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "time_dependent_roc.png", bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: figures/time_dependent_roc.png")

    roc_result_df = pd.DataFrame(roc_results)
    roc_result_df.to_csv(RESULTS_DIR / "time_dependent_roc.csv", index=False)

    # Risk score stratification (KM plot)
    plot_risk_stratification(cox_df, valid_genes)

    return roc_result_df


def plot_risk_stratification(cox_df, valid_genes):
    """KM curves stratified by risk score."""
    print("  Plotting risk score stratification...")

    median_risk = cox_df["risk_score"].median()
    cox_df["risk_group"] = np.where(cox_df["risk_score"] >= median_risk, "High Risk", "Low Risk")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # (A) KM survival by risk group
    ax = axes[0]
    kmf = KaplanMeierFitter()

    for group, color in [("High Risk", "#e74c3c"), ("Low Risk", "#3498db")]:
        mask = cox_df["risk_group"] == group
        kmf.fit(cox_df.loc[mask, "time"], cox_df.loc[mask, "event"], label=group)
        kmf.plot_survival_function(ax=ax, color=color, lw=2.5)

    # Log-rank test
    high = cox_df["risk_group"] == "High Risk"
    lr = logrank_test(cox_df.loc[high, "time"], cox_df.loc[~high, "time"],
                      cox_df.loc[high, "event"], cox_df.loc[~high, "event"])

    ax.set_title(f"(A) Risk Score Stratification (p = {lr.p_value:.2e})", fontweight="bold")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Survival Probability")
    ax.legend(fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    # (B) Risk score distribution
    ax2 = axes[1]

    # Sort by risk score
    sorted_df = cox_df.sort_values("risk_score")
    colors_scatter = ["#e74c3c" if rg == "High Risk" else "#3498db"
                      for rg in sorted_df["risk_group"]]

    ax2.scatter(range(len(sorted_df)), sorted_df["risk_score"],
                c=colors_scatter, s=8, alpha=0.6)
    ax2.axhline(median_risk, color="black", ls="--", lw=1, alpha=0.5, label="Median")
    ax2.set_xlabel("Patients (sorted by risk score)")
    ax2.set_ylabel("Risk Score")
    ax2.set_title("(B) Risk Score Distribution", fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "risk_score_stratification.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: figures/risk_score_stratification.png")

    # Save risk model
    save_cols = [c for c in ["sample", "time", "event", "risk_score", "risk_group"] if c in cox_df.columns]
    save_cols += [g for g in valid_genes if g in cox_df.columns]
    model_info = cox_df[save_cols]
    model_info.to_csv(RESULTS_DIR / "risk_score_model.csv", index=False)


def main():
    print("\n" + "=" * 60)
    print("  Phase 9D: Advanced ML & Time-Dependent ROC")
    print("=" * 60 + "\n")

    log_cpm, metadata, gene_mapping = load_data()

    # Prepare data
    X, y, valid_genes, common = prepare_classification_data(log_cpm, metadata, gene_mapping)
    print(f"  Features: {valid_genes}")
    print(f"  Samples: {len(y)} ({sum(y)} tumor, {len(y)-sum(y)} normal)")

    # Step 1: ML model comparison
    results, X_scaled, scaler = compare_ml_models(X, y, valid_genes)
    plot_ml_comparison(results)

    # Step 2: SHAP analysis
    shap_analysis(X_scaled, y, valid_genes)

    # Step 3: Time-dependent ROC
    time_dependent_roc(log_cpm, metadata, gene_mapping)

    print(f"\n{'=' * 60}")
    print(f"  Phase 9D complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
