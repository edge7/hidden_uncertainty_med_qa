#!/usr/bin/env python3
"""
Reproduce Tables from the Paper:
"Hidden-uncertainty Assessment via Non-verbalized Signatures for Medical QA"

- Table 3: ROC-AUC (unweighted macro average)
- Table 4: Acc-Cov AUC (unweighted macro average)
- Table 5: Accuracy @ Coverage levels (10%, 20%, 30%, 40%, 50%)
- Table 7: Universal vs Model-specific features (delta)
- Table 8: Error rate by disagreement level
- Figure 4: Cross-Model Transfer matrix (LR on universal features)

Data source: Zenodo (DOI: 10.5281/zenodo.18138856)
"""

import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tabulate import tabulate

ZENODO_DOI = "10.5281/zenodo.18138856"

# =============================================================================
# CONFIGURATION
# =============================================================================

MODELS = [
    "openai_gpt-oss-120b",
    "openai_gpt-oss-120b-high",
    "deepseek-ai_DeepSeek-R1-Distill-Qwen-32B",
    "Qwen_Qwen3-32B",
    "allenai_Olmo-3-32B-Think"
]

MODEL_SHORT_NAMES = {
    "openai_gpt-oss-120b": "GPT-oss-120B-medium",
    "openai_gpt-oss-120b-high": "GPT-oss-120B-high",
    "deepseek-ai_DeepSeek-R1-Distill-Qwen-32B": "DeepSeek-R1-Distill-32B",
    "Qwen_Qwen3-32B": "Qwen3-32B",
    "allenai_Olmo-3-32B-Think": "Olmo-3-32B-Think"
}

COVERAGE_LEVELS = [0.10, 0.20, 0.30, 0.40, 0.50]

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_latest_run(model_dir: Path) -> Path:
    """Get the latest run directory for a model."""
    if not model_dir.exists():
        return None
    runs = sorted([d for d in model_dir.iterdir() if d.is_dir()], reverse=True)
    return runs[0] if runs else None


def compute_roc_auc_macro(df: pd.DataFrame, prob_col: str) -> float:
    """Compute macro-averaged (unweighted) ROC-AUC across datasets."""
    aucs = []
    for ds in df['dataset'].unique():
        mask = df['dataset'] == ds
        y_true = df.loc[mask, 'y_true'].values
        y_prob = df.loc[mask, prob_col].values
        if len(np.unique(y_true)) >= 2:
            aucs.append(roc_auc_score(y_true, y_prob))
    return np.mean(aucs) if aucs else np.nan


def compute_acc_cov_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute AUC of Accuracy-Coverage curve."""
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    sorted_indices = np.argsort(-y_prob)
    y_true_sorted = y_true[sorted_indices]

    n = len(y_true)
    cumsum = np.cumsum(y_true_sorted)
    coverages = np.arange(1, n + 1) / n
    accuracies = cumsum / np.arange(1, n + 1)

    # np.trapz deprecated in numpy 2.x, use np.trapezoid
    if hasattr(np, 'trapezoid'):
        auc = np.trapezoid(accuracies, coverages)
    else:
        auc = np.trapz(accuracies, coverages)
    return auc


def compute_acc_cov_auc_macro(df: pd.DataFrame, prob_col: str) -> float:
    """Compute macro-averaged Acc-Cov AUC across datasets."""
    aucs = []
    for ds in df['dataset'].unique():
        mask = df['dataset'] == ds
        y_true = df.loc[mask, 'y_true'].values
        y_prob = df.loc[mask, prob_col].values
        aucs.append(compute_acc_cov_auc(y_true, y_prob))
    return np.mean(aucs) if aucs else np.nan


def get_accuracy_at_coverage(y_true: np.ndarray, confidence: np.ndarray, coverage: float) -> float:
    """Get accuracy when keeping only top-coverage% most confident predictions.

    Note: Uses kind='stable' for reproducibility when there are tied confidence values.
    This ensures consistent results across numpy versions (1.26.x vs 2.x have different
    default sorting behavior for ties).
    """
    n = len(y_true)
    n_keep = max(1, int(n * coverage))
    # kind='stable' ensures reproducibility when there are tied confidence values
    idx = np.argsort(confidence, kind='stable')[::-1][:n_keep]
    return np.mean(y_true[idx])


def get_accuracy_at_coverage_macro(df: pd.DataFrame, conf_col: str, coverage: float) -> float:
    """Compute macro-averaged accuracy at given coverage level."""
    accs = []
    for ds in df['dataset'].unique():
        mask = df['dataset'] == ds
        sub = df[mask]
        if len(sub) >= 10:
            y_true = sub['y_true'].values
            conf = sub[conf_col].values
            accs.append(get_accuracy_at_coverage(y_true, conf, coverage))
    return np.mean(accs) if accs else np.nan


# =============================================================================
# TABLE 3: ROC-AUC (Unweighted Macro Average)
# =============================================================================

def table_3_roc_auc(data_dir: Path) -> pd.DataFrame:
    """
    Reproduce Table 3: ROC-AUC (unweighted macro average)

    Columns: Baselines (AnsProb, Softmax, SV) + Trust-Abstain (RF, LR, MLP) x (No SV, With SV)
    """
    results = []

    for model in MODELS:
        model_dir = data_dir / "results" / model
        run_dir = get_latest_run(model_dir)
        if run_dir is None:
            print(f"  Warning: No data for {model}")
            continue

        # Load predictions
        with open(run_dir / "predictions_no_selfverb.json") as f:
            pred_no_sv = pd.DataFrame(json.load(f))
        with open(run_dir / "predictions_with_selfverb.json") as f:
            pred_with_sv = pd.DataFrame(json.load(f))

        row = {"Model": MODEL_SHORT_NAMES[model]}

        # Baselines
        row["AnsProb"] = compute_roc_auc_macro(pred_no_sv, "ans_prob")
        row["Softmax"] = compute_roc_auc_macro(pred_no_sv, "ans_softmax_prob")
        pred_no_sv["sv_norm"] = pred_no_sv["verbalized_confidence"] / 100.0
        row["SV"] = compute_roc_auc_macro(pred_no_sv, "sv_norm")

        # Trust-Abstain (No SV)
        row["RF_noSV"] = compute_roc_auc_macro(pred_no_sv, "rf_prob")
        row["LR_noSV"] = compute_roc_auc_macro(pred_no_sv, "lr_prob")
        row["MLP_noSV"] = compute_roc_auc_macro(pred_no_sv, "mlp_prob")

        # Trust-Abstain (With SV)
        row["RF_wSV"] = compute_roc_auc_macro(pred_with_sv, "rf_prob")
        row["LR_wSV"] = compute_roc_auc_macro(pred_with_sv, "lr_prob")
        row["MLP_wSV"] = compute_roc_auc_macro(pred_with_sv, "mlp_prob")

        results.append(row)

    return pd.DataFrame(results)


# =============================================================================
# TABLE 4: Acc-Cov AUC (Unweighted Macro Average)
# =============================================================================

def table_4_acc_cov_auc(data_dir: Path) -> pd.DataFrame:
    """
    Reproduce Table 4: Accuracy-Coverage AUC (unweighted macro average)
    """
    results = []

    for model in MODELS:
        model_dir = data_dir / "results" / model
        run_dir = get_latest_run(model_dir)
        if run_dir is None:
            continue

        with open(run_dir / "predictions_no_selfverb.json") as f:
            pred_no_sv = pd.DataFrame(json.load(f))
        with open(run_dir / "predictions_with_selfverb.json") as f:
            pred_with_sv = pd.DataFrame(json.load(f))

        row = {"Model": MODEL_SHORT_NAMES[model]}

        # Baselines
        row["AnsProb"] = compute_acc_cov_auc_macro(pred_no_sv, "ans_prob")
        row["Softmax"] = compute_acc_cov_auc_macro(pred_no_sv, "ans_softmax_prob")
        pred_no_sv["sv_norm"] = pred_no_sv["verbalized_confidence"] / 100.0
        row["SV"] = compute_acc_cov_auc_macro(pred_no_sv, "sv_norm")

        # Trust-Abstain (No SV)
        row["RF_noSV"] = compute_acc_cov_auc_macro(pred_no_sv, "rf_prob")
        row["LR_noSV"] = compute_acc_cov_auc_macro(pred_no_sv, "lr_prob")
        row["MLP_noSV"] = compute_acc_cov_auc_macro(pred_no_sv, "mlp_prob")

        # Trust-Abstain (With SV)
        row["RF_wSV"] = compute_acc_cov_auc_macro(pred_with_sv, "rf_prob")
        row["LR_wSV"] = compute_acc_cov_auc_macro(pred_with_sv, "lr_prob")
        row["MLP_wSV"] = compute_acc_cov_auc_macro(pred_with_sv, "mlp_prob")

        results.append(row)

    return pd.DataFrame(results)


# =============================================================================
# TABLE 5: Accuracy @ Coverage Levels
# =============================================================================

def table_5_accuracy_at_coverage(data_dir: Path) -> pd.DataFrame:
    """
    Reproduce Table 5: Accuracy at coverage levels (10%, 20%, 30%, 40%, 50%)

    Compares Self-Verbalized (SV) vs Trust-Abstain (LR with SV)
    """
    results = []

    for model in MODELS:
        model_dir = data_dir / "results" / model
        run_dir = get_latest_run(model_dir)
        if run_dir is None:
            continue

        with open(run_dir / "predictions_with_selfverb.json") as f:
            pred = pd.DataFrame(json.load(f))

        pred["sv_norm"] = pred["verbalized_confidence"] / 100.0

        row = {"Model": MODEL_SHORT_NAMES[model]}

        for cov in COVERAGE_LEVELS:
            cov_pct = int(cov * 100)
            row[f"SV_{cov_pct}%"] = get_accuracy_at_coverage_macro(pred, "sv_norm", cov)
            row[f"TA_{cov_pct}%"] = get_accuracy_at_coverage_macro(pred, "lr_prob", cov)
            row[f"Δ_{cov_pct}%"] = row[f"TA_{cov_pct}%"] - row[f"SV_{cov_pct}%"]

        results.append(row)

    return pd.DataFrame(results)


# =============================================================================
# TABLE 7: Universal vs Model-Specific Features
# =============================================================================

def table_7_universal_vs_specific(data_dir: Path) -> pd.DataFrame:
    """
    Reproduce Table 7: Performance change when using universal (12) features
    instead of model-specific Boruta-SHAP selected features.

    Compares results/ (model-specific) vs results_common/ (universal)
    """
    results = []

    for model in MODELS:
        # Model-specific (from results/)
        specific_dir = data_dir / "results" / model
        specific_run = get_latest_run(specific_dir)

        # Universal (from results_common/)
        universal_dir = data_dir / "results_common" / model
        universal_run = get_latest_run(universal_dir)

        if specific_run is None or universal_run is None:
            continue

        # Load predictions
        with open(specific_run / "predictions_with_selfverb.json") as f:
            pred_specific = pd.DataFrame(json.load(f))
        with open(universal_run / "predictions.json") as f:
            pred_universal = pd.DataFrame(json.load(f))

        row = {"Model": MODEL_SHORT_NAMES[model]}

        # ROC-AUC delta
        for clf in ["rf", "lr", "mlp"]:
            specific_auc = compute_roc_auc_macro(pred_specific, f"{clf}_prob")
            universal_auc = compute_roc_auc_macro(pred_universal, f"{clf}_prob")
            delta_pct = (universal_auc - specific_auc) / specific_auc * 100
            row[f"Δ_ROC_{clf.upper()}"] = delta_pct

        # Acc-Cov AUC delta
        for clf in ["rf", "lr", "mlp"]:
            specific_auc = compute_acc_cov_auc_macro(pred_specific, f"{clf}_prob")
            universal_auc = compute_acc_cov_auc_macro(pred_universal, f"{clf}_prob")
            delta_pct = (universal_auc - specific_auc) / specific_auc * 100
            row[f"Δ_AccCov_{clf.upper()}"] = delta_pct

        results.append(row)

    return pd.DataFrame(results)


# =============================================================================
# TABLE 8: Error Rate by Disagreement Level
# =============================================================================

def table_8_disagreement(data_dir: Path, threshold: float = 0.3) -> pd.DataFrame:
    """
    Reproduce Table 8: Error rate stratified by disagreement magnitude.

    Disagreement = |classifier_prob - self_verbalized_prob|
    High disagreement (d > threshold) indicates hidden uncertainty.
    """
    results = []

    for model in MODELS:
        model_dir = data_dir / "results" / model
        run_dir = get_latest_run(model_dir)
        if run_dir is None:
            continue

        with open(run_dir / "predictions_with_selfverb.json") as f:
            pred = pd.DataFrame(json.load(f))

        # Normalize self-verbalized to [0,1]
        pred["sv_norm"] = pred["verbalized_confidence"] / 100.0

        # Calculate disagreement (using RF calibrated as the classifier)
        pred["disagreement"] = np.abs(pred["rf_prob_calibrated"] - pred["sv_norm"])

        # Split by disagreement level
        low_mask = pred["disagreement"] <= threshold
        high_mask = pred["disagreement"] > threshold

        # Compute error rates per dataset, then average (unweighted)
        low_errors, high_errors = [], []
        for ds in pred["dataset"].unique():
            ds_mask = pred["dataset"] == ds

            low_ds = pred[ds_mask & low_mask]
            high_ds = pred[ds_mask & high_mask]

            if len(low_ds) > 0:
                low_errors.append(1 - low_ds["y_true"].mean())
            if len(high_ds) > 0:
                high_errors.append(1 - high_ds["y_true"].mean())

        row = {
            "Model": MODEL_SHORT_NAMES[model],
            "Low_d (≤0.3)": np.mean(low_errors) * 100 if low_errors else np.nan,
            "High_d (>0.3)": np.mean(high_errors) * 100 if high_errors else np.nan,
        }
        row["Ratio"] = row["High_d (>0.3)"] / row["Low_d (≤0.3)"] if row["Low_d (≤0.3)"] > 0 else np.nan

        results.append(row)

    return pd.DataFrame(results)


# =============================================================================
# FIGURE 4: Cross-Model Transfer Matrix
# =============================================================================

def cross_model_predict_unweighted(source_data: dict, target_data: dict, classifier_type: str = 'lr'):
    """
    Use models trained on source LLM to predict on target LLM's test sets.
    Returns UNWEIGHTED average of per-dataset metrics.
    """
    source_models = source_data[classifier_type]
    source_scalers = source_data['scalers']
    target_test_sets = target_data['test_sets']
    source_feature_names = source_data['feature_names']

    dataset_rocs = []
    dataset_acc_covs = []

    for dataset_name, test_set in target_test_sets.items():
        if dataset_name not in source_models:
            continue

        X_test = test_set['X_test']
        if hasattr(X_test, 'values'):
            X_test = X_test.values
        X_test = np.asarray(X_test)

        y_test = test_set['y_test']
        if hasattr(y_test, 'values'):
            y_test = y_test.values
        y_test = np.asarray(y_test)

        target_features = test_set['feature_names']

        # Match feature order (source features -> target test set columns)
        feature_indices = [target_features.index(f) for f in source_feature_names]
        X_test_reordered = X_test[:, feature_indices]

        # RF uses unscaled data, LR/MLP use scaled
        if classifier_type == 'rf':
            X_for_predict = X_test_reordered
        else:
            source_scaler = source_scalers[dataset_name]
            X_for_predict = source_scaler.transform(X_test_reordered)

        model = source_models[dataset_name]
        y_proba = model.predict_proba(X_for_predict)[:, 1]

        if len(np.unique(y_test)) >= 2:
            roc = roc_auc_score(y_test, y_proba)
            dataset_rocs.append(roc)

        acc_cov = compute_acc_cov_auc(y_test, y_proba)
        dataset_acc_covs.append(acc_cov)

    return np.mean(dataset_rocs), np.mean(dataset_acc_covs)


def load_model_data_for_transfer(model_dir: Path) -> dict:
    """Load all data needed for cross-model transfer."""
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)

    with open(model_dir / "models_lr_calibrated.pkl", 'rb') as f:
        lr_data = pickle.load(f)
    with open(model_dir / "scalers.pkl", 'rb') as f:
        scalers = pickle.load(f)
    with open(model_dir / "test_sets.pkl", 'rb') as f:
        test_sets = pickle.load(f)

    result = {
        'lr': lr_data['models'],
        'scalers': scalers,
        'test_sets': test_sets,
        'feature_names': lr_data['feature_names'],
    }

    # Try to load RF (may fail due to sklearn version)
    try:
        with open(model_dir / "models_rf_calibrated.pkl", 'rb') as f:
            rf_data = pickle.load(f)
        result['rf'] = rf_data['models']
    except Exception:
        pass

    # Try to load MLP (may fail due to custom modules)
    try:
        with open(model_dir / "models_mlp_calibrated.pkl", 'rb') as f:
            mlp_data = pickle.load(f)
        result['mlp'] = mlp_data['models']
    except Exception:
        pass

    return result


def figure_4_cross_model_transfer(data_dir: Path, classifier_type: str = 'lr') -> tuple:
    """
    Reproduce Figure 4: Cross-Model Transfer Matrix

    Returns:
        roc_matrix: ROC-AUC for each source->target pair
        delta_roc_matrix: Delta from native (diagonal) performance
    """
    # Load all model data from results_common/
    model_data = {}
    for model in MODELS:
        model_dir = data_dir / "results_common" / model
        run_dir = get_latest_run(model_dir)
        if run_dir is not None:
            model_data[model] = load_model_data_for_transfer(run_dir)

    n = len(MODELS)
    roc_matrix = np.zeros((n, n))
    acc_cov_matrix = np.zeros((n, n))

    for i, source in enumerate(MODELS):
        for j, target in enumerate(MODELS):
            if source not in model_data or target not in model_data:
                roc_matrix[i, j] = np.nan
                acc_cov_matrix[i, j] = np.nan
                continue

            roc, acc_cov = cross_model_predict_unweighted(
                model_data[source],
                model_data[target],
                classifier_type
            )
            roc_matrix[i, j] = roc
            acc_cov_matrix[i, j] = acc_cov

    # Delta from native (diagonal)
    native_roc = np.diag(roc_matrix)
    native_acc = np.diag(acc_cov_matrix)

    delta_roc = roc_matrix - native_roc  # Each column shows delta from target's native
    delta_acc = acc_cov_matrix - native_acc

    return roc_matrix, delta_roc, acc_cov_matrix, delta_acc


# =============================================================================
# DOWNLOAD
# =============================================================================

def download_data(output_dir: Path) -> bool:
    """
    Download data from Zenodo using zenodo_get.

    Install with: pip install zenodo_get
    """
    try:
        import zenodo_get
    except ImportError:
        print("Error: zenodo_get not installed.")
        print("Install with: pip install 'hidden-uncertainty-med-qa[download]'")
        print("Or directly: pip install zenodo_get")
        return False

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading data from Zenodo (DOI: {ZENODO_DOI})...")
    print(f"Output directory: {output_dir}")
    print("This may take a while (~23GB)...\n")

    # Use zenodo_get CLI
    try:
        subprocess.run(
            ["zenodo_get", ZENODO_DOI, "-o", str(output_dir)],
            check=True
        )
        print("\nDownload complete!")
        print(f"Data extracted to: {output_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Download failed: {e}")
        return False


# =============================================================================
# MAIN
# =============================================================================

def print_table(df: pd.DataFrame, title: str, float_fmt: str = ".3f", tablefmt: str = "simple"):
    """Pretty print a DataFrame using tabulate."""
    print(f"\n{'='*80}")
    print(f"{title}")
    print('='*80)

    # Format floats
    df_formatted = df.copy()
    for col in df_formatted.columns:
        if df_formatted[col].dtype in ['float64', 'float32']:
            df_formatted[col] = df_formatted[col].apply(
                lambda x: f"{x:{float_fmt}}" if pd.notna(x) else "N/A"
            )

    print(tabulate(df_formatted, headers='keys', tablefmt=tablefmt, showindex=False))
    print()


def print_matrix(matrix: np.ndarray, title: str, labels: list, show_sign: bool = True):
    """Pretty print a matrix with labels using tabulate."""
    print(f"\n{'='*80}")
    print(f"{title}")
    print('='*80)

    short_labels = [MODEL_SHORT_NAMES[m].split('-')[0][:12] for m in labels]

    # Build table data
    table_data = []
    for i, (row, src_lbl) in enumerate(zip(matrix, short_labels)):
        row_data = [src_lbl]
        for j, val in enumerate(row):
            if i == j:
                row_data.append("—")
            elif np.isnan(val):
                row_data.append("N/A")
            else:
                if show_sign:
                    row_data.append(f"{val:+.4f}")
                else:
                    row_data.append(f"{val:.4f}")
        table_data.append(row_data)

    headers = ["Source→Target"] + short_labels
    print(tabulate(table_data, headers=headers, tablefmt="simple"))
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce paper tables from Zenodo data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download data from Zenodo
  python reproduce_tables.py --download

  # Reproduce all tables
  python reproduce_tables.py --data-dir ./data

  # Reproduce specific table
  python reproduce_tables.py --data-dir ./data --table 5
"""
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download data from Zenodo to --data-dir"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data"),
        help="Directory containing results/ and results_common/ (default: ./data)"
    )
    parser.add_argument(
        "--table",
        type=str,
        choices=["3", "4", "5", "7", "8", "fig4", "all"],
        default="all",
        help="Which table to reproduce (default: all)"
    )
    args = parser.parse_args()

    data_dir = args.data_dir

    # Download mode
    if args.download:
        success = download_data(data_dir)
        if not success:
            sys.exit(1)
        return

    # Check data exists
    if not (data_dir / "results").exists():
        print(f"Error: {data_dir / 'results'} not found")
        print(f"\nTo download the data, run:")
        print(f"  python {sys.argv[0]} --download --data-dir {data_dir}")
        sys.exit(1)

    print(f"Data directory: {data_dir}")
    print(f"Available models: {len(list((data_dir / 'results').iterdir()))} in results/")

    tables_to_run = ["3", "4", "5", "7", "8", "fig4"] if args.table == "all" else [args.table]

    if "3" in tables_to_run:
        df = table_3_roc_auc(data_dir)
        print_table(df, "TABLE 3: ROC-AUC (Unweighted Macro Average)")

    if "4" in tables_to_run:
        df = table_4_acc_cov_auc(data_dir)
        print_table(df, "TABLE 4: Acc-Cov AUC (Unweighted Macro Average)")

    if "5" in tables_to_run:
        df = table_5_accuracy_at_coverage(data_dir)
        print_table(df, "TABLE 5: Accuracy @ Coverage (SV vs Trust-Abstain LR)", float_fmt=".1%")

    if "7" in tables_to_run:
        df = table_7_universal_vs_specific(data_dir)
        print_table(df, "TABLE 7: Universal vs Model-Specific (% change)", float_fmt=".2f")

    if "8" in tables_to_run:
        df = table_8_disagreement(data_dir)
        print_table(df, "TABLE 8: Error Rate (%) by Disagreement Level", float_fmt=".1f")

    if "fig4" in tables_to_run:
        roc_matrix, delta_roc, acc_matrix, delta_acc = figure_4_cross_model_transfer(data_dir, "lr")
        print_matrix(delta_roc, "FIGURE 4: Cross-Model Transfer - ΔROC-AUC (LR)", MODELS, show_sign=True)
        print_matrix(delta_acc, "FIGURE 4: Cross-Model Transfer - ΔAcc-Cov AUC (LR)", MODELS, show_sign=True)


if __name__ == "__main__":
    main()
