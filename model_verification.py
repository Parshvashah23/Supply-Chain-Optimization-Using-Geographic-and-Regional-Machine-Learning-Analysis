"""
==========================================================================
Model Efficiency Verification Suite
DataCo Supply Chain Dataset
==========================================================================

Covers:
  1.  Cross-Validation  - TimeSeriesSplit (forecasting), StratifiedKFold (classifier)
  2.  Overfitting Check - Train vs Test metric comparison
  3.  Residual Analysis - distribution, Q-Q plot, residuals vs fitted
  4.  Predicted vs Actual plot
  5.  MAPE              - Mean Absolute Percentage Error
  6.  Precision-Recall Curve
  7.  Calibration Curve - classifier probability reliability
  8.  Learning Curve    - bias-variance diagnosis
  9.  Permutation Importance - model-agnostic feature importance
  10. Regional Model Stability - consistency across regions
==========================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

# Fix Windows console encoding for Unicode characters
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from sklearn.model_selection import (
    TimeSeriesSplit, StratifiedKFold, learning_curve, cross_val_score
)
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    precision_recall_curve, roc_auc_score, f1_score,
    average_precision_score
)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
import joblib
import json

PLOT_DIR  = 'plots/verification'
MODEL_DIR = 'models/regional_forecast'


def ensure_dir():
    os.makedirs(PLOT_DIR, exist_ok=True)


def mape(y_true, y_pred, eps=1e-8):
    """Mean Absolute Percentage Error (excludes near-zero actuals)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = np.abs(y_true) > eps
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


# ─────────────────────────────────────────────────────────────────────────────
# 1. CROSS-VALIDATION  (Regression – TimeSeriesSplit)
# ─────────────────────────────────────────────────────────────────────────────
def cv_forecast(model_class, model_kwargs, df_ts, feature_cols,
                target='Sales', n_splits=5):
    """
    Walk-forward time-series cross-validation for the demand forecasting model.
    Returns per-fold and aggregate metrics.
    """
    print("\n--- [1] Time-Series Cross-Validation (Forecasting) ---")
    ensure_dir()

    df_clean = df_ts.dropna(subset=feature_cols + [target]).sort_values('Date').reset_index(drop=True)
    X = df_clean[feature_cols].values
    y = df_clean[target].values

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = model_class(**model_kwargs)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae_  = mean_absolute_error(y_test, y_pred)
        rmse_ = np.sqrt(mean_squared_error(y_test, y_pred))
        r2_   = r2_score(y_test, y_pred)
        mape_ = mape(y_test, y_pred)

        fold_metrics.append({'Fold': fold, 'MAE': mae_, 'RMSE': rmse_,
                              'R2': r2_, 'MAPE%': mape_,
                              'Train_n': len(train_idx), 'Test_n': len(test_idx)})
        print(f"  Fold {fold}: MAE={mae_:,.1f}  RMSE={rmse_:,.1f}  R2={r2_:.4f}  MAPE={mape_:.2f}%")

    result_df = pd.DataFrame(fold_metrics)

    print(f"\n  CV Summary ({n_splits} folds):")
    print(f"    MAE  mean={result_df['MAE'].mean():,.1f}  std={result_df['MAE'].std():,.1f}")
    print(f"    RMSE mean={result_df['RMSE'].mean():,.1f}  std={result_df['RMSE'].std():,.1f}")
    print(f"    R2   mean={result_df['R2'].mean():.4f}  std={result_df['R2'].std():.4f}")
    print(f"    MAPE mean={result_df['MAPE%'].mean():.2f}%  std={result_df['MAPE%'].std():.2f}%")

    # Plot fold metrics
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, col, color in zip(axes, ['MAE', 'RMSE', 'R2', 'MAPE%'],
                              ['steelblue', 'darkorange', 'mediumseagreen', 'crimson']):
        axes_i = ax
        vals = result_df[col]
        axes_i.bar(result_df['Fold'], vals, color=color, edgecolor='gray')
        axes_i.axhline(vals.mean(), color='black', linestyle='--', linewidth=1.5,
                       label=f'Mean={vals.mean():.2f}')
        axes_i.set_title(col, fontweight='bold')
        axes_i.set_xlabel('Fold')
        axes_i.legend(fontsize=8)
    plt.suptitle('Time-Series CV — Demand Forecasting', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/cv1_forecast_folds.png', dpi=150)
    plt.close()
    print("  Saved: cv1_forecast_folds.png")

    return result_df


# ─────────────────────────────────────────────────────────────────────────────
# 2. CROSS-VALIDATION  (Classification – StratifiedKFold)
# ─────────────────────────────────────────────────────────────────────────────
def cv_classifier(model_class, model_kwargs, X, y, n_splits=5):
    """
    Stratified K-Fold cross-validation for the late delivery classifier.
    """
    print("\n--- [2] Stratified K-Fold Cross-Validation (Classifier) ---")
    ensure_dir()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        model = model_class(**model_kwargs)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        y_proba = model.predict_proba(X_te)[:, 1]

        f1_   = f1_score(y_te, y_pred)
        auc_  = roc_auc_score(y_te, y_proba)
        ap_   = average_precision_score(y_te, y_proba)
        fold_metrics.append({'Fold': fold, 'F1': f1_, 'ROC_AUC': auc_, 'Avg_Precision': ap_})
        print(f"  Fold {fold}: F1={f1_:.4f}  AUC={auc_:.4f}  AP={ap_:.4f}")

    result_df = pd.DataFrame(fold_metrics)
    print(f"\n  CV Summary ({n_splits} folds):")
    for col in ['F1', 'ROC_AUC', 'Avg_Precision']:
        print(f"    {col}: mean={result_df[col].mean():.4f}  std={result_df[col].std():.4f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col, color in zip(axes, ['F1', 'ROC_AUC', 'Avg_Precision'],
                              ['steelblue', 'darkorange', 'mediumseagreen']):
        vals = result_df[col]
        ax.bar(result_df['Fold'], vals, color=color, edgecolor='gray')
        ax.axhline(vals.mean(), color='black', linestyle='--', linewidth=1.5,
                   label=f'Mean={vals.mean():.4f}')
        ax.set_ylim([max(0, vals.min() - 0.05), 1.02])
        ax.set_title(col, fontweight='bold')
        ax.set_xlabel('Fold')
        ax.legend(fontsize=8)
    plt.suptitle('Stratified K-Fold CV — Late Delivery Classifier', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/cv2_classifier_folds.png', dpi=150)
    plt.close()
    print("  Saved: cv2_classifier_folds.png")

    return result_df


# ─────────────────────────────────────────────────────────────────────────────
# 3. OVERFITTING CHECK — Train vs Test
# ─────────────────────────────────────────────────────────────────────────────
def overfitting_check(model, X_train, y_train, X_test, y_test, task='regression'):
    """
    Compare train and test performance to detect overfitting.
    """
    print("\n--- [3] Overfitting Check (Train vs Test) ---")

    if task == 'regression':
        y_train_pred = model.predict(X_train)
        y_test_pred  = model.predict(X_test)

        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae  = mean_absolute_error(y_test,  y_test_pred)
        train_r2  = r2_score(y_train, y_train_pred)
        test_r2   = r2_score(y_test,  y_test_pred)
        train_mape_ = mape(y_train, y_train_pred)
        test_mape_  = mape(y_test,  y_test_pred)

        print(f"  {'Metric':<12} {'Train':>12} {'Test':>12} {'Gap':>10}")
        print(f"  {'-'*46}")
        print(f"  {'MAE':<12} {train_mae:>12,.2f} {test_mae:>12,.2f} {test_mae-train_mae:>+10,.2f}")
        print(f"  {'R2':<12} {train_r2:>12.4f} {test_r2:>12.4f} {test_r2-train_r2:>+10.4f}")
        print(f"  {'MAPE%':<12} {train_mape_:>12.2f} {test_mape_:>12.2f} {test_mape_-train_mape_:>+10.2f}")

        gap_r2 = train_r2 - test_r2
        if gap_r2 > 0.15:
            print("  WARNING: Large R2 gap suggests OVERFITTING")
        elif gap_r2 < 0.02:
            print("  OK: Minimal overfitting detected")
        else:
            print("  MILD overfitting present (common for large ML models)")

        return {'train': {'MAE': train_mae, 'R2': train_r2, 'MAPE': train_mape_},
                'test':  {'MAE': test_mae,  'R2': test_r2,  'MAPE': test_mape_}}

    else:  # classification
        y_train_pred = model.predict(X_train)
        y_test_pred  = model.predict(X_test)

        train_f1 = f1_score(y_train, y_train_pred)
        test_f1  = f1_score(y_test,  y_test_pred)
        train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
        test_auc  = roc_auc_score(y_test,  model.predict_proba(X_test)[:, 1])

        print(f"  {'Metric':<12} {'Train':>12} {'Test':>12} {'Gap':>10}")
        print(f"  {'-'*46}")
        print(f"  {'F1':<12} {train_f1:>12.4f} {test_f1:>12.4f} {test_f1-train_f1:>+10.4f}")
        print(f"  {'ROC-AUC':<12} {train_auc:>12.4f} {test_auc:>12.4f} {test_auc-train_auc:>+10.4f}")

        gap = train_f1 - test_f1
        if gap > 0.05:
            print("  WARNING: Large F1 gap suggests OVERFITTING")
        else:
            print("  OK: Low overfitting gap")

        return {'train': {'F1': train_f1, 'AUC': train_auc},
                'test':  {'F1': test_f1,  'AUC': test_auc}}


# ─────────────────────────────────────────────────────────────────────────────
# 4. RESIDUAL ANALYSIS  (Regression)
# ─────────────────────────────────────────────────────────────────────────────
def residual_analysis(y_true, y_pred, label='Demand Forecasting'):
    """
    Residual distribution, Q-Q plot, and residuals vs fitted values.
    """
    print("\n--- [4] Residual Analysis ---")
    ensure_dir()

    residuals = np.array(y_true) - np.array(y_pred)

    fig = plt.figure(figsize=(18, 5))
    gs  = gridspec.GridSpec(1, 4, figure=fig)

    # (a) Residual histogram
    ax1 = fig.add_subplot(gs[0])
    ax1.hist(residuals, bins=60, color='steelblue', edgecolor='white', alpha=0.8)
    ax1.axvline(0, color='red', linestyle='--')
    ax1.set_title('Residual Distribution', fontweight='bold')
    ax1.set_xlabel('Residual')
    ax1.set_ylabel('Count')

    # (b) Q-Q plot
    ax2 = fig.add_subplot(gs[1])
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist='norm')
    ax2.scatter(osm, osr, s=4, alpha=0.4, color='steelblue')
    ax2.plot(osm, slope * np.array(osm) + intercept, 'r-', linewidth=1.5)
    ax2.set_title(f'Q-Q Plot (r={r:.3f})', fontweight='bold')
    ax2.set_xlabel('Theoretical Quantiles')
    ax2.set_ylabel('Sample Quantiles')

    # (c) Residuals vs Fitted
    ax3 = fig.add_subplot(gs[2])
    ax3.scatter(y_pred, residuals, s=4, alpha=0.3, color='darkorange')
    ax3.axhline(0, color='red', linestyle='--')
    ax3.set_title('Residuals vs Fitted', fontweight='bold')
    ax3.set_xlabel('Fitted Values ($)')
    ax3.set_ylabel('Residual')

    # (d) Predicted vs Actual
    ax4 = fig.add_subplot(gs[3])
    lims = [min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))]
    ax4.scatter(y_true, y_pred, s=4, alpha=0.3, color='mediumseagreen')
    ax4.plot(lims, lims, 'r--', linewidth=1.5)
    ax4.set_title(f'Predicted vs Actual\n(R²={r2_score(y_true, y_pred):.4f})', fontweight='bold')
    ax4.set_xlabel('Actual ($)')
    ax4.set_ylabel('Predicted ($)')

    plt.suptitle(f'Residual Analysis — {label}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/cv3_residual_analysis.png', dpi=150)
    plt.close()
    print(f"  Residual mean:  {residuals.mean():.2f}  (want ~0)")
    print(f"  Residual std:   {residuals.std():.2f}")
    print(f"  Residual skew:  {stats.skew(residuals):.3f}  (want ~0)")
    print(f"  Residual kurt:  {stats.kurtosis(residuals):.3f}")
    print("  Saved: cv3_residual_analysis.png")

    # Normality test (Shapiro-Wilk on a sample)
    sample = np.random.RandomState(42).choice(residuals, min(5000, len(residuals)), replace=False)
    _, p_shapiro = stats.shapiro(sample)
    print(f"  Shapiro-Wilk p-value: {p_shapiro:.4e} "
          f"({'residuals NOT normal' if p_shapiro < 0.05 else 'residuals approximately normal'})")


# ─────────────────────────────────────────────────────────────────────────────
# 5. PRECISION-RECALL CURVE
# ─────────────────────────────────────────────────────────────────────────────
def plot_precision_recall(y_test, y_proba, label='Global Classifier'):
    """Precision-Recall curve with Average Precision score."""
    print("\n--- [5] Precision-Recall Curve ---")
    ensure_dir()

    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    baseline = np.mean(y_test)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, 'b-', linewidth=2, label=f'PR Curve (AP={ap:.4f})')
    ax.axhline(baseline, color='red', linestyle='--', label=f'Baseline (always predict positive, P={baseline:.3f})')
    ax.fill_between(recall, precision, alpha=0.1, color='blue')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve — {label}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/cv4_precision_recall_curve.png', dpi=150)
    plt.close()
    print(f"  Average Precision: {ap:.4f}  (baseline={baseline:.3f})")
    print("  Saved: cv4_precision_recall_curve.png")

    # Optimal threshold by F1
    f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
    best_idx = np.argmax(f1_scores)
    print(f"  Optimal threshold: {thresholds[best_idx]:.4f}  "
          f"(F1={f1_scores[best_idx]:.4f}, P={precision[best_idx]:.4f}, R={recall[best_idx]:.4f})")


# ─────────────────────────────────────────────────────────────────────────────
# 6. CALIBRATION CURVE
# ─────────────────────────────────────────────────────────────────────────────
def plot_calibration(y_test, y_proba, label='Global Classifier', n_bins=10):
    """Plot calibration curve to check probability reliability."""
    print("\n--- [6] Calibration Curve ---")
    ensure_dir()

    fraction_of_positives, mean_predicted = calibration_curve(y_test, y_proba, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax.plot(mean_predicted, fraction_of_positives, 'b-o', linewidth=2, label='Model')
    ax.fill_between(mean_predicted, mean_predicted, fraction_of_positives, alpha=0.1, color='blue')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title(f'Calibration Curve — {label}', fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/cv5_calibration_curve.png', dpi=150)
    plt.close()

    # Calibration error
    cal_error = np.mean(np.abs(fraction_of_positives - mean_predicted))
    print(f"  Mean Calibration Error: {cal_error:.4f}  (lower = better calibrated)")
    print("  Saved: cv5_calibration_curve.png")


# ─────────────────────────────────────────────────────────────────────────────
# 7. LEARNING CURVE
# ─────────────────────────────────────────────────────────────────────────────
def plot_learning_curve(model, X, y, task='regression', cv=5, label='Model'):
    """
    Learning curve to diagnose bias (underfitting) vs variance (overfitting).
    Uses a sample if dataset is very large to keep it fast.
    """
    print("\n--- [7] Learning Curve ---")
    ensure_dir()

    # Cap sample size to avoid very long runtimes
    max_samples = min(5000, len(X))
    idx = np.random.RandomState(42).choice(len(X), max_samples, replace=False)
    X_s = X[idx] if isinstance(X, np.ndarray) else X.iloc[idx].values
    y_s = y[idx] if isinstance(y, np.ndarray) else y.iloc[idx].values

    scoring = 'r2' if task == 'regression' else 'f1'
    train_sizes = np.linspace(0.1, 1.0, 8)

    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X_s, y_s,
        train_sizes=train_sizes,
        cv=cv, scoring=scoring,
        n_jobs=-1, random_state=42
    )

    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(train_sizes_abs, train_mean, 'b-o', label='Training score')
    ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std,
                    alpha=0.15, color='blue')
    ax.plot(train_sizes_abs, val_mean, 'r-o', label='Validation score')
    ax.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std,
                    alpha=0.15, color='red')
    ax.set_title(f'Learning Curve ({scoring.upper()}) — {label}',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel(scoring.upper())
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/cv6_learning_curve.png', dpi=150)
    plt.close()
    print(f"  Final train {scoring}: {train_mean[-1]:.4f} ± {train_std[-1]:.4f}")
    print(f"  Final val   {scoring}: {val_mean[-1]:.4f}  ± {val_std[-1]:.4f}")
    gap = train_mean[-1] - val_mean[-1]
    if gap > 0.15:
        print(f"  HIGH VARIANCE (gap={gap:.3f}): model may be overfitting — try regularization")
    elif val_mean[-1] < 0.6:
        print(f"  HIGH BIAS (val={val_mean[-1]:.3f}): model may be underfitting — try more features")
    else:
        print(f"  GOOD FIT (gap={gap:.3f}): model generalises well")
    print("  Saved: cv6_learning_curve.png")


# ─────────────────────────────────────────────────────────────────────────────
# 8. PERMUTATION IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────
def plot_permutation_importance(model, X_test, y_test, feature_names,
                                task='regression', label='Model', n_repeats=10):
    """
    Model-agnostic permutation importance: measures how much performance drops
    when each feature is randomly shuffled.
    """
    print("\n--- [8] Permutation Importance ---")
    ensure_dir()

    scoring = 'r2' if task == 'regression' else 'f1_weighted'
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats, random_state=42, scoring=scoring, n_jobs=-1
    )

    perm_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean':    result.importances_mean,
        'Std':     result.importances_std
    }).sort_values('Mean', ascending=True)

    top_n = min(20, len(perm_df))
    perm_top = perm_df.tail(top_n)

    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.35)))
    ax.barh(perm_top['Feature'], perm_top['Mean'],
            xerr=perm_top['Std'], color='coral', edgecolor='gray', capsize=3)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title(f'Permutation Importance (top {top_n}) — {label}',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel(f'Decrease in {scoring.upper()} when feature is shuffled')
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/cv7_permutation_importance.png', dpi=150)
    plt.close()
    print(f"  Top 5 features by permutation importance:")
    for _, row in perm_top.tail(5).iloc[::-1].iterrows():
        print(f"    {row['Feature']}: {row['Mean']:+.4f} ± {row['Std']:.4f}")
    print("  Saved: cv7_permutation_importance.png")

    return perm_df


# ─────────────────────────────────────────────────────────────────────────────
# 9. REGIONAL MODEL STABILITY
# ─────────────────────────────────────────────────────────────────────────────
def regional_stability_report(regional_perf_df, task='forecast'):
    """
    Analyse consistency and stability of regional models.
    Flags regions with unusually high error (outliers).
    """
    print(f"\n--- [9] Regional Model Stability ({task}) ---")
    ensure_dir()

    if regional_perf_df is None or (isinstance(regional_perf_df, pd.DataFrame) and regional_perf_df.empty):
        print("  No regional performance data available.")
        return

    metric_col = 'MAE' if task == 'forecast' else 'F1'
    df = regional_perf_df.copy()

    q1   = df[metric_col].quantile(0.25)
    q3   = df[metric_col].quantile(0.75)
    iqr  = q3 - q1
    upper = q3 + 1.5 * iqr
    lower = q1 - 1.5 * iqr

    # High error (for MAE: high = bad; for F1: low = bad)
    if task == 'forecast':
        outliers = df[df[metric_col] > upper]
        direction = 'HIGH ERROR'
    else:
        outliers = df[df[metric_col] < lower]
        direction = 'LOW F1'

    print(f"  {metric_col} stats: mean={df[metric_col].mean():.4f}  "
          f"std={df[metric_col].std():.4f}  "
          f"min={df[metric_col].min():.4f}  max={df[metric_col].max():.4f}")

    if not outliers.empty:
        print(f"  OUTLIER regions ({direction}):")
        for _, row in outliers.iterrows():
            print(f"    - {row['Region']}: {metric_col}={row[metric_col]:.4f}")
    else:
        print("  No outlier regions detected — models are stable across regions")

    # Coefficient of Variation
    cv = df[metric_col].std() / df[metric_col].mean() if df[metric_col].mean() != 0 else 0
    print(f"  Coefficient of Variation (CV): {cv:.4f}  "
          f"({'INCONSISTENT' if cv > 0.4 else 'CONSISTENT'} across regions)")

    # Visual
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    df_sorted = df.sort_values(metric_col, ascending=(task != 'forecast'))
    color = ['salmon' if (task == 'forecast' and v > upper) or
                         (task != 'forecast' and v < lower)
             else 'steelblue'
             for v in df_sorted[metric_col]]
    axes[0].barh(df_sorted['Region'], df_sorted[metric_col],
                 color=color, edgecolor='gray')
    axes[0].axvline(df[metric_col].mean(), color='black', linestyle='--', label='Mean')
    axes[0].set_title(f'{metric_col} by Region (red = outlier)', fontweight='bold')
    axes[0].legend()

    axes[1].boxplot(df[metric_col], vert=True, patch_artist=True,
                    boxprops=dict(facecolor='lightblue'),
                    medianprops=dict(color='red', linewidth=2))
    axes[1].scatter([1] * len(df), df[metric_col], color='navy', s=25, zorder=3, alpha=0.6)
    axes[1].set_title(f'{metric_col} Distribution', fontweight='bold')
    axes[1].set_xticklabels([metric_col])

    fname = f'cv8_regional_stability_{task}.png'
    plt.suptitle(f'Regional Model Stability — {task.title()}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/{fname}', dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────
def run_verification(df):
    """
    Run the full model efficiency verification suite.
    Loads saved models and re-uses preprocessed data.
    """
    from sklearn.model_selection import train_test_split
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier

    print("\n" + "=" * 70)
    print("MODEL EFFICIENCY VERIFICATION SUITE")
    print("=" * 70)
    ensure_dir()

    # ── Load metadata ─────────────────────────────────────────────────────
    with open(f'{MODEL_DIR}/metadata.json', 'r') as f:
        meta = json.load(f)

    # ══════════════════════════════════════════════════════════════════════
    # SECTION A: DEMAND FORECASTING VERIFICATION
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SECTION A — DEMAND FORECASTING VERIFICATION")
    print("=" * 70)

    from demand_forecasting import build_regional_time_series, get_feature_columns

    df_ts = build_regional_time_series(df)
    if df_ts is not None:
        feature_cols = get_feature_columns(df_ts)
        target = 'Sales'
        df_clean = df_ts.dropna(subset=feature_cols + [target]).sort_values('Date')

        X_all = df_clean[feature_cols].values
        y_all = df_clean[target].values

        split_idx = int(len(df_clean) * 0.7)
        X_tr, X_te = X_all[:split_idx], X_all[split_idx:]
        y_tr, y_te = y_all[:split_idx], y_all[split_idx:]

        # Load saved global model
        global_forecast = joblib.load(f'{MODEL_DIR}/global_xgboost_forecast.pkl')

        # 1. Time-Series Cross-Validation
        cv_forecast(
            xgb.XGBRegressor,
            dict(n_estimators=200, max_depth=6, learning_rate=0.05,
                 subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0),
            df_ts, feature_cols, target, n_splits=5
        )

        # 3. Overfitting Check
        overfitting_check(global_forecast, X_tr, y_tr, X_te, y_te, task='regression')

        # 4. Residual Analysis
        y_pred_test = global_forecast.predict(X_te)
        residual_analysis(y_te, y_pred_test, label='Demand Forecasting (Global XGBoost)')

        # 7. Learning Curve
        plot_learning_curve(
            xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                              subsample=0.8, random_state=42, verbosity=0),
            X_all, y_all, task='regression', cv=3, label='Demand Forecasting'
        )

        # 8. Permutation Importance
        plot_permutation_importance(
            global_forecast, X_te, y_te, feature_cols,
            task='regression', label='Global Forecast'
        )

        # 9. Regional stability
        regional_perf_records = meta.get('regional_forecast_performance', [])
        regional_perf_df = pd.DataFrame(regional_perf_records) if regional_perf_records else pd.DataFrame()
        regional_stability_report(regional_perf_df, task='forecast')

    # ══════════════════════════════════════════════════════════════════════
    # SECTION B: LATE DELIVERY CLASSIFIER VERIFICATION
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SECTION B — LATE DELIVERY CLASSIFIER VERIFICATION")
    print("=" * 70)

    from late_delivery_classifier import prepare_classification_features

    if 'Late_delivery_risk' in df.columns:
        df_class, clf_feature_cols = prepare_classification_features(df)
        X_clf = df_class[clf_feature_cols].values
        y_clf = df_class['Late_delivery_risk'].astype(int).values

        X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(
            X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
        )

        global_clf = joblib.load(f'{MODEL_DIR}/global_rf_classifier.pkl')

        # 2. Stratified K-Fold CV
        cv_classifier(
            RandomForestClassifier,
            dict(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1),
            X_clf, y_clf, n_splits=5
        )

        # 3. Overfitting Check
        overfitting_check(global_clf, X_tr_c, y_tr_c, X_te_c, y_te_c, task='classification')

        # Predict probabilities for curves
        y_proba_test = global_clf.predict_proba(X_te_c)[:, 1]

        # 5. Precision-Recall Curve
        plot_precision_recall(y_te_c, y_proba_test)

        # 6. Calibration Curve
        plot_calibration(y_te_c, y_proba_test)

        # 7. Learning Curve
        plot_learning_curve(
            RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1),
            X_clf, y_clf, task='classification', cv=3, label='Late Delivery Classifier'
        )

        # 8. Permutation Importance (classifier)
        plot_permutation_importance(
            global_clf, X_te_c, y_te_c, clf_feature_cols,
            task='classification', label='Late Delivery Classifier'
        )

        # 9. Regional stability (classifier)
        regional_clf_records = meta.get('regional_classifier_performance', [])
        regional_clf_df = pd.DataFrame(regional_clf_records) if regional_clf_records else pd.DataFrame()
        regional_stability_report(regional_clf_df, task='classifier')

    # ── Final summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)
    generated = [f for f in os.listdir(PLOT_DIR) if f.endswith('.png')]
    print(f"  {len(generated)} verification plots saved to: {PLOT_DIR}/")
    for f in sorted(generated):
        print(f"    - {f}")
    print("=" * 70)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    from data_preprocessing import prepare_full_dataset
    from clustering import run_clustering

    df = prepare_full_dataset()
    df, _ = run_clustering(df)
    run_verification(df)
