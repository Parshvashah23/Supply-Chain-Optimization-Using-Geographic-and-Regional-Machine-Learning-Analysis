"""
Late Delivery Prediction - Regional Classifiers
RandomForest & XGBoost classifiers (Global + Per-Region)
DataCo Supply Chain Dataset
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_score, recall_score, f1_score, roc_curve
)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import os
import warnings
warnings.filterwarnings('ignore')

PLOT_DIR = 'plots'


def ensure_plot_dir():
    os.makedirs(PLOT_DIR, exist_ok=True)


def prepare_classification_features(df):
    """Prepare features for late delivery classification."""
    print("\n  Preparing classification features...")

    feature_candidates = [
        'Days for shipment (scheduled)',
        'Order Item Total', 'Sales', 'Order Item Quantity',
        'Order Item Discount Rate', 'Benefit per order',
        'Shipping_Distance_KM', 'Weather_Risk_Score',
        'Region_Avg_Late_Delivery_Rate', 'Region_Avg_Order_Value',
        'Region_Avg_Shipping_Days',
        'Order_Month', 'Order_DayOfWeek', 'Is_Weekend', 'Holiday_Flag',
        'Is_Cross_Border', 'KMeans_Cluster'
    ]

    # Categorical features to encode
    cat_features = ['Shipping Mode', 'Season', 'Customer Segment', 'Macro_Region']

    available_num = [c for c in feature_candidates if c in df.columns]
    available_cat = [c for c in cat_features if c in df.columns]

    df_class = df[available_num + available_cat + ['Late_delivery_risk']].copy()
    df_class = df_class.dropna(subset=['Late_delivery_risk'])

    # Encode categoricals
    for col in available_cat:
        df_class = pd.get_dummies(df_class, columns=[col], drop_first=True, prefix=col)

    feature_cols = [c for c in df_class.columns if c != 'Late_delivery_risk']
    print(f"  Classification features: {len(feature_cols)}")

    return df_class, feature_cols


def train_global_classifier(df_class, feature_cols):
    """Train a global RandomForest classifier for late delivery prediction."""
    print("\n--- Global Late Delivery Classifier (RandomForest) ---")

    X = df_class[feature_cols]
    y = df_class['Late_delivery_risk'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"  Train: {len(X_train):,}, Test: {len(X_test):,}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['On-Time', 'Late']))

    global_metrics = {
        'Precision': precision, 'Recall': recall,
        'F1': f1, 'ROC_AUC': roc_auc
    }

    return model, global_metrics, X_test, y_test, y_pred, y_proba


def train_regional_classifiers(df, df_class, feature_cols):
    """Train separate classifiers per region."""
    print("\n--- Region-Specific Late Delivery Classifiers ---")

    if 'Macro_Region' not in df.columns:
        print("  Macro_Region not available. Skipping regional classifiers.")
        return {}, pd.DataFrame()

    # We need the Macro_Region aligned with df_class indices
    df_class_with_region = df_class.copy()
    df_class_with_region['_Macro_Region'] = df.loc[df_class.index, 'Macro_Region'].values

    regions = df_class_with_region['_Macro_Region'].unique()
    regional_classifiers = {}
    regional_performance = []

    for region in sorted(regions):
        df_r = df_class_with_region[df_class_with_region['_Macro_Region'] == region]
        y_r = df_r['Late_delivery_risk'].astype(int)

        if len(df_r) < 100 or y_r.sum() < 20 or (y_r == 0).sum() < 20:
            continue

        X_r = df_r[feature_cols]

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_r, y_r, test_size=0.2, random_state=42, stratify=y_r
        )

        rf = RandomForestClassifier(
            n_estimators=150, max_depth=8, random_state=42, n_jobs=-1
        )
        rf.fit(X_tr, y_tr)

        y_pred_r = rf.predict(X_te)
        y_proba_r = rf.predict_proba(X_te)[:, 1]

        p = precision_score(y_te, y_pred_r)
        r = recall_score(y_te, y_pred_r)
        f1_val = f1_score(y_te, y_pred_r)
        auc = roc_auc_score(y_te, y_proba_r)

        regional_classifiers[region] = rf
        regional_performance.append({
            'Region': region, 'Precision': p, 'Recall': r,
            'F1': f1_val, 'ROC_AUC': auc,
            'Samples': len(df_r), 'Late_Count': int(y_r.sum())
        })
        print(f"  [{region}] P={p:.4f} R={r:.4f} F1={f1_val:.4f} AUC={auc:.4f} (n={len(df_r)})")

    perf_df = pd.DataFrame(regional_performance)
    if not perf_df.empty:
        print(f"\n  Trained {len(regional_classifiers)} regional classifiers")
        print(f"  Average F1: {perf_df['F1'].mean():.4f}")
        print(f"  Average AUC: {perf_df['ROC_AUC'].mean():.4f}")

    return regional_classifiers, perf_df


def plot_confusion_matrix(y_test, y_pred, title='Global Classifier'):
    """Plot confusion matrix."""
    ensure_plot_dir()
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['On-Time', 'Late'],
                yticklabels=['On-Time', 'Late'])
    ax.set_title(f'Confusion Matrix - {title}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    fname = f"13_confusion_matrix_{title.lower().replace(' ', '_')}.png"
    plt.savefig(f'{PLOT_DIR}/{fname}', dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


def plot_roc_curve(y_test, y_proba, title='Global Classifier'):
    """Plot ROC curve."""
    ensure_plot_dir()
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve - {title}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    plt.tight_layout()
    fname = f"14_roc_curve_{title.lower().replace(' ', '_')}.png"
    plt.savefig(f'{PLOT_DIR}/{fname}', dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


def plot_feature_importance_classifier(model, feature_cols, title='Global Classifier'):
    """Plot feature importance for the classifier."""
    ensure_plot_dir()
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    feat_imp.plot(kind='barh', color='coral', ax=ax)
    ax.set_title(f'Top 15 Feature Importance - {title}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance')
    plt.tight_layout()
    fname = f"15_classifier_feature_importance.png"
    plt.savefig(f'{PLOT_DIR}/{fname}', dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


def plot_regional_classifier_comparison(regional_perf_df):
    """Bar chart comparing regional classifier performance."""
    ensure_plot_dir()
    if regional_perf_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # F1 Score by Region
    regional_perf_df.sort_values('F1', ascending=True).plot(
        x='Region', y='F1', kind='barh', ax=axes[0], color='steelblue', legend=False
    )
    axes[0].set_title('F1 Score by Region', fontweight='bold')
    axes[0].set_xlabel('F1 Score')

    # ROC-AUC by Region
    regional_perf_df.sort_values('ROC_AUC', ascending=True).plot(
        x='Region', y='ROC_AUC', kind='barh', ax=axes[1], color='darkorange', legend=False
    )
    axes[1].set_title('ROC-AUC by Region', fontweight='bold')
    axes[1].set_xlabel('ROC-AUC')

    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/16_regional_classifier_comparison.png', dpi=150)
    plt.close()
    print("  Saved: 16_regional_classifier_comparison.png")


def plot_combined_roc_curves(global_clf, regional_clfs, regional_perf_df, df, df_class, feature_cols):
    """Plot global and top 6 regional ROC curves on a single figure."""
    if not regional_clfs or regional_perf_df.empty:
        return
        
    fig, ax = plt.subplots(figsize=(9, 8))
    
    # 1. Global ROC
    X = df_class[feature_cols]
    y = df_class['Late_delivery_risk'].astype(int)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    y_proba = global_clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    global_auc = roc_auc_score(y_test, y_proba)
    
    ax.plot(fpr, tpr, 'k--', linewidth=2.5, label=f'GLOBAL Model (AUC = {global_auc:.4f})', zorder=10)
    
    # 2. Regional ROCs (Top 6 by AUC)
    top_regions = regional_perf_df.sort_values('ROC_AUC', ascending=False).head(6)['Region'].tolist()
    
    df_class_with_region = df_class.copy()
    df_class_with_region['_Macro_Region'] = df.loc[df_class.index, 'Macro_Region'].values
    
    cmap = plt.get_cmap('tab10')
    
    for i, region in enumerate(top_regions):
        if region not in regional_clfs:
            continue
            
        df_r = df_class_with_region[df_class_with_region['_Macro_Region'] == region]
        y_r = df_r['Late_delivery_risk'].astype(int)
        X_r = df_r[feature_cols]
        
        _, X_te, _, y_te = train_test_split(X_r, y_r, test_size=0.2, random_state=42, stratify=y_r)
        
        y_proba_r = regional_clfs[region].predict_proba(X_te)[:, 1]
        fpr_r, tpr_r, _ = roc_curve(y_te, y_proba_r)
        auc_r = roc_auc_score(y_te, y_proba_r)
        
        ax.plot(fpr_r, tpr_r, linewidth=2, color=cmap(i), label=f'Region: {region} (AUC = {auc_r:.4f})')
        
    # Baseline
    ax.plot([0, 1], [0, 1], 'gray', linestyle=':', label='Baseline')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Late Delivery Classifier: Global vs Regional ROC/AUC', fontsize=15, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    # Save to verification directory
    out_dir = os.path.join(PLOT_DIR, 'verification')
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{out_dir}/cv9_combined_roc_curves.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved: verification/cv9_combined_roc_curves.png")


def run_late_delivery_classification(df):
    """Full late delivery classification pipeline."""
    print("\n" + "=" * 70)
    print("LATE DELIVERY PREDICTION (CLASSIFICATION)")
    print("=" * 70)

    if 'Late_delivery_risk' not in df.columns:
        print("Late_delivery_risk column not found. Skipping classification.")
        return df, None, {}, pd.DataFrame()

    df_class, feature_cols = prepare_classification_features(df)

    # Global classifier
    global_clf, global_metrics, X_test, y_test, y_pred, y_proba = \
        train_global_classifier(df_class, feature_cols)

    # Plots
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_proba)
    plot_feature_importance_classifier(global_clf, feature_cols)

    # Regional classifiers
    regional_clfs, regional_perf_df = train_regional_classifiers(df, df_class, feature_cols)
    plot_regional_classifier_comparison(regional_perf_df)

    # Summary
    print("\n" + "=" * 65)
    print("CLASSIFICATION SUMMARY")
    print("=" * 65)
    print(f"Global Classifier:  F1={global_metrics['F1']:.4f}  AUC={global_metrics['ROC_AUC']:.4f}")
    if not regional_perf_df.empty:
        print(f"Regional Avg F1:    {regional_perf_df['F1'].mean():.4f}")
        print(f"Regional Avg AUC:   {regional_perf_df['ROC_AUC'].mean():.4f}")
        best = regional_perf_df.loc[regional_perf_df['F1'].idxmax()]
        worst = regional_perf_df.loc[regional_perf_df['F1'].idxmin()]
        print(f"Best Region:  {best['Region']} (F1={best['F1']:.4f})")
        print(f"Worst Region: {worst['Region']} (F1={worst['F1']:.4f})")
    print("=" * 65)

    # Add probabilities to the main dataframe
    df['late_delivery_probability'] = global_clf.predict_proba(df_class[feature_cols])[:, 1]
    
    # Overlay plots
    plot_combined_roc_curves(global_clf, regional_clfs, regional_perf_df, df, df_class, feature_cols)

    return df, global_clf, regional_clfs, regional_perf_df


if __name__ == '__main__':
    from data_preprocessing import prepare_full_dataset
    from clustering import run_clustering
    df = prepare_full_dataset()
    df, _ = run_clustering(df)
    run_late_delivery_classification(df)
