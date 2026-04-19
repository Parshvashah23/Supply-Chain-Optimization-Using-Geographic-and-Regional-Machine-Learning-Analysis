"""
Regional Demand Forecasting
Global, Region-Specific, and Ensemble XGBoost Models
DataCo Supply Chain Dataset
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import os
import warnings
warnings.filterwarnings('ignore')

PLOT_DIR = 'plots'


def ensure_plot_dir():
    os.makedirs(PLOT_DIR, exist_ok=True)


def build_regional_time_series(df):
    """Aggregate daily demand by region and create lag/rolling features."""
    print("\n  Building regional time series features...")

    date_col = 'order date (DateOrders)'
    if date_col not in df.columns or not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        print("  ERROR: Date column not available or not parsed. Cannot build time series.")
        return None

    df_ts = df.copy()
    df_ts['Date'] = df_ts[date_col].dt.date

    # Aggregate daily by region
    agg = df_ts.groupby(['Macro_Region', 'Date']).agg({
        'Sales': 'sum',
        'Order Item Quantity': 'sum',
        'Late_delivery_risk': 'mean',
        'Customer Id': 'nunique'
    }).reset_index()
    agg.rename(columns={'Customer Id': 'Unique_Customers'}, inplace=True)
    agg['Date'] = pd.to_datetime(agg['Date'])
    agg = agg.sort_values(['Macro_Region', 'Date']).reset_index(drop=True)

    # Lag features
    for lag in [1, 7, 14, 30]:
        agg[f'Sales_lag_{lag}'] = agg.groupby('Macro_Region')['Sales'].shift(lag)
        agg[f'Qty_lag_{lag}'] = agg.groupby('Macro_Region')['Order Item Quantity'].shift(lag)

    # Rolling features
    for window in [7, 30]:
        agg[f'Sales_roll_{window}d'] = agg.groupby('Macro_Region')['Sales'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )

    # Temporal features
    agg['DayOfWeek'] = agg['Date'].dt.dayofweek
    agg['Month'] = agg['Date'].dt.month
    agg['Quarter'] = agg['Date'].dt.quarter
    agg['Is_Weekend'] = agg['DayOfWeek'].isin([5, 6]).astype(int)
    agg['Is_Month_End'] = agg['Date'].dt.is_month_end.astype(int)

    # Merge regional aggregate features if available
    regional_cols = ['Macro_Region', 'Region_Avg_Late_Delivery_Rate',
                     'Region_Avg_Order_Value', 'Region_Total_Orders']
    available_regional = [c for c in regional_cols if c in df.columns]
    if len(available_regional) > 1:
        regional_stats = df[available_regional].drop_duplicates()
        agg = agg.merge(regional_stats, on='Macro_Region', how='left')

    # Add cluster feature if available
    if 'KMeans_Cluster' in df.columns:
        cluster_map = df.groupby('Macro_Region')['KMeans_Cluster'].agg(lambda x: x.mode().iloc[0])
        agg['Geographic_Cluster'] = agg['Macro_Region'].map(cluster_map).fillna(0).astype(int)

    print(f"  Time series dataset: {agg.shape[0]:,} rows, {agg.shape[1]} columns")
    print(f"  Regions: {agg['Macro_Region'].nunique()}")
    return agg


def get_feature_columns(df_ts):
    """Return the feature columns available in the time series dataframe."""
    candidates = [
        'Sales_lag_1', 'Sales_lag_7', 'Sales_lag_14', 'Sales_lag_30',
        'Qty_lag_1', 'Qty_lag_7',
        'Sales_roll_7d', 'Sales_roll_30d',
        'DayOfWeek', 'Month', 'Quarter', 'Is_Weekend', 'Is_Month_End',
        'Late_delivery_risk',
        'Region_Avg_Late_Delivery_Rate', 'Region_Avg_Order_Value',
        'Geographic_Cluster'
    ]
    return [c for c in candidates if c in df_ts.columns]


def train_global_model(df_ts, feature_cols, target='Sales'):
    """Train a global XGBoost model across all regions."""
    print("\n--- Global XGBoost Demand Forecasting Model ---")

    df_model = df_ts.dropna(subset=feature_cols + [target]).copy()
    df_model = df_model.sort_values('Date').reset_index(drop=True)

    X = df_model[feature_cols]
    y = df_model[target]

    # 70/30 temporal split
    split_idx = int(len(df_model) * 0.7)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"  Train size: {len(X_train):,}, Test size: {len(X_test):,}")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R2:   {r2:.4f}")

    global_metrics = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
    return model, global_metrics, X_train.columns.tolist()


def train_regional_models(df_ts, feature_cols, target='Sales'):
    """Train separate XGBoost models per region."""
    print("\n--- Region-Specific Demand Forecasting Models ---")

    regions = df_ts['Macro_Region'].unique()
    regional_models = {}
    regional_performance = []

    for region in sorted(regions):
        df_region = df_ts[df_ts['Macro_Region'] == region].copy()
        df_region = df_region.dropna(subset=feature_cols + [target])
        df_region = df_region.sort_values('Date').reset_index(drop=True)

        if len(df_region) < 60:
            print(f"  [{region}] Skipped - only {len(df_region)} samples")
            continue

        X = df_region[feature_cols]
        y = df_region[target]

        split_idx = int(len(df_region) * 0.7)
        if split_idx < 20 or (len(df_region) - split_idx) < 10:
            print(f"  [{region}] Skipped - insufficient train/test split")
            continue

        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        model = xgb.XGBRegressor(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        regional_models[region] = model
        regional_performance.append({
            'Region': region, 'MAE': mae, 'RMSE': rmse, 'R2': r2,
            'Samples': len(df_region), 'Train': len(X_train), 'Test': len(X_test)
        })
        print(f"  [{region}] MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.4f}  (n={len(df_region)})")

    perf_df = pd.DataFrame(regional_performance)
    if not perf_df.empty:
        print(f"\n  Trained {len(regional_models)} regional models")
        print(f"  Average Regional MAE: {perf_df['MAE'].mean():.2f}")
        print(f"  Average Regional R2:  {perf_df['R2'].mean():.4f}")

    return regional_models, perf_df


def ensemble_predict(X, region, global_model, regional_models, weight_regional=0.7):
    """Weighted ensemble: regional + global model predictions."""
    global_pred = global_model.predict(X)
    if region in regional_models:
        regional_pred = regional_models[region].predict(X)
        return weight_regional * regional_pred + (1 - weight_regional) * global_pred
    return global_pred


def evaluate_ensemble(df_ts, feature_cols, global_model, regional_models, target='Sales'):
    """Evaluate ensemble model across all regions on the test set."""
    print("\n--- Ensemble Model Evaluation ---")

    df_model = df_ts.dropna(subset=feature_cols + [target]).copy()
    df_model = df_model.sort_values('Date').reset_index(drop=True)
    split_idx = int(len(df_model) * 0.7)
    df_test = df_model.iloc[split_idx:]

    all_preds = []
    all_actuals = []

    for region in df_test['Macro_Region'].unique():
        df_r = df_test[df_test['Macro_Region'] == region]
        if len(df_r) == 0:
            continue
        X_r = df_r[feature_cols]
        y_r = df_r[target]
        pred = ensemble_predict(X_r, region, global_model, regional_models)
        all_preds.extend(pred)
        all_actuals.extend(y_r.values)

    if len(all_preds) == 0:
        print("  No predictions generated.")
        return {}

    mae = mean_absolute_error(all_actuals, all_preds)
    rmse = np.sqrt(mean_squared_error(all_actuals, all_preds))
    r2 = r2_score(all_actuals, all_preds)

    print(f"  Ensemble MAE:  {mae:.2f}")
    print(f"  Ensemble RMSE: {rmse:.2f}")
    print(f"  Ensemble R2:   {r2:.4f}")

    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}


def plot_feature_importance(model, feature_cols, title='Global Model'):
    """Plot XGBoost feature importance."""
    ensure_plot_dir()
    importance = model.feature_importances_
    feat_imp = pd.Series(importance, index=feature_cols).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(4, len(feature_cols) * 0.35)))
    feat_imp.plot(kind='barh', color='steelblue', ax=ax)
    ax.set_title(f'Feature Importance - {title}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance')
    plt.tight_layout()
    fname = f"11_feature_importance_{title.lower().replace(' ', '_')}.png"
    plt.savefig(f'{PLOT_DIR}/{fname}', dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


def plot_model_comparison(global_metrics, regional_perf_df, ensemble_metrics):
    """Bar chart: comparing Global vs Regional vs Ensemble."""
    ensure_plot_dir()
    data = {
        'Model': ['Global', 'Regional (Avg)', 'Ensemble'],
        'MAE': [
            global_metrics['MAE'],
            regional_perf_df['MAE'].mean() if not regional_perf_df.empty else 0,
            ensemble_metrics.get('MAE', 0)
        ],
        'RMSE': [
            global_metrics['RMSE'],
            regional_perf_df['RMSE'].mean() if not regional_perf_df.empty else 0,
            ensemble_metrics.get('RMSE', 0)
        ],
        'R2': [
            global_metrics['R2'],
            regional_perf_df['R2'].mean() if not regional_perf_df.empty else 0,
            ensemble_metrics.get('R2', 0)
        ]
    }
    comp_df = pd.DataFrame(data)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, metric in enumerate(['MAE', 'RMSE', 'R2']):
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        axes[i].bar(comp_df['Model'], comp_df[metric], color=colors, edgecolor='gray')
        axes[i].set_title(metric, fontsize=14, fontweight='bold')
        axes[i].set_ylabel(metric)
        for j, v in enumerate(comp_df[metric]):
            axes[i].text(j, v + v * 0.01, f'{v:.2f}', ha='center', fontsize=9)

    plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/12_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 12_model_comparison.png")

    # Print comparison table
    print("\n" + "=" * 65)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 65)
    print(comp_df.to_string(index=False))
    if global_metrics['MAE'] > 0 and not regional_perf_df.empty:
        improvement = ((global_metrics['MAE'] - regional_perf_df['MAE'].mean()) /
                       global_metrics['MAE']) * 100
        print(f"\nRegional vs Global MAE Improvement: {improvement:+.2f}%")
    if global_metrics['MAE'] > 0 and ensemble_metrics.get('MAE', 0) > 0:
        improvement_ens = ((global_metrics['MAE'] - ensemble_metrics['MAE']) /
                           global_metrics['MAE']) * 100
        print(f"Ensemble vs Global MAE Improvement: {improvement_ens:+.2f}%")
    print("=" * 65)


def run_demand_forecasting(df):
    """Full demand forecasting pipeline."""
    print("\n" + "=" * 70)
    print("REGIONAL DEMAND FORECASTING")
    print("=" * 70)

    # Build time series
    df_ts = build_regional_time_series(df)
    if df_ts is None:
        print("Cannot proceed without time series data.")
        return df, {}, {}, {}, pd.DataFrame()

    feature_cols = get_feature_columns(df_ts)
    print(f"  Feature columns ({len(feature_cols)}): {feature_cols}")

    if len(feature_cols) < 3:
        print("  Insufficient features for forecasting. Skipping.")
        return df, {}, {}, {}, pd.DataFrame()

    # Global model
    global_model, global_metrics, trained_features = train_global_model(df_ts, feature_cols)
    plot_feature_importance(global_model, feature_cols, 'Global Model')

    # Regional models
    regional_models, regional_perf_df = train_regional_models(df_ts, feature_cols)

    # Ensemble evaluation
    ensemble_metrics = evaluate_ensemble(df_ts, feature_cols, global_model, regional_models)

    # Comparison plot
    plot_model_comparison(global_metrics, regional_perf_df, ensemble_metrics)

    print("\n" + "=" * 70)
    return df, global_model, regional_models, global_metrics, regional_perf_df


if __name__ == '__main__':
    from data_preprocessing import prepare_full_dataset
    from clustering import run_clustering
    df = prepare_full_dataset()
    df, _ = run_clustering(df)
    run_demand_forecasting(df)
