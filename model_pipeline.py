"""
Model Saving, Loading, and Prediction Pipeline
DataCo Supply Chain Dataset
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

MODEL_DIR = 'models/regional_forecast'


def save_models(global_forecast_model, regional_forecast_models,
                global_classifier, regional_classifiers,
                feature_cols_forecast, feature_cols_classify,
                global_forecast_metrics, regional_forecast_perf,
                global_classify_metrics, regional_classify_perf,
                clustering_metrics):
    """Save all trained models and metadata to disk."""
    print("\n" + "=" * 70)
    print("SAVING MODELS")
    print("=" * 70)

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save forecasting models
    if global_forecast_model is not None:
        joblib.dump(global_forecast_model, f'{MODEL_DIR}/global_xgboost_forecast.pkl')
        print(f"  Saved: global_xgboost_forecast.pkl")

    if regional_forecast_models:
        for region, model in regional_forecast_models.items():
            safe_name = region.replace(' ', '_').replace('/', '_')
            joblib.dump(model, f'{MODEL_DIR}/{safe_name}_xgboost_forecast.pkl')
        print(f"  Saved: {len(regional_forecast_models)} regional forecast models")

    # Save classifiers
    if global_classifier is not None:
        joblib.dump(global_classifier, f'{MODEL_DIR}/global_rf_classifier.pkl')
        print(f"  Saved: global_rf_classifier.pkl")

    if regional_classifiers:
        for region, model in regional_classifiers.items():
            safe_name = region.replace(' ', '_').replace('/', '_')
            joblib.dump(model, f'{MODEL_DIR}/{safe_name}_rf_classifier.pkl')
        print(f"  Saved: {len(regional_classifiers)} regional classifiers")

    # Save metadata
    metadata = {
        'training_date': datetime.now().isoformat(),
        'forecast_regions': list(regional_forecast_models.keys()) if regional_forecast_models else [],
        'classifier_regions': list(regional_classifiers.keys()) if regional_classifiers else [],
        'forecast_features': feature_cols_forecast if feature_cols_forecast else [],
        'classifier_features': feature_cols_classify if feature_cols_classify else [],
        'global_forecast_metrics': global_forecast_metrics if global_forecast_metrics else {},
        'global_classifier_metrics': global_classify_metrics if global_classify_metrics else {},
        'clustering_metrics': clustering_metrics if clustering_metrics else {},
    }

    if isinstance(regional_forecast_perf, pd.DataFrame) and not regional_forecast_perf.empty:
        metadata['regional_forecast_performance'] = regional_forecast_perf.to_dict('records')
    if isinstance(regional_classify_perf, pd.DataFrame) and not regional_classify_perf.empty:
        metadata['regional_classifier_performance'] = regional_classify_perf.to_dict('records')

    with open(f'{MODEL_DIR}/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  Saved: metadata.json")

    print("=" * 70)
    return metadata


class RegionalDemandPredictor:
    """Production-ready regional demand forecasting + late delivery prediction pipeline."""

    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir = model_dir
        self.global_forecast = None
        self.regional_forecasts = {}
        self.global_classifier = None
        self.regional_classifiers = {}
        self.metadata = {}

        self._load_models()

    def _load_models(self):
        """Load all models from disk."""
        meta_path = f'{self.model_dir}/metadata.json'
        if not os.path.exists(meta_path):
            print(f"No metadata found at {meta_path}. Pipeline not initialized.")
            return

        with open(meta_path, 'r') as f:
            self.metadata = json.load(f)

        # Load global forecast
        gf_path = f'{self.model_dir}/global_xgboost_forecast.pkl'
        if os.path.exists(gf_path):
            self.global_forecast = joblib.load(gf_path)

        # Load regional forecasts
        for region in self.metadata.get('forecast_regions', []):
            safe_name = region.replace(' ', '_').replace('/', '_')
            path = f'{self.model_dir}/{safe_name}_xgboost_forecast.pkl'
            if os.path.exists(path):
                self.regional_forecasts[region] = joblib.load(path)

        # Load global classifier
        gc_path = f'{self.model_dir}/global_rf_classifier.pkl'
        if os.path.exists(gc_path):
            self.global_classifier = joblib.load(gc_path)

        # Load regional classifiers
        for region in self.metadata.get('classifier_regions', []):
            safe_name = region.replace(' ', '_').replace('/', '_')
            path = f'{self.model_dir}/{safe_name}_rf_classifier.pkl'
            if os.path.exists(path):
                self.regional_classifiers[region] = joblib.load(path)

        print(f"Loaded: {1 if self.global_forecast else 0} global forecast, "
              f"{len(self.regional_forecasts)} regional forecasts, "
              f"{1 if self.global_classifier else 0} global classifier, "
              f"{len(self.regional_classifiers)} regional classifiers")

    def predict_demand(self, region, features_dict):
        """Predict demand for a specific region (ensemble: 70% regional, 30% global)."""
        X = pd.DataFrame([features_dict])

        # Align columns
        expected = self.metadata.get('forecast_features', [])
        for col in expected:
            if col not in X.columns:
                X[col] = 0
        X = X.reindex(columns=expected, fill_value=0)

        global_pred = self.global_forecast.predict(X)[0] if self.global_forecast else 0

        if region in self.regional_forecasts:
            regional_pred = self.regional_forecasts[region].predict(X)[0]
            return 0.7 * regional_pred + 0.3 * global_pred
        return global_pred

    def predict_late_delivery(self, features_dict, region=None):
        """Predict late delivery probability."""
        X = pd.DataFrame([features_dict])

        expected = self.metadata.get('classifier_features', [])
        for col in expected:
            if col not in X.columns:
                X[col] = 0
        X = X.reindex(columns=expected, fill_value=0)

        if region and region in self.regional_classifiers:
            model = self.regional_classifiers[region]
        elif self.global_classifier:
            model = self.global_classifier
        else:
            return {'prediction': None, 'probability': None}

        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0][1]

        return {'prediction': int(pred), 'probability': float(proba)}

    def predict_all_regions(self, features_dict):
        """Predict demand for all available regions."""
        results = {}
        for region in self.metadata.get('forecast_regions', []):
            results[region] = self.predict_demand(region, features_dict)
        return results

    def summary(self):
        """Print model summary."""
        print("\n" + "=" * 60)
        print("MODEL PIPELINE SUMMARY")
        print("=" * 60)
        print(f"Training Date: {self.metadata.get('training_date', 'N/A')}")
        print(f"Forecast Regions: {len(self.metadata.get('forecast_regions', []))}")
        print(f"Classifier Regions: {len(self.metadata.get('classifier_regions', []))}")

        gfm = self.metadata.get('global_forecast_metrics', {})
        if gfm:
            print(f"\nGlobal Forecast -> MAE: {gfm.get('MAE', 'N/A'):.2f}, "
                  f"R2: {gfm.get('R2', 'N/A'):.4f}")

        gcm = self.metadata.get('global_classifier_metrics', {})
        if gcm:
            print(f"Global Classifier -> F1: {gcm.get('F1', 'N/A'):.4f}, "
                  f"AUC: {gcm.get('ROC_AUC', 'N/A'):.4f}")
        print("=" * 60)


def generate_insights(df, global_forecast_metrics, regional_forecast_perf,
                      global_classify_metrics, regional_classify_perf, clustering_metrics):
    """Generate automated business insights."""
    print("\n" + "=" * 70)
    print("KEY BUSINESS INSIGHTS")
    print("=" * 70)

    insights = []

    # 1. Late delivery hotspots
    if 'Macro_Region' in df.columns and 'Late_delivery_risk' in df.columns:
        region_late = df.groupby('Macro_Region')['Late_delivery_risk'].mean().sort_values(ascending=False)
        high_risk = region_late[region_late > 0.50]
        if not high_risk.empty:
            regions_str = ', '.join([f"{r} ({v:.0%})" for r, v in high_risk.items()])
            insights.append(f"HIGH-RISK REGIONS (>50% late delivery): {regions_str}")

    # 2. Revenue concentration
    if 'Macro_Region' in df.columns and 'Sales' in df.columns:
        region_sales = df.groupby('Macro_Region')['Sales'].sum().sort_values(ascending=False)
        total = region_sales.sum()
        top3 = region_sales.head(3)
        top3_pct = (top3.sum() / total) * 100
        regions_str = ', '.join(top3.index.tolist())
        insights.append(f"TOP 3 REGIONS by revenue ({regions_str}): {top3_pct:.1f}% of total sales")

    # 3. Model performance
    if global_forecast_metrics:
        mae = global_forecast_metrics.get('MAE', 0)
        r2 = global_forecast_metrics.get('R2', 0)
        insights.append(f"DEMAND FORECASTING: Global MAE={mae:.2f}, R2={r2:.4f}")

    if isinstance(regional_forecast_perf, pd.DataFrame) and not regional_forecast_perf.empty:
        avg_mae = regional_forecast_perf['MAE'].mean()
        if global_forecast_metrics and global_forecast_metrics.get('MAE', 0) > 0:
            imp = ((global_forecast_metrics['MAE'] - avg_mae) / global_forecast_metrics['MAE']) * 100
            insights.append(f"REGIONAL MODELS improve forecast MAE by {imp:+.1f}% vs global")

    if global_classify_metrics:
        f1 = global_classify_metrics.get('F1', 0)
        auc = global_classify_metrics.get('ROC_AUC', 0)
        insights.append(f"LATE DELIVERY CLASSIFIER: F1={f1:.4f}, ROC-AUC={auc:.4f}")

    # 4. Clustering
    if clustering_metrics and 'kmeans' in clustering_metrics:
        sil = clustering_metrics['kmeans'].get('silhouette', 0)
        insights.append(f"GEOGRAPHIC CLUSTERING: K-Means Silhouette Score = {sil:.4f}")

    # 5. Shipping patterns
    if 'Days for shipping (real)' in df.columns and 'Macro_Region' in df.columns:
        region_ship = df.groupby('Macro_Region')['Days for shipping (real)'].mean()
        fastest = region_ship.idxmin()
        slowest = region_ship.idxmax()
        insights.append(f"FASTEST SHIPPING: {fastest} ({region_ship[fastest]:.1f} days avg)")
        insights.append(f"SLOWEST SHIPPING: {slowest} ({region_ship[slowest]:.1f} days avg)")

    for i, insight in enumerate(insights, 1):
        print(f"  {i}. {insight}")
    print("=" * 70)

    return insights


if __name__ == '__main__':
    predictor = RegionalDemandPredictor()
    predictor.summary()
