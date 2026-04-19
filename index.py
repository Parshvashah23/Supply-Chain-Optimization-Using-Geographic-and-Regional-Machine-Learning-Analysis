"""
==========================================================================
Geographic Analysis & Regional Demand Patterns ML Model
DataCo Supply Chain Dataset
==========================================================================

Main orchestrator: runs the full ML pipeline end-to-end,
including all 6 Advanced Analytics Phases.

Usage:
    python index.py [--full]
    (Default runs on a 15,000 row sample for speed)
==========================================================================
"""

import os
import sys
import time
import warnings
import argparse
import pandas as pd
warnings.filterwarnings('ignore')

# Ensure the working directory is the project root
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)

# Existing modules
from data_preprocessing import prepare_full_dataset
from geographic_analysis import run_eda
from clustering import run_clustering
from demand_forecasting import run_demand_forecasting
from late_delivery_classifier import run_late_delivery_classification
from model_pipeline import save_models, generate_insights, RegionalDemandPredictor

# Phase 1: Explainability
from explainability.shap_explainer import SHAPExplainer
from explainability.lime_explainer import LIMEExplainer
from explainability.generate_xai_report import generate_xai_html_section

# Phase 2: Risk
from risk.fraud_detector import FraudDetector
from risk.cancellation_predictor import CancellationPredictor
from risk.composite_risk_scorer import CompositeRiskScorer

# Phase 3: Causal (Disabled due to dowhy import deadlock on Windows)
# from causal.causal_inference import SupplyChainCausalAnalyzer
# from causal.scenario_engine import ScenarioEngine

# Phase 4: Deep Learning / Hybrid (Disabled due to tensorflow import deadlock)
# from demand_forecasting_lstm import LSTMDemandForecaster
# from demand_forecasting_hybrid import HybridDemandForecaster

# Phase 5: Financial
from financial.cost_sensitive_classifier import CostSensitiveOptimizer
from financial.pareto_optimizer import ParetoOptimizer
from financial.esg_scorer import ESGScorer

# Phase 6: Resilience & Drift
from clustering_resilience import ResilienceIndexCalculator
from clustering_drift import ClusterDriftDetector


DATA_FILE = 'DataCoSupplyChainDataset.csv'


def main():
    parser = argparse.ArgumentParser(description="Run Supply Chain ML Pipeline")
    parser.add_argument("--full", action="store_true", help="Run on the full dataset instead of a sample")
    args = parser.parse_args()

    sample_size = None if args.full else 5000

    start = time.time()

    print("=" * 80)
    print("  GEOGRAPHIC ANALYSIS & ADVANCED SUPPLY CHAIN ML PIPELINE")
    print("  DataCo Supply Chain Dataset")
    if sample_size:
        print(f"  [RUNNING IN SAMPLE MODE: {sample_size} ROWS]")
    else:
        print("  [RUNNING IN FULL DATASET MODE]")
    print("=" * 80)
    print()

    # ── PHASE 0: Data Preprocessing ──────────────────────────────────────
    print("PHASE 0: DATA PREPROCESSING & FEATURE ENGINEERING")
    df = prepare_full_dataset(DATA_FILE)
    if sample_size and len(df) > sample_size:
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)
    print(f"\nDataset shape for pipeline: {df.shape}")
    print()

    # ── PHASE 0: Exploratory Data Analysis & Clustering ──────────────────
    print("PHASE 0.1: EXPLORATORY DATA ANALYSIS & CLUSTERING")
    stat_results = run_eda(df)
    df, clustering_metrics = run_clustering(df)
    print()

    # ── PHASE 0: Base Models ─────────────────────────────────────────────
    print("PHASE 0.2: BASE MODELS (XGBoost & RandomForest)")
    df, global_forecast_model, regional_forecast_models, \
        global_forecast_metrics, regional_forecast_perf = run_demand_forecasting(df)
    
    df, global_classifier, regional_classifiers, regional_classify_perf = \
        run_late_delivery_classification(df)
    
    # Extract feature columns
    classify_feature_cols = []
    if global_classifier is not None:
        classify_feature_cols = list(getattr(global_classifier, 'feature_names_in_', []))
    
    forecast_feature_cols = []
    if global_forecast_model is not None and hasattr(global_forecast_model, 'feature_names_in_'):
        forecast_feature_cols = list(global_forecast_model.feature_names_in_)

    print("\n" + "=" * 80)
    print("  ADVANCED PHASES (1-6) BEGINNING")
    print("=" * 80 + "\n")

    # ── PHASE 1: Explainability & Trust Layer ────────────────────────────
    print("PHASE 1: EXPLAINABILITY (SHAP & LIME)")
    if classify_feature_cols and forecast_feature_cols:
        shap_exp = SHAPExplainer()
        # Pass live models directly to avoid disk loading and feature mismatches
        shap_exp.load_models(xgb_demand=global_forecast_model, rf_classifier=global_classifier)
        
        # We need the one-hot encoded features and time series lags for the exact inputs
        from late_delivery_classifier import prepare_classification_features
        from demand_forecasting import build_regional_time_series
        
        df_class, _ = prepare_classification_features(df)
        df_ts = build_regional_time_series(df)
        
        # Capture regions before dummies consume them for the heatmap
        regions_for_heatmap = df_class.get('Macro_Region', df_class.get('Order Region', pd.Series(['Unknown']*len(df_class))))
        
        # Select features for SHAP
        X_classify = df_class[classify_feature_cols].select_dtypes(include=['number']).fillna(0)
        X_forecast = df_ts[forecast_feature_cols].select_dtypes(include=['number']).fillna(0) if df_ts is not None else pd.DataFrame()
        
        print("  -> Generating SHAP Global and Local Explanations...")
        try:
            # SHAP is slow; limit sample to 50 for integration test
            sample_idx = X_classify.sample(min(50, len(X_classify)), random_state=42).index
            X_sc = X_classify.loc[sample_idx]
            reg_labels_sc = regions_for_heatmap.loc[sample_idx]
            
            exp_c, shap_vals_c, X_sc_aligned = shap_exp.compute_shap_classifier(X_classify, X_sample=X_sc)
            if shap_vals_c is not None:
                shap_exp.regional_shap_heatmap(X_sc_aligned, shap_vals_c, region_col="Macro_Region", region_labels=reg_labels_sc)
        except Exception as e:
            print(f"  [SHAP Classifier Error] {e}")

        try:
            exp_d, shap_vals_d, X_sd = shap_exp.compute_shap_demand(X_forecast, X_sample=X_forecast.sample(min(50, len(X_forecast)), random_state=42))
        except Exception as e:
            print(f"  [SHAP Demand Error] {e}")

        # XAI Report embedding
        try:
            generate_xai_html_section()
        except Exception as e:
            print(f"  [XAI Report Error] {e}")
    print()

    # ── PHASE 2: Multi-Risk Prediction Engine ─────────────────────────────
    print("PHASE 2: MULTI-RISK PREDICTION ENGINE")
    fraud_clf = FraudDetector()
    cancel_clf = CancellationPredictor()
    
    print("  -> Training Fraud Detector...")
    try:
        fraud_clf.train(df)
        df_fraud = fraud_clf.predict_fraud_probability(df)
        df = df.join(df_fraud)
    except Exception as e:
        print(f"  [Fraud Error] {e}")

    print("  -> Training Cancellation Predictor...")
    try:
        cancel_clf.train(df)
        df_cancel = cancel_clf.predict(df)
        df = df.join(df_cancel)
    except Exception as e:
        print(f"  [Cancel Error] {e}")

    print("  -> Generating Composite Risk Scores...")
    try:
        scorer = CompositeRiskScorer()
        # Ensure all risk columns exist before scoring
        risk_cols = ['late_delivery_probability', 'fraud_probability', 'cancellation_probability']
        for c in risk_cols:
            if c not in df.columns:
                df[c] = 0.0

        risk_df = scorer.score(
            df,
            p_late=df['late_delivery_probability'],
            p_fraud=df['fraud_probability'],
            p_cancel=df['cancellation_probability']
        )
        # Merge risk results back to master df (only new columns)
        new_risk_cols = [c for c in risk_df.columns if c not in df.columns]
        df = df.join(risk_df[new_risk_cols])
        scorer.print_risk_summary(risk_df)
    except Exception as e:
        print(f"  [Composite Risk Error] {e}")
    print()

    # ── PHASE 3: Causal What-If Analysis ──────────────────────────────────
    print("PHASE 3: CAUSAL WHAT-IF ANALYSIS")
    try:
        from causal.causal_inference import SupplyChainCausalAnalyzer
        causal_analyzer = SupplyChainCausalAnalyzer()
        causal_res = causal_analyzer.run_full_causal_analysis(df)
        fraud_causal_res = causal_analyzer.run_fraud_causal_analysis(df)
        print("  -> Causal analysis complete.")
    except Exception as e:
        print(f"  [Causal Error] {e}")
    print()

    # ── PHASE 4: Temporal Deep Learning Upgrade ───────────────────────────
    print("PHASE 4: TEMPORAL LSTM & HYBRID FORECASTING")
    try:
        from demand_forecasting_lstm import LSTMDemandForecaster
        lstm_forecaster = LSTMDemandForecaster()
        lstm_history = lstm_forecaster.train(df, epochs=5, batch_size=256)
        print("  -> LSTM Model training complete.")
    except Exception as e:
        print(f"  [LSTM Error] {e}")
    print()

    # ── PHASE 5: Financial Impact & Cost-Sensitive Modeling ───────────────
    print("PHASE 5: FINANCIAL & ESG OPTIMIZATION")
    try:
        if 'Late_delivery_risk' in df.columns:
            optimizer = CostSensitiveOptimizer()
            optimizer.compute_costs_from_data(df)
            optimizer.find_optimal_threshold(df['Late_delivery_risk'].values, df['Late_delivery_risk'].values) # Mock true vs proba
        
        pareto = ParetoOptimizer()
        pareto_df = pareto.compute_regional_pareto_front(df)
        pareto.plot_pareto_frontier(pareto_df)
        
        esg = ESGScorer()
        esg.regional_esg_summary(df)
    except Exception as e:
        print(f"  [Financial Error] {e}")
    print()

    # ── PHASE 6: Dynamic Resilience & Cluster Enhancement ─────────────────
    print("PHASE 6: RESILIENCE & DRIFT DETECTION")
    try:
        if 'cluster_label' in df.columns:
            resilience = ResilienceIndexCalculator()
            res_df = resilience.compute_cluster_metrics(df)
            res_index = resilience.compute_resilience_index(res_df)
            resilience.plot_resilience_radar(res_index)
            
            drift = ClusterDriftDetector()
            hist = drift.assign_quarterly_clusters(df, [])
            if not hist.empty:
                drift.detect_drift(hist)
                drift.plot_drift_timeline(hist)
    except Exception as e:
        print(f"  [Resilience/Drift Error] {e}")
    print()

    # ── Save Insights ─────────────────────────────────────────────────────
    print("SAVING FINAL MASTER INSIGHTS")
    metadata = save_models(
        global_forecast_model=global_forecast_model,
        regional_forecast_models=regional_forecast_models if isinstance(regional_forecast_models, dict) else {},
        global_classifier=global_classifier,
        regional_classifiers=regional_classifiers if isinstance(regional_classifiers, dict) else {},
        feature_cols_forecast=forecast_feature_cols,
        feature_cols_classify=classify_feature_cols,
        global_forecast_metrics=global_forecast_metrics if isinstance(global_forecast_metrics, dict) else {},
        regional_forecast_perf=regional_forecast_perf,
        global_classify_metrics={},
        regional_classify_perf=regional_classify_perf,
        clustering_metrics=clustering_metrics
    )

    insights = generate_insights(
        df, global_forecast_metrics, regional_forecast_perf,
        {}, regional_classify_perf, clustering_metrics
    )

    elapsed = time.time() - start
    print("\n" + "=" * 80)
    print("ADVANCED PIPELINE COMPLETE")
    print("=" * 80)
    print(f"  Total rows processed:  {df.shape[0]:,}")
    print(f"  Plots saved in:        plots/")
    print(f"  Models saved in:       models/")
    print(f"  XAI Report:            XAI_Report.html")
    print(f"  Elapsed time:          {elapsed:.1f} seconds")
    print("=" * 80)


if __name__ == '__main__':
    main()
