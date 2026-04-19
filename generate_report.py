"""
==========================================================================
ML Pipeline Report Generator
Runs the full pipeline, collects all outputs, metrics, and predictions,
then writes a comprehensive HTML summary document.

Usage:
    python generate_report.py
==========================================================================
"""

import os
import sys
import time
import json
import io
import contextlib
from datetime import datetime

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)

REPORT_FILE = 'Model_Efficiency_Report.html'


# ─── Helpers ────────────────────────────────────────────────────────────────

class MetricsCollector:
    """Collects all metrics from every pipeline phase into one place."""

    def __init__(self):
        self.dataset_info = {}
        self.feature_engineering = {}
        self.statistical_tests = {}
        self.clustering = {}
        self.forecast_global = {}
        self.forecast_regional = pd.DataFrame()
        self.forecast_ensemble = {}
        self.classify_global = {}
        self.classify_regional = pd.DataFrame()
        self.verification = {}
        self.predictions_sample = {}
        self.insights = []
        self.logs = []

    def log(self, msg):
        self.logs.append(msg)
        print(msg)


def capture_stdout(func, *args, **kwargs):
    """Run a function and capture its stdout alongside the return value."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = func(*args, **kwargs)
    return result, buf.getvalue()


# ─── Pipeline Runner ────────────────────────────────────────────────────────

def run_pipeline(mc: MetricsCollector):
    """Execute the full ML pipeline and collect all outputs into mc."""

    # ── Phase 1: Preprocessing ───────────────────────────────────────────
    mc.log("PHASE 1: Data Preprocessing & Feature Engineering")
    from data_preprocessing import prepare_full_dataset
    df = prepare_full_dataset()

    mc.dataset_info = {
        'rows': df.shape[0],
        'columns': df.shape[1],
        'column_list': df.columns.tolist(),
        'dtypes': df.dtypes.value_counts().to_dict(),
        'missing_total': int(df.isna().sum().sum()),
        'regions': sorted(df['Macro_Region'].unique().tolist()) if 'Macro_Region' in df.columns else [],
        'date_range': '',
    }
    date_col = 'order date (DateOrders)'
    if date_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[date_col]):
        mc.dataset_info['date_range'] = (
            f"{df[date_col].min().strftime('%Y-%m-%d')} to {df[date_col].max().strftime('%Y-%m-%d')}"
        )

    mc.feature_engineering = {
        'geographic': ['Macro_Region', 'Shipping_Distance_KM', 'Is_Cross_Border'],
        'temporal': ['Order_Month', 'Order_Year', 'Order_DayOfWeek', 'Season',
                     'Weather_Risk_Score', 'Holiday_Flag', 'Is_Weekend'],
        'regional_agg': [c for c in df.columns if c.startswith('Region_')],
    }

    # ── Phase 2: EDA & Statistical Tests ─────────────────────────────────
    mc.log("PHASE 2: Exploratory Data Analysis")
    from geographic_analysis import run_eda
    stat_results = run_eda(df)
    mc.statistical_tests = stat_results or {}

    # ── Phase 3: Clustering ──────────────────────────────────────────────
    mc.log("PHASE 3: Geographic Clustering")
    from clustering import run_clustering
    df, clustering_metrics = run_clustering(df)
    mc.clustering = clustering_metrics

    # ── Phase 4: Demand Forecasting ──────────────────────────────────────
    mc.log("PHASE 4: Regional Demand Forecasting")
    from demand_forecasting import (
        run_demand_forecasting, build_regional_time_series, get_feature_columns
    )
    df, global_forecast, regional_forecasts, global_f_metrics, regional_f_perf = \
        run_demand_forecasting(df)

    mc.forecast_global = global_f_metrics if isinstance(global_f_metrics, dict) else {}
    mc.forecast_regional = regional_f_perf if isinstance(regional_f_perf, pd.DataFrame) else pd.DataFrame()

    # Ensemble metrics (recompute from saved comparison)
    if isinstance(regional_f_perf, pd.DataFrame) and not regional_f_perf.empty and mc.forecast_global:
        from demand_forecasting import evaluate_ensemble
        df_ts = build_regional_time_series(df)
        if df_ts is not None:
            feat_cols = get_feature_columns(df_ts)
            mc.forecast_ensemble = evaluate_ensemble(
                df_ts, feat_cols, global_forecast,
                regional_forecasts if isinstance(regional_forecasts, dict) else {}
            )

    # ── Phase 5: Late Delivery Classification ────────────────────────────
    mc.log("PHASE 5: Late Delivery Classification")
    from late_delivery_classifier import run_late_delivery_classification
    df, global_clf, regional_clfs, regional_c_perf = run_late_delivery_classification(df)

    if global_clf is not None:
        from sklearn.model_selection import train_test_split
        from late_delivery_classifier import prepare_classification_features
        from sklearn.metrics import f1_score, roc_auc_score
        df_class, clf_feat = prepare_classification_features(df)
        X_c = df_class[clf_feat].values
        y_c = df_class['Late_delivery_risk'].astype(int).values
        _, X_te, _, y_te = train_test_split(X_c, y_c, test_size=0.2, random_state=42, stratify=y_c)
        y_pred_c = global_clf.predict(X_te)
        y_proba_c = global_clf.predict_proba(X_te)[:, 1]
        mc.classify_global = {
            'F1': float(f1_score(y_te, y_pred_c)),
            'ROC_AUC': float(roc_auc_score(y_te, y_proba_c)),
            'Test_Size': len(X_te),
        }
    mc.classify_regional = regional_c_perf if isinstance(regional_c_perf, pd.DataFrame) else pd.DataFrame()

    # ── Phase 6: Verification Metrics ────────────────────────────────────
    mc.log("PHASE 6: Model Verification")
    from model_verification import (
        cv_forecast, cv_classifier, overfitting_check, residual_analysis,
        plot_precision_recall, plot_calibration, plot_learning_curve,
        plot_permutation_importance, regional_stability_report
    )
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier

    verification = {}

    # Forecast CV
    df_ts = build_regional_time_series(df)
    if df_ts is not None:
        feat_cols = get_feature_columns(df_ts)
        cv_f_df = cv_forecast(
            xgb.XGBRegressor,
            dict(n_estimators=200, max_depth=6, learning_rate=0.05,
                 subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0),
            df_ts, feat_cols, 'Sales', n_splits=5
        )
        verification['forecast_cv'] = cv_f_df.to_dict('records')

        # Overfitting check (forecast)
        df_clean = df_ts.dropna(subset=feat_cols + ['Sales']).sort_values('Date')
        X_all = df_clean[feat_cols].values
        y_all = df_clean['Sales'].values
        split = int(len(df_clean) * 0.7)
        X_tr, X_te = X_all[:split], X_all[split:]
        y_tr, y_te = y_all[:split], y_all[split:]

        import joblib
        gf_model = joblib.load('models/regional_forecast/global_xgboost_forecast.pkl')
        overfit_reg = overfitting_check(gf_model, X_tr, y_tr, X_te, y_te, task='regression')
        verification['forecast_overfit'] = overfit_reg

        # Residuals
        y_pred_f = gf_model.predict(X_te)
        residual_analysis(y_te, y_pred_f, label='Demand Forecasting')

        # Learning curve, permutation importance
        plot_learning_curve(
            xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                              subsample=0.8, random_state=42, verbosity=0),
            X_all, y_all, task='regression', cv=3, label='Demand Forecasting'
        )
        plot_permutation_importance(gf_model, X_te, y_te, feat_cols,
                                    task='regression', label='Global Forecast')

        # Regional stability (forecast)
        regional_stability_report(mc.forecast_regional, task='forecast')

    # Classifier CV
    if global_clf is not None:
        cv_c_df = cv_classifier(
            RandomForestClassifier,
            dict(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1),
            X_c, y_c, n_splits=5
        )
        verification['classifier_cv'] = cv_c_df.to_dict('records')

        # Overfitting check (classifier)
        _, X_te_c, _, y_te_c = train_test_split(X_c, y_c, test_size=0.2,
                                                 random_state=42, stratify=y_c)
        X_tr_c = X_c[: len(X_c) - len(X_te_c)]
        y_tr_c = y_c[: len(y_c) - len(y_te_c)]
        overfit_clf = overfitting_check(global_clf, X_tr_c, y_tr_c, X_te_c, y_te_c,
                                        task='classification')
        verification['classifier_overfit'] = overfit_clf

        y_proba_test = global_clf.predict_proba(X_te_c)[:, 1]
        plot_precision_recall(y_te_c, y_proba_test)
        plot_calibration(y_te_c, y_proba_test)

        plot_learning_curve(
            RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1),
            X_c, y_c, task='classification', cv=3, label='Late Delivery Classifier'
        )
        plot_permutation_importance(global_clf, X_te_c, y_te_c, clf_feat,
                                    task='classification', label='Late Delivery Classifier')

        regional_stability_report(mc.classify_regional, task='classifier')

    mc.verification = verification

    # ── Sample Predictions ───────────────────────────────────────────────
    mc.log("Generating sample predictions...")
    if global_forecast is not None and isinstance(regional_forecasts, dict):
        from model_pipeline import RegionalDemandPredictor
        try:
            predictor = RegionalDemandPredictor()
            meta = json.load(open('models/regional_forecast/metadata.json'))
            sample_f = {col: 0.0 for col in meta.get('forecast_features', [])}
            sample_f.update({'DayOfWeek': 1, 'Month': 3, 'Is_Weekend': 0,
                             'Sales_lag_1': 5000, 'Sales_lag_7': 4500, 'Sales_roll_7d': 4800})
            mc.predictions_sample = predictor.predict_all_regions(sample_f)
        except Exception as e:
            mc.log(f"  Sample predictions failed: {e}")

    # ── Business Insights ────────────────────────────────────────────────
    from model_pipeline import generate_insights
    mc.insights = generate_insights(
        df, mc.forecast_global, mc.forecast_regional,
        mc.classify_global, mc.classify_regional, mc.clustering
    )

    return df


# ─── HTML Report Builder ────────────────────────────────────────────────────

def _metric_row(label, value, fmt='.4f'):
    """One table row."""
    if isinstance(value, float):
        v = f'{value:{fmt}}'
    else:
        v = str(value)
    return f'<tr><td>{label}</td><td><strong>{v}</strong></td></tr>\n'


def _dataframe_to_html(df, max_rows=50):
    """Render a dataframe to an HTML table string."""
    return df.head(max_rows).to_html(index=False, classes='dataframe', border=0,
                                     float_format='%.4f')


def _section(title, body, level=2):
    tag = f'h{level}'
    return f'<{tag}>{title}</{tag}>\n{body}\n'


def _card(title, content):
    return (f'<div class="card"><div class="card-title">{title}</div>'
            f'<div class="card-body">{content}</div></div>\n')


def _image_tag(path):
    """Return <img> tag if the file exists, else a placeholder."""
    if os.path.exists(path):
        return f'<img src="{path}" alt="{os.path.basename(path)}">'
    return f'<p class="missing">[Plot not found: {path}]</p>'


def build_html(mc: MetricsCollector, elapsed: float):
    """Build the full HTML report from collected metrics."""

    plots = []
    for root, _, files in os.walk('plots'):
        if 'verification' in root: continue
        for f in sorted(files):
            if f.endswith('.png') and not f.startswith('verification'):
                plots.append(os.path.join(root, f).replace('\\', '/'))
    v_plots  = []
    vdir = 'plots/verification'
    if os.path.isdir(vdir):
        v_plots = [f'{vdir}/{f}' for f in sorted(os.listdir(vdir)) if f.endswith('.png')]

    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ML Pipeline Report - DataCo Supply Chain</title>
<style>
  :root {{ --bg: #f5f7fa; --card-bg: #fff; --accent: #2c6fbb; --text: #222; --muted: #666; }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', Tahoma, sans-serif; background: var(--bg); color: var(--text);
         line-height: 1.6; padding: 2rem; max-width: 1200px; margin: auto; }}
  h1 {{ color: var(--accent); border-bottom: 3px solid var(--accent); padding-bottom: .4rem;
       margin-bottom: 1rem; }}
  h2 {{ color: var(--accent); margin-top: 2rem; border-left: 4px solid var(--accent);
       padding-left: .6rem; }}
  h3 {{ color: #444; margin-top: 1.2rem; }}
  .meta {{ color: var(--muted); font-size: .9rem; margin-bottom: 1.5rem; }}
  .card {{ background: var(--card-bg); border-radius: 8px; padding: 1.2rem;
           margin: .8rem 0; box-shadow: 0 1px 4px rgba(0,0,0,.1); }}
  .card-title {{ font-weight: 700; font-size: 1.05rem; color: var(--accent);
                 margin-bottom: .6rem; border-bottom: 1px solid #eee; padding-bottom: .3rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: .5rem 0; font-size: .92rem; }}
  th, td {{ text-align: left; padding: .45rem .7rem; border-bottom: 1px solid #e0e0e0; }}
  th {{ background: #eef3fa; color: #333; }}
  tr:hover {{ background: #f9fbfe; }}
  .dataframe th {{ background: #eef3fa; }}
  img {{ max-width: 100%; height: auto; border-radius: 6px; margin: .5rem 0; }}
  .gallery {{ display: flex; flex-wrap: wrap; gap: 1rem; }}
  .gallery img {{ width: 48%; }}
  .verdict {{ padding: .6rem 1rem; border-radius: 6px; margin: .5rem 0; font-weight: 600; }}
  .pass {{ background: #d4edda; color: #155724; }}
  .warn {{ background: #fff3cd; color: #856404; }}
  .fail {{ background: #f8d7da; color: #721c24; }}
  .missing {{ color: #999; font-style: italic; }}
  .insight {{ background: #eef3fa; padding: .4rem .8rem; border-radius: 4px;
              margin: .3rem 0; }}
  .toc {{ background: var(--card-bg); padding: 1rem 1.5rem; border-radius: 8px;
          margin-bottom: 1.5rem; box-shadow: 0 1px 4px rgba(0,0,0,.1); }}
  .toc a {{ text-decoration: none; color: var(--accent); }}
  .toc a:hover {{ text-decoration: underline; }}
  .toc li {{ margin: .2rem 0; }}
</style>
</head>
<body>

<h1>ML Pipeline Efficiency Report</h1>
<p class="meta">
  <strong>Dataset:</strong> DataCo Supply Chain &nbsp;|&nbsp;
  <strong>Generated:</strong> {now} &nbsp;|&nbsp;
  <strong>Pipeline Runtime:</strong> {elapsed:.1f}s
</p>

<div class="toc">
<strong>Table of Contents</strong>
<ol>
<li><a href="#s1">Dataset Overview</a></li>
<li><a href="#s2">Feature Engineering Summary</a></li>
<li><a href="#s3">Statistical Tests</a></li>
<li><a href="#s4">Clustering Results</a></li>
<li><a href="#s5">Demand Forecasting Performance</a></li>
<li><a href="#s6">Late Delivery Classifier Performance</a></li>
<li><a href="#s7">Model Verification &amp; Efficiency</a></li>
<li><a href="#s8">Sample Predictions</a></li>
<li><a href="#s9">Advanced Intelligence (Phases 1-6)</a></li>
<li><a href="#s10">Business Insights</a></li>
<li><a href="#s11">Visualizations</a></li>
<li><a href="#s12">Conclusion &amp; Verdict</a></li>
</ol>
</div>
"""

    # ── 1. Dataset Overview ──────────────────────────────────────────────
    di = mc.dataset_info
    html += f'<h2 id="s1">1. Dataset Overview</h2>\n'
    html += _card('Dataset Statistics', f"""
<table>
{_metric_row('Total Rows', f"{di.get('rows',0):,}", 's')}
{_metric_row('Total Columns', di.get('columns',0), 'd')}
{_metric_row('Missing Values', di.get('missing_total',0), 'd')}
{_metric_row('Regions', len(di.get('regions',[])), 'd')}
{_metric_row('Date Range', di.get('date_range','N/A'), 's')}
</table>
<p style="margin-top:.5rem; color:#666;">Regions: {', '.join(di.get('regions',[]))}</p>
""")

    # ── 2. Feature Engineering ───────────────────────────────────────────
    fe = mc.feature_engineering
    html += f'<h2 id="s2">2. Feature Engineering Summary</h2>\n'
    feat_body = '<table><tr><th>Category</th><th>Features</th></tr>\n'
    for cat, feats in fe.items():
        feat_body += f'<tr><td>{cat.replace("_"," ").title()}</td><td>{", ".join(feats)}</td></tr>\n'
    feat_body += '</table>'
    html += _card('Engineered Features', feat_body)

    # ── 3. Statistical Tests ─────────────────────────────────────────────
    html += f'<h2 id="s3">3. Statistical Tests</h2>\n'
    st = mc.statistical_tests
    if st:
        st_body = '<table><tr><th>Test</th><th>Statistic</th><th>p-value</th><th>Result</th></tr>\n'
        for name, vals in st.items():
            stat_val = [v for k, v in vals.items() if k != 'p-value' and k != 'dof']
            stat_str = ', '.join(f'{v:.4f}' for v in stat_val) if stat_val else 'N/A'
            p = vals.get('p-value', 1.0)
            sig = '<span style="color:green">Significant</span>' if p < 0.05 else 'Not Significant'
            st_body += f'<tr><td>{name}</td><td>{stat_str}</td><td>{p:.6f}</td><td>{sig}</td></tr>\n'
        st_body += '</table>'
        html += _card('Hypothesis Tests', st_body)
    else:
        html += _card('Hypothesis Tests', '<p class="missing">No statistical test results.</p>')

    # ── 4. Clustering ────────────────────────────────────────────────────
    html += f'<h2 id="s4">4. Clustering Results</h2>\n'
    cl = mc.clustering
    cl_body = '<table><tr><th>Method</th><th>Metric</th><th>Value</th></tr>\n'
    if 'kmeans' in cl:
        for k, v in cl['kmeans'].items():
            cl_body += f'<tr><td>K-Means</td><td>{k}</td><td>{v:.4f}</td></tr>\n'
    if 'dbscan' in cl:
        for k, v in cl['dbscan'].items():
            cl_body += f'<tr><td>DBSCAN</td><td>{k}</td><td>{v}</td></tr>\n'
    if 'hierarchical' in cl:
        for k, v in cl['hierarchical'].items():
            cl_body += f'<tr><td>Hierarchical</td><td>{k}</td><td>{v}</td></tr>\n'
    cl_body += '</table>'
    html += _card('Clustering Metrics', cl_body)

    # ── 5. Demand Forecasting ────────────────────────────────────────────
    html += f'<h2 id="s5">5. Demand Forecasting Performance</h2>\n'

    fg = mc.forecast_global
    fe_ens = mc.forecast_ensemble
    if fg:
        comp_body = '<table><tr><th>Model</th><th>MAE</th><th>RMSE</th><th>R2</th></tr>\n'
        comp_body += f'<tr><td>Global XGBoost</td><td>{fg["MAE"]:,.2f}</td><td>{fg["RMSE"]:,.2f}</td><td>{fg["R2"]:.4f}</td></tr>\n'
        if not mc.forecast_regional.empty:
            avg = mc.forecast_regional[['MAE', 'RMSE', 'R2']].mean()
            comp_body += f'<tr><td>Regional Avg</td><td>{avg["MAE"]:,.2f}</td><td>{avg["RMSE"]:,.2f}</td><td>{avg["R2"]:.4f}</td></tr>\n'
        if fe_ens:
            comp_body += f'<tr style="font-weight:700"><td>Ensemble</td><td>{fe_ens["MAE"]:,.2f}</td><td>{fe_ens["RMSE"]:,.2f}</td><td>{fe_ens["R2"]:.4f}</td></tr>\n'
        comp_body += '</table>'
        if fg["MAE"] > 0 and fe_ens:
            imp = ((fg["MAE"] - fe_ens["MAE"]) / fg["MAE"]) * 100
            comp_body += f'<p><strong>Ensemble improvement over Global: {imp:+.1f}% MAE reduction</strong></p>'
        html += _card('Model Comparison', comp_body)

    if not mc.forecast_regional.empty:
        html += _card('Regional Forecast Breakdown', _dataframe_to_html(mc.forecast_regional))

    # ── 6. Classification ────────────────────────────────────────────────
    html += f'<h2 id="s6">6. Late Delivery Classifier Performance</h2>\n'
    gc = mc.classify_global
    if gc:
        c_body = '<table><tr><th>Metric</th><th>Global</th><th>Regional Avg</th></tr>\n'
        rc_avg_f1 = mc.classify_regional['F1'].mean() if not mc.classify_regional.empty else 0
        rc_avg_auc = mc.classify_regional['ROC_AUC'].mean() if not mc.classify_regional.empty else 0
        c_body += f'<tr><td>F1 Score</td><td>{gc["F1"]:.4f}</td><td>{rc_avg_f1:.4f}</td></tr>\n'
        c_body += f'<tr><td>ROC-AUC</td><td>{gc["ROC_AUC"]:.4f}</td><td>{rc_avg_auc:.4f}</td></tr>\n'
        c_body += '</table>'
        html += _card('Classifier Comparison', c_body)

    if not mc.classify_regional.empty:
        html += _card('Regional Classifier Breakdown', _dataframe_to_html(mc.classify_regional))

    # ── 7. Verification ──────────────────────────────────────────────────
    html += f'<h2 id="s7">7. Model Verification &amp; Efficiency</h2>\n'

    v = mc.verification

    # Forecast CV
    if 'forecast_cv' in v:
        cv_df = pd.DataFrame(v['forecast_cv'])
        cv_body = _dataframe_to_html(cv_df)
        cv_body += f"""<p><strong>CV Mean MAE:</strong> {cv_df['MAE'].mean():,.1f} &plusmn; {cv_df['MAE'].std():,.1f} &nbsp;|&nbsp;
                     <strong>CV Mean R2:</strong> {cv_df['R2'].mean():.4f} &plusmn; {cv_df['R2'].std():.4f}</p>"""
        html += _card('Demand Forecast - 5-Fold Time-Series CV', cv_body)

    # Classifier CV
    if 'classifier_cv' in v:
        cv_c_df = pd.DataFrame(v['classifier_cv'])
        cv_c_body = _dataframe_to_html(cv_c_df)
        cv_c_body += f"""<p><strong>CV Mean F1:</strong> {cv_c_df['F1'].mean():.4f} &plusmn; {cv_c_df['F1'].std():.4f} &nbsp;|&nbsp;
                       <strong>CV Mean AUC:</strong> {cv_c_df['ROC_AUC'].mean():.4f} &plusmn; {cv_c_df['ROC_AUC'].std():.4f}</p>"""
        html += _card('Late Delivery Classifier - 5-Fold Stratified CV', cv_c_body)

    # Overfitting
    if 'forecast_overfit' in v:
        of = v['forecast_overfit']
        of_body = '<table><tr><th>Metric</th><th>Train</th><th>Test</th><th>Gap</th><th>Verdict</th></tr>\n'
        for m in of.get('train', {}):
            tr = of['train'][m]
            te = of['test'][m]
            gap = tr - te
            verdict = 'OVERFIT' if abs(gap) > 0.15 else 'OK'
            css = 'warn' if verdict == 'OVERFIT' else 'pass'
            of_body += f'<tr><td>{m}</td><td>{tr:.4f}</td><td>{te:.4f}</td><td>{gap:+.4f}</td><td class="{css}">{verdict}</td></tr>\n'
        of_body += '</table>'
        html += _card('Overfitting Check - Demand Forecasting', of_body)

    if 'classifier_overfit' in v:
        of = v['classifier_overfit']
        of_body = '<table><tr><th>Metric</th><th>Train</th><th>Test</th><th>Gap</th><th>Verdict</th></tr>\n'
        for m in of.get('train', {}):
            tr = of['train'][m]
            te = of['test'][m]
            gap = tr - te
            verdict = 'OVERFIT' if abs(gap) > 0.05 else 'OK'
            css = 'warn' if verdict == 'OVERFIT' else 'pass'
            of_body += f'<tr><td>{m}</td><td>{tr:.4f}</td><td>{te:.4f}</td><td>{gap:+.4f}</td><td class="{css}">{verdict}</td></tr>\n'
        of_body += '</table>'
        html += _card('Overfitting Check - Late Delivery Classifier', of_body)

    # Verification plots
    if v_plots:
        html += '<h3>Verification Plots</h3>\n<div class="gallery">\n'
        for p in v_plots:
            html += _image_tag(p) + '\n'
        html += '</div>\n'

    # ── 8. Sample Predictions ────────────────────────────────────────────
    html += f'<h2 id="s8">8. Sample Predictions (Demo Input)</h2>\n'
    if mc.predictions_sample:
        pred_body = '<table><tr><th>Region</th><th>Predicted Daily Demand ($)</th></tr>\n'
        for region, val in sorted(mc.predictions_sample.items()):
            pred_body += f'<tr><td>{region}</td><td>${val:,.2f}</td></tr>\n'
        pred_body += '</table>'
        html += _card('Regional Demand Predictions', pred_body)
    else:
        html += _card('Regional Demand Predictions', '<p class="missing">No predictions available.</p>')

    # ── 9. Advanced Intelligence ─────────────────────────────────────────
    html += f'<h2 id="s9">9. Advanced Intelligence (Phases 1-6)</h2>\n'
    html += _card('Advanced Extensions', """
    <p>This pipeline incorporates advanced operations including Explainability (SHAP/LIME),
    Multi-Risk Assessment (CatBoost), Causal Inference (DoWhy), Deep Learning (LSTM), and
    Multi-Objective Financial Pareto optimization.</p>
    <ul>
        <li><a href="XAI_Report.html" target="_blank" style="color:var(--accent); font-weight:bold;">View Dedicated XAI Report (SHAP & LIME)</a></li>
    </ul>
    """)

    # ── 10. Insights ──────────────────────────────────────────────────────
    html += f'<h2 id="s10">10. Business Insights</h2>\n'
    if mc.insights:
        ins_body = ''
        for i, ins in enumerate(mc.insights, 1):
            ins_body += f'<div class="insight"><strong>{i}.</strong> {ins}</div>\n'
        html += _card('Automated Insights', ins_body)

    # ── 11. All Visualizations ───────────────────────────────────────────
    html += f'<h2 id="s11">11. Visualizations</h2>\n'
    html += '<h3>EDA, Model, and Advanced Plots</h3>\n<div class="gallery">\n'
    for p in plots:
        html += _image_tag(p) + '\n'
    html += '</div>\n'

    # ── 12. Conclusion ───────────────────────────────────────────────────
    html += f'<h2 id="s12">12. Conclusion &amp; Overall Verdict</h2>\n'

    # Build verdict
    verdicts = []
    # Forecasting
    r2 = mc.forecast_ensemble.get('R2', mc.forecast_global.get('R2', 0))
    if r2 > 0.8:
        verdicts.append(('Demand Forecasting', 'STRONG', 'pass', f'R2={r2:.4f}'))
    elif r2 > 0.6:
        verdicts.append(('Demand Forecasting', 'MODERATE', 'warn', f'R2={r2:.4f}'))
    else:
        verdicts.append(('Demand Forecasting', 'NEEDS IMPROVEMENT', 'fail', f'R2={r2:.4f}'))

    # Classification
    clf_f1 = mc.classify_global.get('F1', 0)
    if clf_f1 > 0.9:
        verdicts.append(('Late Delivery Classifier', 'STRONG', 'pass', f'F1={clf_f1:.4f}'))
    elif clf_f1 > 0.75:
        verdicts.append(('Late Delivery Classifier', 'MODERATE', 'warn', f'F1={clf_f1:.4f}'))
    else:
        verdicts.append(('Late Delivery Classifier', 'NEEDS IMPROVEMENT', 'fail', f'F1={clf_f1:.4f}'))

    # CV stability
    if 'forecast_cv' in v:
        cv_std = pd.DataFrame(v['forecast_cv'])['R2'].std()
        if cv_std < 0.05:
            verdicts.append(('Forecast CV Stability', 'STABLE', 'pass', f'R2 std={cv_std:.4f}'))
        else:
            verdicts.append(('Forecast CV Stability', 'VARIABLE', 'warn', f'R2 std={cv_std:.4f}'))

    v_body = '<table><tr><th>Component</th><th>Verdict</th><th>Key Metric</th></tr>\n'
    for comp, verdict, css, metric in verdicts:
        v_body += f'<tr><td>{comp}</td><td class="verdict {css}">{verdict}</td><td>{metric}</td></tr>\n'
    v_body += '</table>'
    html += _card('Overall Model Efficiency Verdict', v_body)

    html += f"""
<p class="meta" style="margin-top: 2rem;">
  Report generated on {now} | Pipeline runtime: {elapsed:.1f}s |
  {di.get('rows',0):,} rows processed | {len(plots)+len(v_plots)} plots generated
</p>
</body></html>"""

    return html


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    start = time.time()

    print("=" * 70)
    print("  ML PIPELINE REPORT GENERATOR")
    print("=" * 70)

    mc = MetricsCollector()
    df = run_pipeline(mc)

    elapsed = time.time() - start

    mc.log(f"\nBuilding HTML report...")
    html = build_html(mc, elapsed)

    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\nReport saved to: {os.path.abspath(REPORT_FILE)}")
    print(f"Total pipeline time: {elapsed:.1f}s")
    print("=" * 70)


if __name__ == '__main__':
    main()
