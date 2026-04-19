# Supply Chain ML Project — Extended Implementation Plan

> **Base project:** DataCo Smart Supply Chain · Geographic Analysis & Regional Demand Forecasting  
> **Extension scope:** 6 phases, 18 new features, grounded in 10 reviewed papers  
> **Estimated timeline:** 13 weeks

---

## Table of Contents

1. [Phase 1 — Explainability & Trust Layer](#phase-1--explainability--trust-layer)
2. [Phase 2 — Multi-Risk Prediction Engine](#phase-2--multi-risk-prediction-engine)
3. [Phase 3 — Causal What-If Analysis](#phase-3--causal-what-if-analysis)
4. [Phase 4 — Temporal Deep Learning Upgrade](#phase-4--temporal-deep-learning-upgrade)
5. [Phase 5 — Financial Impact & Cost-Sensitive Modeling](#phase-5--financial-impact--cost-sensitive-modeling)
6. [Phase 6 — Dynamic Resilience & Cluster Enhancement](#phase-6--dynamic-resilience--cluster-enhancement)
7. [Updated Project Structure](#updated-project-structure)
8. [Dependencies Reference](#dependencies-reference)

---

## Phase 1 — Explainability & Trust Layer

**Timeline:** Weeks 1–2  
**Complexity:** Low  
**Research justification:** Zogaan et al. (2025) identify lack of interpretability as the primary limitation of DL/hybrid supply chain models. Camur et al. (2024) flag the same for their GAC-RF model. Yang et al. (2025) demonstrate SHAP for cluster-based resilience but do not extend it to regional late delivery prediction.  
**Key insight:** Your XGBoost and Random Forest models are already trained — this phase adds zero retraining and directly elevates research contribution.

---

### Feature 1.1 — SHAP Global & Local Explanations

**New file:** `explainability/shap_explainer.py`

**What it does:**
- Computes SHAP values for the existing XGBoost demand forecaster and RF late delivery classifier
- Produces global summary plots (which features drive predictions across the whole dataset)
- Produces local waterfall plots (why was this specific order predicted as late?)
- Generates regional feature importance heatmaps (which features matter most per Order Region)

**Implementation:**

```python
# explainability/shap_explainer.py

import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

matplotlib.use('Agg')

class SHAPExplainer:
    """
    Generates SHAP explanations for existing XGBoost and RF models.
    Produces both global summary plots and local per-prediction waterfall plots.
    """

    def __init__(self, model_dir="models/", output_dir="plots/shap/"):
        self.model_dir = model_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def load_models(self):
        """Load existing trained models from the pipeline."""
        self.xgb_demand = joblib.load(f"{self.model_dir}xgb_demand_global.pkl")
        self.rf_classifier = joblib.load(f"{self.model_dir}rf_late_delivery.pkl")
        print("[SHAP] Models loaded successfully.")

    def compute_shap_demand(self, X_train: pd.DataFrame, X_sample: pd.DataFrame = None):
        """
        Compute SHAP values for the XGBoost demand forecaster.
        Uses TreeExplainer (fast, exact for tree models).

        Args:
            X_train: Full training feature matrix
            X_sample: Optional subset for faster computation (use 500–1000 rows)
        """
        if X_sample is None:
            X_sample = X_train.sample(min(500, len(X_train)), random_state=42)

        explainer = shap.TreeExplainer(self.xgb_demand)
        shap_values = explainer.shap_values(X_sample)

        # Global summary plot — bar chart of mean |SHAP| per feature
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        plt.title("Demand Forecaster — Global Feature Importance (SHAP)")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}demand_shap_global.png", dpi=150)
        plt.close()

        # Beeswarm plot — direction and magnitude of each feature
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title("Demand Forecaster — SHAP Beeswarm")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}demand_shap_beeswarm.png", dpi=150)
        plt.close()

        print(f"[SHAP] Demand forecaster plots saved to {self.output_dir}")
        return explainer, shap_values, X_sample

    def compute_shap_classifier(self, X_train: pd.DataFrame, X_sample: pd.DataFrame = None):
        """
        Compute SHAP values for the RF late delivery classifier.

        Returns:
            explainer, shap_values, X_sample
        """
        if X_sample is None:
            X_sample = X_train.sample(min(500, len(X_train)), random_state=42)

        explainer = shap.TreeExplainer(self.rf_classifier)
        shap_values = explainer.shap_values(X_sample)

        # For binary classification, shap_values is a list [class_0, class_1]
        sv_class1 = shap_values[1] if isinstance(shap_values, list) else shap_values

        plt.figure(figsize=(10, 6))
        shap.summary_plot(sv_class1, X_sample, plot_type="bar", show=False)
        plt.title("Late Delivery Classifier — Global Feature Importance (SHAP)")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}classifier_shap_global.png", dpi=150)
        plt.close()

        plt.figure(figsize=(10, 8))
        shap.summary_plot(sv_class1, X_sample, show=False)
        plt.title("Late Delivery Classifier — SHAP Beeswarm")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}classifier_shap_beeswarm.png", dpi=150)
        plt.close()

        print(f"[SHAP] Classifier plots saved to {self.output_dir}")
        return explainer, sv_class1, X_sample

    def explain_single_prediction(self, explainer, shap_values_row: np.ndarray,
                                   X_row: pd.DataFrame, prediction_label: str,
                                   filename: str):
        """
        Generate a waterfall plot for a single order prediction.

        Args:
            explainer: Fitted SHAP TreeExplainer
            shap_values_row: SHAP values for one row (1D array)
            X_row: Feature values for that row (single-row DataFrame)
            prediction_label: Human-readable prediction (e.g., 'Late Delivery Risk: HIGH')
            filename: Output filename (without extension)
        """
        plt.figure(figsize=(10, 5))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values_row,
                base_values=explainer.expected_value if not isinstance(
                    explainer.expected_value, list) else explainer.expected_value[1],
                data=X_row.values[0],
                feature_names=X_row.columns.tolist()
            ),
            show=False
        )
        plt.title(f"Order Explanation — {prediction_label}")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}{filename}.png", dpi=150)
        plt.close()
        print(f"[SHAP] Single prediction waterfall saved: {filename}.png")

    def regional_shap_heatmap(self, X: pd.DataFrame, shap_values: np.ndarray,
                               region_col: str = "Order Region"):
        """
        Aggregate mean |SHAP| values per Order Region and plot a heatmap.
        Shows which features matter most in each geographic region.
        """
        import seaborn as sns

        shap_df = pd.DataFrame(shap_values, columns=X.columns, index=X.index)
        shap_df[region_col] = X[region_col].values

        regional_means = (
            shap_df.groupby(region_col)
            .mean()
            .abs()
        )

        # Keep top 10 most globally important features for readability
        top_features = shap_df.drop(columns=[region_col]).abs().mean().nlargest(10).index
        regional_means = regional_means[top_features]

        plt.figure(figsize=(14, 8))
        sns.heatmap(
            regional_means.T,
            cmap="YlOrRd",
            annot=True, fmt=".3f",
            linewidths=0.5,
            cbar_kws={"label": "Mean |SHAP value|"}
        )
        plt.title("Regional Feature Importance Heatmap (Late Delivery Classifier)")
        plt.xlabel("Order Region")
        plt.ylabel("Feature")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}regional_shap_heatmap.png", dpi=150)
        plt.close()
        print(f"[SHAP] Regional heatmap saved.")
```

**Usage in pipeline:**

```python
# In index.py or a new run_explainability.py

from explainability.shap_explainer import SHAPExplainer

explainer_module = SHAPExplainer()
explainer_module.load_models()

# Global explanations
exp, shap_vals, X_s = explainer_module.compute_shap_classifier(X_train, X_test)

# Explain a single high-risk order
idx = 0  # first order in test set
explainer_module.explain_single_prediction(
    exp, shap_vals[idx], X_test.iloc[[idx]],
    prediction_label="Late Delivery Risk: HIGH",
    filename=f"order_{X_test.index[idx]}_explanation"
)

# Regional heatmap
explainer_module.regional_shap_heatmap(X_test, shap_vals, region_col="Order Region")
```

**Required packages:**
```
shap>=0.44.0
seaborn>=0.12.0
```

---

### Feature 1.2 — LIME for Individual Orders

**New file:** `explainability/lime_explainer.py`

**What it does:**
- Builds a local surrogate linear model around any single prediction
- Produces actionable counterfactuals: "If Shipping Mode were First Class, late delivery probability drops from 0.82 → 0.41"
- Complements SHAP (LIME is model-agnostic and works for any model added later)

**Implementation:**

```python
# explainability/lime_explainer.py

import lime
import lime.lime_tabular
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

class LIMEExplainer:
    """
    Local Interpretable Model-agnostic Explanations for individual order predictions.
    Works with any sklearn-compatible classifier or regressor.
    """

    def __init__(self, feature_names: list, class_names: list = None,
                 output_dir: str = "plots/lime/"):
        self.feature_names = feature_names
        self.class_names = class_names or ["On Time", "Late"]
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def build_explainer(self, X_train: pd.DataFrame, mode: str = "classification"):
        """
        Build the LIME tabular explainer from training data distribution.

        Args:
            X_train: Training features (used to learn data distribution)
            mode: 'classification' or 'regression'
        """
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode=mode,
            random_state=42,
            discretize_continuous=True
        )
        print("[LIME] Explainer built from training data distribution.")
        return self.explainer

    def explain_order(self, model, X_row: pd.DataFrame,
                      order_id: str, num_features: int = 10,
                      save_plot: bool = True) -> dict:
        """
        Generate a LIME explanation for a single order.

        Args:
            model: Trained classifier with predict_proba method
            X_row: Single-row DataFrame with the order's features
            order_id: Identifier for plot filename
            num_features: Number of top features to display

        Returns:
            dict with feature contributions and counterfactual suggestions
        """
        explanation = self.explainer.explain_instance(
            data_row=X_row.values[0],
            predict_fn=model.predict_proba,
            num_features=num_features,
            num_samples=1000
        )

        if save_plot:
            fig = explanation.as_pyplot_figure()
            fig.suptitle(f"LIME Explanation — Order {order_id}", fontsize=12)
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}order_{order_id}_lime.png", dpi=150)
            plt.close()
            print(f"[LIME] Plot saved for order {order_id}")

        # Extract counterfactuals: features where changing value reduces late risk
        feature_weights = explanation.as_list()
        counterfactuals = []
        for feature_condition, weight in feature_weights:
            if weight > 0.05:  # Positive weight = increases late risk
                counterfactuals.append({
                    "feature_condition": feature_condition,
                    "late_risk_contribution": round(weight, 4),
                    "action": f"Modify: {feature_condition} to reduce late risk"
                })

        return {
            "order_id": order_id,
            "predicted_probability": model.predict_proba(X_row.values)[0][1],
            "top_features": feature_weights,
            "counterfactual_suggestions": counterfactuals
        }

    def batch_explain(self, model, X_df: pd.DataFrame,
                      high_risk_only: bool = True, threshold: float = 0.7):
        """
        Run LIME explanations for multiple orders (e.g., all high-risk orders today).

        Args:
            high_risk_only: If True, only explain orders above threshold risk
            threshold: Late delivery probability threshold for 'high risk'
        """
        results = []
        probas = model.predict_proba(X_df.values)[:, 1]

        for i, (idx, row) in enumerate(X_df.iterrows()):
            if high_risk_only and probas[i] < threshold:
                continue
            result = self.explain_order(
                model, X_df.iloc[[i]],
                order_id=str(idx), save_plot=True
            )
            results.append(result)
            if len(results) % 10 == 0:
                print(f"[LIME] Explained {len(results)} high-risk orders...")

        print(f"[LIME] Batch complete. Explained {len(results)} orders.")
        return results
```

---

### Feature 1.3 — Interactive Feature Importance Dashboard

**New file:** `explainability/generate_xai_report.py`

**What it does:**
- Extends the existing `generate_report.py` with an XAI section
- Embeds SHAP plots and LIME summaries into the HTML report
- Adds a per-region importance ranking table

**Implementation:**

```python
# explainability/generate_xai_report.py

import os
import base64
import json
import pandas as pd
from datetime import datetime


def image_to_base64(image_path: str) -> str:
    """Convert a plot image to base64 for embedding in HTML."""
    if not os.path.exists(image_path):
        return ""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def generate_xai_html_section(shap_plot_dir: str = "plots/shap/",
                               lime_results: list = None,
                               output_path: str = "XAI_Report.html"):
    """
    Generate a standalone XAI report embedding all SHAP and LIME outputs.
    Appends to or creates an HTML file.
    """

    plots = {
        "Global Demand SHAP (Bar)": f"{shap_plot_dir}demand_shap_global.png",
        "Global Demand SHAP (Beeswarm)": f"{shap_plot_dir}demand_shap_beeswarm.png",
        "Late Delivery Classifier SHAP (Bar)": f"{shap_plot_dir}classifier_shap_global.png",
        "Late Delivery Classifier SHAP (Beeswarm)": f"{shap_plot_dir}classifier_shap_beeswarm.png",
        "Regional Feature Importance Heatmap": f"{shap_plot_dir}regional_shap_heatmap.png",
    }

    # Build HTML
    sections_html = ""
    for title, path in plots.items():
        b64 = image_to_base64(path)
        if b64:
            sections_html += f"""
            <div class="plot-card">
                <h3>{title}</h3>
                <img src="data:image/png;base64,{b64}" style="max-width:100%; border-radius:8px;">
            </div>
            """

    # LIME counterfactual table
    lime_table = ""
    if lime_results:
        rows = ""
        for res in lime_results[:20]:  # Show top 20
            prob = round(res["predicted_probability"] * 100, 1)
            top_feat = res["top_features"][0][0] if res["top_features"] else "N/A"
            rows += f"<tr><td>{res['order_id']}</td><td>{prob}%</td><td>{top_feat}</td></tr>"

        lime_table = f"""
        <h2>LIME — High-Risk Order Explanations (Top 20)</h2>
        <table>
            <thead><tr><th>Order ID</th><th>Late Risk %</th><th>Top Driver</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>
        """

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>XAI Report — Supply Chain ML</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1100px; margin: auto; padding: 40px; background: #fafafa; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 40px; }}
        h3 {{ color: #555; font-size: 14px; margin: 10px 0; }}
        .plot-card {{ background: white; border-radius: 8px; padding: 20px; margin: 20px 0;
                      box-shadow: 0 1px 4px rgba(0,0,0,0.1); }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th {{ background: #3498db; color: white; padding: 10px 14px; text-align: left; }}
        td {{ border: 1px solid #ddd; padding: 9px 14px; }}
        tr:nth-child(even) {{ background: #f5f5f5; }}
        .meta {{ color: #888; font-size: 13px; margin-bottom: 30px; }}
    </style>
</head>
<body>
    <h1>Explainability & Trust Layer — XAI Report</h1>
    <p class="meta">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} |
       Dataset: DataCo Smart Supply Chain</p>

    <h2>SHAP Analysis — Global Model Explanations</h2>
    <p>SHAP (SHapley Additive exPlanations) decomposes each prediction into per-feature
       contributions. Bar charts show mean absolute impact; beeswarm plots show direction
       and distribution of each feature's effect.</p>

    {sections_html}
    {lime_table}
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)
    print(f"[XAI Report] Saved to {output_path}")
```

---

## Phase 2 — Multi-Risk Prediction Engine

**Timeline:** Weeks 3–4  
**Complexity:** Medium  
**Research justification:** Lokanan & Maddhesia (2025) focus exclusively on fraud detection and explicitly call for "multi-risk prediction expanding to fraud + delay + disruption." The DataCo dataset contains `SUSPECTED_FRAUD` and `CANCELED` order statuses that are currently unused in the project. Wang et al. (2024) demonstrate AutoML approaches yielding high precision on multi-domain supply chain risk tasks.

---

### Feature 2.1 — Fraud Detection Module

**New file:** `risk/fraud_detector.py`

**What it does:**
- Trains a CatBoost classifier on `Order Status == SUSPECTED_FRAUD` labels
- Key signals: Benefit per order, Order Item Discount Rate, Customer Segment, Order Region
- Produces fraud probability per order and a ranked list of suspicious orders

**Implementation:**

```python
# risk/fraud_detector.py

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt
import os

class FraudDetector:
    """
    CatBoost-based fraud detection for supply chain orders.
    Targets Order Status == SUSPECTED_FRAUD.

    Key fraud signals from Lokanan & Maddhesia (2025):
    - Unusually high discounts relative to order value
    - Mismatch between customer segment and order size
    - High-risk geographic regions with disproportionate order frequency
    """

    FRAUD_FEATURES = [
        "Order Item Discount Rate",
        "Order Item Product Price",
        "Order Item Profit Ratio",
        "Benefit per order",
        "Order Item Quantity",
        "Sales per customer",
        "Customer Segment",     # categorical
        "Order Region",         # categorical
        "Market",               # categorical
        "Shipping Mode",        # categorical
        "Days for shipping (real)",
        "Days for shipment (scheduled)",
    ]

    CATEGORICAL_FEATURES = ["Customer Segment", "Order Region", "Market", "Shipping Mode"]

    def __init__(self, model_dir: str = "models/", output_dir: str = "plots/fraud/"):
        self.model_dir = model_dir
        self.output_dir = output_dir
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        self.model = None
        self.label_encoders = {}

    def prepare_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Create binary fraud label from Order Status.
        1 = SUSPECTED_FRAUD, 0 = all other statuses.
        """
        return (df["Order Status"] == "SUSPECTED_FRAUD").astype(int)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categoricals and select fraud features."""
        X = df[self.FRAUD_FEATURES].copy()
        for col in self.CATEGORICAL_FEATURES:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
            else:
                X[col] = self.label_encoders[col].transform(X[col].astype(str))
        return X

    def train(self, df: pd.DataFrame):
        """
        Train CatBoost fraud classifier with cross-validation.
        Handles class imbalance automatically via CatBoost's built-in mechanisms.
        """
        y = self.prepare_labels(df)
        X = self.preprocess(df)

        fraud_rate = y.mean()
        print(f"[Fraud] Dataset: {len(df)} orders | Fraud rate: {fraud_rate:.2%}")

        # CatBoost handles categorical features natively if indices are passed
        cat_feature_indices = [X.columns.tolist().index(c)
                                for c in self.CATEGORICAL_FEATURES]

        self.model = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            eval_metric="AUC",
            auto_class_weights="Balanced",   # Handles fraud imbalance
            cat_features=cat_feature_indices,
            random_seed=42,
            verbose=100
        )

        # Stratified cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_aucs = cross_val_score(self.model, X, y, cv=skf, scoring="roc_auc")
        print(f"[Fraud] 5-Fold CV AUC: {cv_aucs.mean():.4f} ± {cv_aucs.std():.4f}")

        self.model.fit(X, y)
        joblib.dump(self.model, f"{self.model_dir}fraud_detector.pkl")
        joblib.dump(self.label_encoders, f"{self.model_dir}fraud_encoders.pkl")
        print(f"[Fraud] Model saved.")

        self._plot_feature_importance(X.columns.tolist())
        return self.model

    def predict_fraud_probability(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score new orders with fraud probability.

        Returns:
            DataFrame with original index + fraud_probability + fraud_flag columns
        """
        X = self.preprocess(df)
        probas = self.model.predict_proba(X)[:, 1]
        return pd.DataFrame({
            "fraud_probability": probas,
            "fraud_flag": (probas >= 0.5).astype(int),
            "fraud_risk_tier": pd.cut(probas,
                bins=[0, 0.3, 0.6, 0.8, 1.0],
                labels=["Low", "Medium", "High", "Critical"])
        }, index=df.index)

    def get_top_suspicious_orders(self, df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
        """Return the top N most suspicious orders for manual review."""
        scores = self.predict_fraud_probability(df)
        result = df[["Order Id", "Order Customer Id", "Order Region",
                      "Sales", "Order Item Discount Rate", "Order Status"]].copy()
        result = result.join(scores)
        return result.nlargest(top_n, "fraud_probability")

    def _plot_feature_importance(self, feature_names: list):
        importances = self.model.get_feature_importance()
        sorted_idx = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.barh([feature_names[i] for i in sorted_idx],
                  [importances[i] for i in sorted_idx], color="#3498db")
        plt.xlabel("Feature Importance")
        plt.title("Fraud Detector — CatBoost Feature Importance")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}fraud_feature_importance.png", dpi=150)
        plt.close()
        print(f"[Fraud] Feature importance plot saved.")
```

---

### Feature 2.2 — Order Cancellation Risk Predictor

**New file:** `risk/cancellation_predictor.py`

**What it does:**
- Predicts probability of CANCELED or ON_HOLD order status
- Uses Product Status (stock availability), order profit margin, and region-level historical cancel rates as key signals
- Enables proactive order flagging before fulfillment begins

**Implementation:**

```python
# risk/cancellation_predictor.py

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

class CancellationPredictor:
    """
    Predicts order cancellation / hold risk before fulfillment.
    Targets: Order Status in [CANCELED, ON_HOLD].

    Key signals:
    - Product Status (0 = available, 1 = not available)
    - Order Item Profit Ratio (low margin orders are more likely cancelled)
    - Regional historical cancellation rate (computed feature)
    - Order Item Discount Rate (high discounts may signal problematic orders)
    """

    FEATURES = [
        "Product Status",
        "Order Item Profit Ratio",
        "Order Item Discount Rate",
        "Order Item Product Price",
        "Order Item Quantity",
        "regional_cancel_rate",   # Engineered feature
        "Shipping Mode",
        "Customer Segment",
        "Market",
        "Days for shipment (scheduled)",
    ]

    def __init__(self, model_dir: str = "models/"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.model = None
        self.regional_cancel_rates = {}

    def engineer_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Engineer regional cancellation rate as a feature.
        If fit=True, computes rates from the data (training).
        If fit=False, applies stored rates (inference).
        """
        df = df.copy()

        if fit:
            cancel_mask = df["Order Status"].isin(["CANCELED", "ON_HOLD"])
            self.regional_cancel_rates = (
                df.groupby("Order Region")["Order Status"]
                .apply(lambda s: s.isin(["CANCELED", "ON_HOLD"]).mean())
                .to_dict()
            )

        df["regional_cancel_rate"] = df["Order Region"].map(
            self.regional_cancel_rates
        ).fillna(df["Order Region"].map(self.regional_cancel_rates).mean())

        # Encode categoricals with simple label encoding
        for col in ["Shipping Mode", "Customer Segment", "Market"]:
            df[col] = df[col].astype("category").cat.codes

        return df

    def prepare_labels(self, df: pd.DataFrame) -> pd.Series:
        return df["Order Status"].isin(["CANCELED", "ON_HOLD"]).astype(int)

    def train(self, df: pd.DataFrame):
        y = self.prepare_labels(df)
        df_eng = self.engineer_features(df, fit=True)
        X = df_eng[self.FEATURES]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        cancel_rate = y_train.mean()
        scale_pos_weight = (1 - cancel_rate) / cancel_rate  # Handle class imbalance

        self.model = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train,
                       eval_set=[(X_test, y_test)],
                       verbose=False)

        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        print("[Cancellation] Test Set Evaluation:")
        print(classification_report(y_test, y_pred,
              target_names=["Active", "Canceled/Hold"]))
        print(f"[Cancellation] ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

        joblib.dump(self.model, f"{self.model_dir}cancellation_predictor.pkl")
        joblib.dump(self.regional_cancel_rates,
                    f"{self.model_dir}regional_cancel_rates.pkl")
        print(f"[Cancellation] Model saved.")
        return self.model

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df_eng = self.engineer_features(df, fit=False)
        X = df_eng[self.FEATURES]
        probas = self.model.predict_proba(X)[:, 1]
        return pd.DataFrame({
            "cancellation_probability": probas,
            "cancellation_flag": (probas >= 0.5).astype(int)
        }, index=df.index)
```

---

### Feature 2.3 — Unified Risk Score per Order

**New file:** `risk/composite_risk_scorer.py`

**What it does:**
- Combines late delivery probability + fraud probability + cancellation probability
- Produces a single composite risk score (0–1) per order
- Routes high-risk orders (score > 0.7) to a flagged output for manual review

**Implementation:**

```python
# risk/composite_risk_scorer.py

import pandas as pd
import numpy as np
import joblib
import os

class CompositeRiskScorer:
    """
    Combines three risk signals into one composite score per order.

    Score formula (weighted combination):
        composite = w_late * P(late) + w_fraud * P(fraud) + w_cancel * P(cancel)

    Default weights reflect business priority: delay > fraud > cancellation
    These should be tuned based on actual cost data (see Phase 5).
    """

    DEFAULT_WEIGHTS = {
        "late_delivery": 0.45,
        "fraud": 0.35,
        "cancellation": 0.20
    }

    def __init__(self, weights: dict = None, model_dir: str = "models/"):
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.model_dir = model_dir

    def score(self,
              df: pd.DataFrame,
              p_late: pd.Series,
              p_fraud: pd.Series,
              p_cancel: pd.Series) -> pd.DataFrame:
        """
        Compute composite risk score.

        Args:
            df: Original order DataFrame
            p_late, p_fraud, p_cancel: Probability Series (same index as df)

        Returns:
            DataFrame with all component probabilities + composite score + risk tier
        """
        composite = (
            self.weights["late_delivery"] * p_late +
            self.weights["fraud"] * p_fraud +
            self.weights["cancellation"] * p_cancel
        )

        risk_tier = pd.cut(
            composite,
            bins=[0, 0.3, 0.5, 0.7, 1.01],
            labels=["Low", "Medium", "High", "Critical"],
            include_lowest=True
        )

        result = pd.DataFrame({
            "Order Id": df["Order Id"] if "Order Id" in df.columns else df.index,
            "Order Region": df.get("Order Region", pd.Series(["Unknown"] * len(df))),
            "Market": df.get("Market", pd.Series(["Unknown"] * len(df))),
            "p_late_delivery": p_late.round(4),
            "p_fraud": p_fraud.round(4),
            "p_cancellation": p_cancel.round(4),
            "composite_risk_score": composite.round(4),
            "risk_tier": risk_tier,
            "requires_review": (composite >= 0.7).astype(int)
        }, index=df.index)

        return result

    def get_flagged_orders(self, risk_df: pd.DataFrame,
                           output_path: str = "flagged_orders.csv") -> pd.DataFrame:
        """Export orders requiring manual review."""
        flagged = risk_df[risk_df["requires_review"] == 1].sort_values(
            "composite_risk_score", ascending=False
        )
        flagged.to_csv(output_path, index=False)
        print(f"[Risk] {len(flagged)} orders flagged for review → {output_path}")
        return flagged

    def print_risk_summary(self, risk_df: pd.DataFrame):
        """Print regional risk distribution summary."""
        summary = risk_df.groupby(["Order Region", "risk_tier"]).size().unstack(fill_value=0)
        print("\n[Risk] Regional Risk Distribution:")
        print(summary.to_string())
```

---

## Phase 3 — Causal What-If Analysis

**Timeline:** Weeks 5–6  
**Complexity:** Medium-High  
**Research justification:** Wyrembek et al. (2025) is the only paper in the reviewed set applying causal ML to supply chains. They explicitly identify lack of "scalable causal ML frameworks integrated with real-time decision systems" as the primary gap. On DataCo, the `Shipping Mode → Late_delivery_risk` causal chain is a natural starting point, since confounders (Market, Order Region, Distance) are all available.

---

### Feature 3.1 — Causal Inference with DoWhy

**New file:** `causal/causal_inference.py`

**What it does:**
- Models the causal effect of Shipping Mode on Late Delivery Risk using DoWhy
- Answers: "Does upgrading to First Class actually reduce delays, or is the correlation explained by confounders like order region?"
- Uses Propensity Score Matching and Linear IV as estimation methods

**Installation:**
```bash
pip install dowhy econml
```

**Implementation:**

```python
# causal/causal_inference.py

import pandas as pd
import numpy as np
import dowhy
from dowhy import CausalModel
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings("ignore")

class SupplyChainCausalAnalyzer:
    """
    Causal inference engine for supply chain decisions.

    Primary causal question:
        Does Shipping Mode causally reduce Late Delivery Risk,
        or is the relationship confounded by Order Region / Market?

    Causal graph (DAG):
        Order Region → Shipping Mode (region determines available modes)
        Order Region → Late_delivery_risk (region affects infrastructure)
        Shipping Mode → Late_delivery_risk (treatment effect of interest)
        Distance → Shipping Mode (longer distances use different modes)
        Distance → Late_delivery_risk (longer routes = more delays)
    """

    def __init__(self, output_dir: str = "plots/causal/"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare binary treatment: 1 = First Class / Same Day, 0 = Standard / Second Class.
        Outcome: Late_delivery_risk (binary).
        """
        df = df.copy()

        # Binary treatment variable
        df["fast_shipping"] = df["Shipping Mode"].isin(
            ["First Class", "Same Day"]
        ).astype(int)

        # Confounders: encode categoricals
        df["region_code"] = df["Order Region"].astype("category").cat.codes
        df["market_code"] = df["Market"].astype("category").cat.codes
        df["segment_code"] = df["Customer Segment"].astype("category").cat.codes

        # Distance proxy (if not already computed by data_preprocessing.py)
        if "shipping_distance_km" not in df.columns:
            df["shipping_distance_km"] = np.sqrt(
                (df["Latitude"] - df["Latitude"].mean())**2 +
                (df["Longitude"] - df["Longitude"].mean())**2
            ) * 111  # rough km conversion

        return df

    def build_causal_model(self, df: pd.DataFrame) -> CausalModel:
        """
        Define the causal DAG and build the DoWhy model.
        """
        # GML-format causal graph
        causal_graph = """
        graph [
            directed 1
            node [ id "fast_shipping" label "fast_shipping" ]
            node [ id "Late_delivery_risk" label "Late_delivery_risk" ]
            node [ id "region_code" label "region_code" ]
            node [ id "market_code" label "market_code" ]
            node [ id "shipping_distance_km" label "shipping_distance_km" ]
            node [ id "segment_code" label "segment_code" ]
            edge [ source "region_code" target "fast_shipping" ]
            edge [ source "region_code" target "Late_delivery_risk" ]
            edge [ source "market_code" target "fast_shipping" ]
            edge [ source "market_code" target "Late_delivery_risk" ]
            edge [ source "shipping_distance_km" target "fast_shipping" ]
            edge [ source "shipping_distance_km" target "Late_delivery_risk" ]
            edge [ source "segment_code" target "fast_shipping" ]
            edge [ source "fast_shipping" target "Late_delivery_risk" ]
        ]
        """

        model = CausalModel(
            data=df,
            treatment="fast_shipping",
            outcome="Late_delivery_risk",
            graph=causal_graph
        )

        print("[Causal] Model built. Displaying causal graph...")
        return model

    def estimate_causal_effect(self, model: CausalModel,
                                method: str = "backdoor.propensity_score_matching"):
        """
        Estimate Average Treatment Effect (ATE) of fast shipping on late delivery risk.

        Args:
            method: Estimation method. Options:
                'backdoor.propensity_score_matching'   — PSM (robust, interpretable)
                'backdoor.linear_regression'           — Linear regression adjustment
                'backdoor.propensity_score_weighting'  — IPW

        Returns:
            Causal estimate object with ATE value
        """
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        print(f"[Causal] Identified estimand:\n{identified_estimand}")

        estimate = model.estimate_effect(
            identified_estimand,
            method_name=method,
            target_units="ate",   # Average Treatment Effect
            confidence_intervals=True,
            test_significance=True
        )

        print(f"\n[Causal] Estimated ATE ({method}):")
        print(f"  Fast shipping reduces late delivery risk by: "
              f"{abs(estimate.value):.4f} (probability units)")
        print(f"  Interpretation: Upgrading from Standard to First Class reduces "
              f"P(late delivery) by ~{abs(estimate.value)*100:.1f} percentage points")

        return estimate

    def refute_estimate(self, model: CausalModel, estimate, identified_estimand):
        """
        Run refutation tests to check robustness of causal estimate.
        A good causal estimate should NOT change much under placebo/random treatment.
        """
        print("\n[Causal] Running refutation tests...")

        # Placebo treatment (should give ~0 effect)
        placebo = model.refute_estimate(
            identified_estimand, estimate,
            method_name="placebo_treatment_refuter",
            placebo_type="permute"
        )
        print(f"  Placebo test — new effect: {placebo.new_effect:.4f} "
              f"(should be near 0 if causal)")

        # Random common cause (robustness to unmeasured confounding)
        random_cc = model.refute_estimate(
            identified_estimand, estimate,
            method_name="random_common_cause"
        )
        print(f"  Random common cause — new effect: {random_cc.new_effect:.4f} "
              f"(should be similar to original)")

        return {"placebo": placebo, "random_common_cause": random_cc}

    def run_full_causal_analysis(self, df: pd.DataFrame) -> dict:
        """Run the complete causal analysis pipeline."""
        df_prepared = self.prepare_data(df)
        model = self.build_causal_model(df_prepared)
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

        estimate_psm = self.estimate_causal_effect(
            model, method="backdoor.propensity_score_matching"
        )
        estimate_lr = self.estimate_causal_effect(
            model, method="backdoor.linear_regression"
        )

        refutation_results = self.refute_estimate(model, estimate_psm, identified_estimand)

        return {
            "psm_ate": estimate_psm.value,
            "lr_ate": estimate_lr.value,
            "refutations": refutation_results,
            "model": model,
            "estimand": identified_estimand
        }
```

---

### Feature 3.2 — Counterfactual Scenario Engine

**New file:** `causal/scenario_engine.py`

**What it does:**
- Interactive what-if analysis: "What if I applied First Class to all orders in Southeast Asia above $200?"
- Estimates impact on delay rate, cost delta, and profit per order
- Produces a scenario comparison table across multiple interventions

**Implementation:**

```python
# causal/scenario_engine.py

import pandas as pd
import numpy as np
from typing import List, Dict, Any
import joblib

class ScenarioEngine:
    """
    Counterfactual scenario engine for supply chain decision support.

    Answers business questions of the form:
        'What if we change [variable X] to [value Y] for orders
         matching [condition Z]?'

    Uses trained ML models to estimate the downstream impact.
    """

    def __init__(self, late_delivery_model=None, model_dir: str = "models/"):
        self.model_dir = model_dir
        if late_delivery_model:
            self.late_model = late_delivery_model
        else:
            self.late_model = joblib.load(f"{model_dir}rf_late_delivery.pkl")

    def define_scenario(self,
                         name: str,
                         filter_condition: str,
                         intervention: Dict[str, Any]) -> Dict:
        """
        Define a what-if scenario.

        Args:
            name: Descriptive scenario name
            filter_condition: Pandas query string to select affected orders
                              e.g., "Market == 'Southeast Asia' and Sales > 200"
            intervention: Dict of column → new value to apply
                          e.g., {"Shipping Mode": "First Class"}

        Returns:
            Scenario definition dict
        """
        return {
            "name": name,
            "filter": filter_condition,
            "intervention": intervention
        }

    def simulate_scenario(self, df: pd.DataFrame,
                           scenario: Dict,
                           feature_columns: List[str]) -> pd.DataFrame:
        """
        Simulate a counterfactual scenario and measure impact.

        Returns:
            Summary DataFrame comparing baseline vs. intervention metrics
        """
        # Select affected orders
        affected = df.query(scenario["filter"]).copy()
        n_affected = len(affected)

        if n_affected == 0:
            print(f"[Scenario] Warning: No orders match filter: {scenario['filter']}")
            return pd.DataFrame()

        print(f"[Scenario] '{scenario['name']}': {n_affected} orders affected")

        # Baseline predictions
        X_baseline = self._prepare_features(affected, feature_columns)
        baseline_late_prob = self.late_model.predict_proba(X_baseline)[:, 1]

        # Apply intervention
        affected_intervention = affected.copy()
        for col, new_val in scenario["intervention"].items():
            affected_intervention[col] = new_val

        # Intervention predictions
        X_intervention = self._prepare_features(affected_intervention, feature_columns)
        intervention_late_prob = self.late_model.predict_proba(X_intervention)[:, 1]

        # Cost / profit impact
        baseline_profit = affected["Order Profit Per Order"].sum()
        # Shipping cost delta (heuristic: First Class = +$5/order, Same Day = +$12/order)
        shipping_cost_delta = self._estimate_shipping_cost_delta(
            affected, scenario["intervention"]
        )

        results = pd.DataFrame({
            "Scenario": scenario["name"],
            "Orders Affected": [n_affected],
            "Baseline Late Rate (%)": [baseline_late_prob.mean() * 100],
            "Intervention Late Rate (%)": [intervention_late_prob.mean() * 100],
            "Late Rate Delta (pp)": [(intervention_late_prob.mean() -
                                       baseline_late_prob.mean()) * 100],
            "Orders Saved from Delay": [
                int((baseline_late_prob - intervention_late_prob).clip(0).sum())
            ],
            "Shipping Cost Delta ($)": [shipping_cost_delta],
            "Baseline Total Profit ($)": [baseline_profit],
            "Net Profit Impact ($)": [baseline_profit - shipping_cost_delta],
        })

        return results

    def compare_scenarios(self, df: pd.DataFrame,
                           scenarios: List[Dict],
                           feature_columns: List[str]) -> pd.DataFrame:
        """Run multiple scenarios and produce a comparison table."""
        all_results = []
        for scenario in scenarios:
            result = self.simulate_scenario(df, scenario, feature_columns)
            if not result.empty:
                all_results.append(result)

        if all_results:
            comparison = pd.concat(all_results, ignore_index=True)
            print("\n[Scenario] Comparison Table:")
            print(comparison.to_string(index=False))
            return comparison
        return pd.DataFrame()

    def _prepare_features(self, df: pd.DataFrame,
                           feature_columns: List[str]) -> np.ndarray:
        """Encode and select features for model prediction."""
        X = df[feature_columns].copy()
        for col in X.select_dtypes(include="object").columns:
            X[col] = X[col].astype("category").cat.codes
        return X.values

    def _estimate_shipping_cost_delta(self, df: pd.DataFrame,
                                       intervention: Dict) -> float:
        """Heuristic shipping cost uplift for mode changes."""
        cost_map = {
            "First Class": 5.0,
            "Same Day": 12.0,
            "Second Class": 2.0,
            "Standard Class": 0.0
        }
        if "Shipping Mode" in intervention:
            new_mode = intervention["Shipping Mode"]
            old_mode_costs = df["Shipping Mode"].map(cost_map).fillna(0)
            new_cost = cost_map.get(new_mode, 0)
            return (new_cost - old_mode_costs).sum()
        return 0.0


# Example usage
def run_example_scenarios(df: pd.DataFrame, feature_columns: List[str]):
    engine = ScenarioEngine()

    scenarios = [
        engine.define_scenario(
            name="Upgrade Southeast Asia high-value orders to First Class",
            filter_condition="Market == 'Pacific Asia' and Sales > 200",
            intervention={"Shipping Mode": "First Class"}
        ),
        engine.define_scenario(
            name="Same Day for all West Africa orders",
            filter_condition="Order Region == 'West Africa'",
            intervention={"Shipping Mode": "Same Day"}
        ),
        engine.define_scenario(
            name="Remove discounts for Corporate segment",
            filter_condition="Customer Segment == 'Corporate'",
            intervention={"Order Item Discount Rate": 0.0}
        ),
    ]

    return engine.compare_scenarios(df, scenarios, feature_columns)
```

---

## Phase 4 — Temporal Deep Learning Upgrade

**Timeline:** Weeks 7–9  
**Complexity:** High  
**Research justification:** Zogaan et al. (2025) demonstrate GRU outperforms ML models on automotive supply chains and CNN achieves 99.3% on pharma — showing that sequence models outperform tabular methods when temporal patterns exist. Your README explicitly identifies LSTM upgrade as a future gap.

---

### Feature 4.1 — LSTM Regional Demand Forecaster

**New file:** `demand_forecasting_lstm.py`

**What it does:**
- Replaces the XGBoost time-series module with an LSTM-based sequence model
- Encodes Order Region and Category Name as learned embeddings
- Captures seasonal patterns (e.g., LATAM Q4 spikes) that XGBoost misses

**Installation:**
```bash
pip install tensorflow>=2.13.0
# OR: pip install torch torchvision  (PyTorch alternative)
```

**Implementation:**

```python
# demand_forecasting_lstm.py

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, LSTM, Dense, Embedding,
                                      Concatenate, Flatten, Dropout)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import joblib
import os

class LSTMDemandForecaster:
    """
    LSTM-based regional demand forecaster.

    Architecture:
    - Time-series branch: LSTM over sequence of past order quantities
    - Embedding branch: Learned embeddings for Order Region + Category Name
    - Combined head: Dense layers → demand prediction

    Improves over XGBoost by:
    - Capturing long-range temporal dependencies (multi-month seasonality)
    - Learning region-specific seasonal patterns through embeddings
    - End-to-end feature learning (no manual feature engineering for time)
    """

    SEQUENCE_LENGTH = 12   # Look back 12 time steps (e.g., 12 weeks)
    FORECAST_HORIZON = 4   # Predict next 4 time steps

    def __init__(self, model_dir: str = "models/"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.scaler = MinMaxScaler()
        self.region_encoder = LabelEncoder()
        self.category_encoder = LabelEncoder()
        self.model = None

    def prepare_sequences(self, df: pd.DataFrame) -> tuple:
        """
        Convert order data to LSTM sequences.

        Creates time series: for each (region, category) pair,
        aggregate weekly order quantity and create sliding window sequences.

        Returns:
            X_seq: (n_samples, seq_len, n_features) — time series features
            X_region: (n_samples,) — region integer codes
            X_category: (n_samples,) — category integer codes
            y: (n_samples, forecast_horizon) — target quantities
        """
        df = df.copy()
        df["order_week"] = pd.to_datetime(df["order date (DateOrders)"]).dt.to_period("W")
        df["region_code"] = self.region_encoder.fit_transform(df["Order Region"])
        df["category_code"] = self.category_encoder.fit_transform(df["Category Name"])

        # Weekly aggregation per (region, category)
        weekly = (
            df.groupby(["order_week", "region_code", "category_code"])
            .agg(
                total_quantity=("Order Item Quantity", "sum"),
                total_sales=("Sales", "sum"),
                avg_discount=("Order Item Discount Rate", "mean"),
                order_count=("Order Id", "count")
            )
            .reset_index()
            .sort_values(["region_code", "category_code", "order_week"])
        )

        # Normalize quantity for LSTM
        weekly["total_quantity_scaled"] = self.scaler.fit_transform(
            weekly[["total_quantity"]]
        )

        X_seq, X_region, X_category, y = [], [], [], []

        for (region_code, cat_code), group in weekly.groupby(
            ["region_code", "category_code"]
        ):
            if len(group) < self.SEQUENCE_LENGTH + self.FORECAST_HORIZON:
                continue  # Skip sparse region-category pairs

            series = group["total_quantity_scaled"].values
            feats = group[["total_quantity_scaled", "avg_discount",
                            "order_count"]].values

            for i in range(len(series) - self.SEQUENCE_LENGTH - self.FORECAST_HORIZON + 1):
                X_seq.append(feats[i: i + self.SEQUENCE_LENGTH])
                X_region.append(region_code)
                X_category.append(cat_code)
                y.append(series[i + self.SEQUENCE_LENGTH:
                                i + self.SEQUENCE_LENGTH + self.FORECAST_HORIZON])

        return (np.array(X_seq), np.array(X_region),
                np.array(X_category), np.array(y))

    def build_model(self, n_regions: int, n_categories: int) -> Model:
        """
        Build LSTM model with region/category embeddings.
        """
        # Time-series input branch
        ts_input = Input(shape=(self.SEQUENCE_LENGTH, 3), name="time_series")
        lstm_out = LSTM(64, return_sequences=True)(ts_input)
        lstm_out = LSTM(32)(lstm_out)

        # Region embedding branch
        region_input = Input(shape=(1,), name="region")
        region_emb = Embedding(input_dim=n_regions, output_dim=8)(region_input)
        region_emb = Flatten()(region_emb)

        # Category embedding branch
        category_input = Input(shape=(1,), name="category")
        category_emb = Embedding(input_dim=n_categories, output_dim=8)(category_input)
        category_emb = Flatten()(category_emb)

        # Combine all branches
        combined = Concatenate()([lstm_out, region_emb, category_emb])
        x = Dense(64, activation="relu")(combined)
        x = Dropout(0.2)(x)
        x = Dense(32, activation="relu")(x)
        output = Dense(self.FORECAST_HORIZON, activation="linear", name="demand")(x)

        model = Model(
            inputs=[ts_input, region_input, category_input],
            outputs=output
        )
        model.compile(optimizer="adam", loss="huber", metrics=["mae"])

        print(f"[LSTM] Model built: {model.count_params():,} parameters")
        model.summary()
        return model

    def train(self, df: pd.DataFrame, epochs: int = 50, batch_size: int = 64):
        """Train the LSTM demand forecaster."""
        X_seq, X_region, X_category, y = self.prepare_sequences(df)

        n_regions = len(self.region_encoder.classes_)
        n_categories = len(self.category_encoder.classes_)

        self.model = self.build_model(n_regions, n_categories)

        split = int(len(X_seq) * 0.8)
        train_inputs = [X_seq[:split], X_region[:split], X_category[:split]]
        val_inputs = [X_seq[split:], X_region[split:], X_category[split:]]

        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss"),
            ModelCheckpoint(
                f"{self.model_dir}lstm_demand_best.h5",
                save_best_only=True, monitor="val_loss"
            )
        ]

        history = self.model.fit(
            train_inputs, y[:split],
            validation_data=(val_inputs, y[split:]),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        # Save encoders and scaler for inference
        joblib.dump(self.region_encoder, f"{self.model_dir}lstm_region_encoder.pkl")
        joblib.dump(self.category_encoder, f"{self.model_dir}lstm_category_encoder.pkl")
        joblib.dump(self.scaler, f"{self.model_dir}lstm_scaler.pkl")

        print(f"[LSTM] Training complete. Best val_loss: "
              f"{min(history.history['val_loss']):.4f}")
        return history

    def forecast(self, region: str, category: str,
                  recent_data: np.ndarray) -> np.ndarray:
        """
        Generate demand forecast for a specific region-category pair.

        Args:
            region: Order Region string
            category: Category Name string
            recent_data: (SEQUENCE_LENGTH, 3) array of recent weekly features

        Returns:
            Forecasted quantities for next FORECAST_HORIZON weeks (original scale)
        """
        region_code = self.region_encoder.transform([region])[0]
        category_code = self.category_encoder.transform([category])[0]

        X_seq = recent_data.reshape(1, self.SEQUENCE_LENGTH, 3)
        X_region = np.array([[region_code]])
        X_category = np.array([[category_code]])

        forecast_scaled = self.model.predict([X_seq, X_region, X_category], verbose=0)
        forecast = self.scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()

        return np.maximum(forecast, 0)  # Demand cannot be negative
```

---

### Feature 4.2 — Hybrid Ensemble (XGBoost + LSTM)

**New file:** `demand_forecasting_hybrid.py`

**What it does:**
- Stacks LSTM (temporal patterns) + XGBoost (tabular features) outputs
- A meta-learner (Ridge regression) combines both — learns when to trust each model
- Mirrors Mitra et al.'s (2025) RF-XGBoost-LR hybrid achieving 91% accuracy

**Implementation:**

```python
# demand_forecasting_hybrid.py

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

class HybridDemandForecaster:
    """
    Hybrid XGBoost + LSTM ensemble for regional demand forecasting.

    Architecture (stacked ensemble):
    Level 0: XGBoost (tabular features) + LSTM (sequence features)
    Level 1: Ridge meta-learner combines Level 0 predictions

    Why this works:
    - XGBoost excels at: categorical features, discount effects, segment patterns
    - LSTM excels at: seasonal trends, week-over-week momentum, lag dependencies
    - Ridge meta-learner learns to weight each model based on validation performance
    """

    def __init__(self, model_dir: str = "models/"):
        self.model_dir = model_dir
        self.xgb_model = None
        self.lstm_forecaster = None
        self.meta_learner = Ridge(alpha=1.0)

    def load_base_models(self):
        """Load existing XGBoost and trained LSTM models."""
        self.xgb_model = joblib.load(f"{self.model_dir}xgb_demand_global.pkl")
        print("[Hybrid] XGBoost model loaded.")
        # LSTM loaded separately through LSTMDemandForecaster

    def get_oof_predictions(self, X_xgb: np.ndarray,
                             X_lstm: list, y: np.ndarray,
                             n_folds: int = 5) -> tuple:
        """
        Generate out-of-fold predictions for both base models.
        These are used to train the meta-learner without leakage.

        Returns:
            (oof_xgb, oof_lstm) — out-of-fold predictions from each model
        """
        from sklearn.model_selection import KFold

        oof_xgb = np.zeros(len(y))
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_xgb)):
            fold_model = joblib.load(f"{self.model_dir}xgb_demand_global.pkl")
            fold_model.fit(X_xgb[train_idx], y[train_idx])
            oof_xgb[val_idx] = fold_model.predict(X_xgb[val_idx])
            print(f"[Hybrid] XGBoost OOF fold {fold+1}/{n_folds} done")

        return oof_xgb

    def train_meta_learner(self, xgb_preds: np.ndarray,
                            lstm_preds: np.ndarray, y: np.ndarray):
        """
        Train Ridge meta-learner on stacked predictions.

        Args:
            xgb_preds: XGBoost OOF predictions
            lstm_preds: LSTM OOF predictions
            y: True demand values
        """
        meta_X = np.column_stack([xgb_preds, lstm_preds])
        self.meta_learner.fit(meta_X, y)

        # Report meta-learner weights (Ridge coefficients = trust weights)
        print(f"[Hybrid] Meta-learner weights:")
        print(f"  XGBoost weight: {self.meta_learner.coef_[0]:.4f}")
        print(f"  LSTM weight:    {self.meta_learner.coef_[1]:.4f}")

        joblib.dump(self.meta_learner, f"{self.model_dir}hybrid_meta_learner.pkl")
        print(f"[Hybrid] Meta-learner saved.")

    def evaluate(self, xgb_preds: np.ndarray, lstm_preds: np.ndarray,
                  y: np.ndarray) -> dict:
        """Compare XGBoost-only, LSTM-only, and hybrid performance."""
        meta_X = np.column_stack([xgb_preds, lstm_preds])
        hybrid_preds = self.meta_learner.predict(meta_X)

        results = {}
        for name, preds in [("XGBoost only", xgb_preds),
                              ("LSTM only", lstm_preds),
                              ("Hybrid ensemble", hybrid_preds)]:
            results[name] = {
                "MAE": mean_absolute_error(y, preds),
                "RMSE": np.sqrt(mean_squared_error(y, preds)),
                "R2": r2_score(y, preds)
            }
            print(f"[Hybrid] {name}: MAE={results[name]['MAE']:.2f}, "
                  f"RMSE={results[name]['RMSE']:.2f}, "
                  f"R²={results[name]['R2']:.4f}")

        return results
```

---

## Phase 5 — Financial Impact & Cost-Sensitive Modeling

**Timeline:** Weeks 10–11  
**Complexity:** Medium  
**Research justification:** Gonçalves & Cortez (2025) demonstrate that optimizing classifier thresholds for misclassification costs (rather than accuracy) improves business decision-making in supply delay risk. Sattar et al. (2025) introduce ESG-augmented metrics but note their dataset (DataCo) lacks universally standardized ESG fields — making proxy-based ESG scoring a contribution.

---

### Feature 5.1 — Misclassification Cost Matrix

**New file:** `financial/cost_sensitive_classifier.py`

**What it does:**
- Assigns real dollar costs to false positives and false negatives using `Benefit per order` and `Order Profit Per Order`
- Optimizes the classification threshold for profit maximization rather than F1/accuracy
- Shows the business cost of the current default (0.5) threshold vs. the optimal threshold

**Implementation:**

```python
# financial/cost_sensitive_classifier.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import joblib
import os


class CostSensitiveOptimizer:
    """
    Optimizes classifier threshold based on business cost of misclassification.

    Cost structure (Late Delivery Classifier):
    - False Negative (missed late delivery): HIGH COST
        → Order delivered late, customer dissatisfied, potential churn
        → Estimated cost = average Benefit per order (revenue at risk)
    - False Positive (flagged as late, was on time): LOW COST
        → Unnecessary expediting cost (e.g., express handling)
        → Estimated cost = fixed expediting overhead

    Per Gonçalves & Cortez (2025): optimizing for cost outperforms
    optimizing for F1 in real supply chain deployment.
    """

    def __init__(self, cost_fn: float = None, cost_fp: float = 5.0,
                 output_dir: str = "plots/financial/"):
        """
        Args:
            cost_fn: Cost of a False Negative (missed late delivery).
                     If None, computed from data as mean Benefit per order.
            cost_fp: Cost of a False Positive (unnecessary expediting). Default $5.
            output_dir: Directory for output plots.
        """
        self.cost_fn = cost_fn
        self.cost_fp = cost_fp
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.optimal_threshold = 0.5

    def compute_costs_from_data(self, df: pd.DataFrame) -> dict:
        """
        Compute realistic FN and FP costs from the DataCo dataset.

        FN cost = mean Benefit per order (revenue lost when late delivery causes return/churn)
        FP cost = fixed expediting cost (conservative $5 default)
        """
        self.cost_fn = df["Benefit per order"].clip(lower=0).mean()
        print(f"[Cost] FN cost (mean benefit at risk): ${self.cost_fn:.2f} per order")
        print(f"[Cost] FP cost (expediting overhead):  ${self.cost_fp:.2f} per order")
        return {"cost_fn": self.cost_fn, "cost_fp": self.cost_fp}

    def compute_total_cost(self, y_true: np.ndarray,
                            y_pred: np.ndarray) -> float:
        """Compute total business cost for a set of predictions."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        total_cost = fn * self.cost_fn + fp * self.cost_fp
        return total_cost

    def find_optimal_threshold(self, y_true: np.ndarray,
                                y_proba: np.ndarray) -> dict:
        """
        Sweep thresholds from 0.1 to 0.9 and find the one minimizing total cost.

        Returns:
            Dict with optimal threshold, cost comparison, and savings vs. default
        """
        thresholds = np.arange(0.1, 0.91, 0.01)
        costs = []

        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            costs.append(self.compute_total_cost(y_true, y_pred))

        costs = np.array(costs)
        opt_idx = np.argmin(costs)
        self.optimal_threshold = thresholds[opt_idx]

        # Default threshold cost
        y_pred_default = (y_proba >= 0.5).astype(int)
        default_cost = self.compute_total_cost(y_true, y_pred_default)

        # Optimal threshold cost
        optimal_cost = costs[opt_idx]
        savings = default_cost - optimal_cost

        print(f"[Cost] Default threshold (0.5) cost:          ${default_cost:,.2f}")
        print(f"[Cost] Optimal threshold ({self.optimal_threshold:.2f}) cost: ${optimal_cost:,.2f}")
        print(f"[Cost] Estimated savings:                      ${savings:,.2f} "
              f"({savings/default_cost*100:.1f}% reduction)")

        # Plot cost vs threshold
        self._plot_cost_curve(thresholds, costs, self.optimal_threshold, default_cost)

        return {
            "optimal_threshold": self.optimal_threshold,
            "optimal_cost": optimal_cost,
            "default_cost": default_cost,
            "savings": savings,
            "savings_pct": savings / default_cost * 100
        }

    def _plot_cost_curve(self, thresholds: np.ndarray, costs: np.ndarray,
                          optimal_t: float, default_cost: float):
        plt.figure(figsize=(10, 5))
        plt.plot(thresholds, costs, linewidth=2, color="#2c3e50", label="Total business cost")
        plt.axvline(x=0.5, linestyle="--", color="#e74c3c",
                     label=f"Default threshold (0.5) — ${default_cost:,.0f}")
        opt_cost = costs[np.argmin(costs)]
        plt.axvline(x=optimal_t, linestyle="--", color="#27ae60",
                     label=f"Optimal threshold ({optimal_t:.2f}) — ${opt_cost:,.0f}")
        plt.scatter([optimal_t], [opt_cost], color="#27ae60", s=100, zorder=5)
        plt.xlabel("Classification Threshold")
        plt.ylabel("Total Business Cost ($)")
        plt.title("Cost-Sensitive Threshold Optimization\n"
                  "(Late Delivery Classifier)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}cost_threshold_optimization.png", dpi=150)
        plt.close()
        print(f"[Cost] Cost curve saved.")
```

---

### Feature 5.2 — Service Level vs. Cost Pareto Optimizer

**New file:** `financial/pareto_optimizer.py`

**What it does:**
- Multi-objective optimization: maximize service level (minimize late delivery) while minimizing operational cost
- Produces Pareto frontier visualization per Market region
- Helps identify which shipping mode configurations dominate across regions

**Implementation:**

```python
# financial/pareto_optimizer.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import os

class ParetoOptimizer:
    """
    Multi-objective Pareto optimization for shipping configuration.

    Objectives (simultaneously optimized):
    1. Minimize: Late delivery rate (service level)
    2. Minimize: Operational cost per order

    Decision variables: Shipping Mode allocation per Market / Order Region

    Reference: Gonçalves & Cortez (2025) extend single-objective delay risk
    models to multi-objective; this implementation applies to DataCo.
    """

    SHIPPING_COST_MAP = {
        "Standard Class": 0,
        "Second Class": 2.5,
        "First Class": 7.5,
        "Same Day": 18.0
    }

    def __init__(self, output_dir: str = "plots/financial/"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def compute_regional_pareto_front(self, df: pd.DataFrame,
                                       group_col: str = "Market") -> pd.DataFrame:
        """
        For each shipping mode in each region, compute:
        - Late delivery rate (objective 1: minimize)
        - Average shipping cost proxy (objective 2: minimize)

        Identifies Pareto-dominant configurations.
        """
        df = df.copy()
        df["shipping_cost_proxy"] = df["Shipping Mode"].map(self.SHIPPING_COST_MAP)

        results = []
        for group_val, group_df in df.groupby(group_col):
            for mode, mode_df in group_df.groupby("Shipping Mode"):
                late_rate = mode_df["Late_delivery_risk"].mean()
                avg_cost = self.SHIPPING_COST_MAP.get(mode, 0)
                avg_profit = mode_df["Order Profit Per Order"].mean()
                n_orders = len(mode_df)

                results.append({
                    group_col: group_val,
                    "Shipping Mode": mode,
                    "Late Rate": late_rate,
                    "Cost Proxy ($)": avg_cost,
                    "Avg Profit ($)": avg_profit,
                    "Order Count": n_orders,
                    "Pareto Dominant": False  # Filled below
                })

        results_df = pd.DataFrame(results)

        # Mark Pareto-dominant configurations per group
        for group_val in results_df[group_col].unique():
            mask = results_df[group_col] == group_val
            subset = results_df[mask].copy()
            results_df.loc[mask, "Pareto Dominant"] = self._mark_pareto(subset)

        return results_df

    def _mark_pareto(self, df: pd.DataFrame) -> pd.Series:
        """
        Mark each row as Pareto dominant (not dominated by any other row).
        Minimizing both Late Rate and Cost Proxy.
        """
        is_pareto = pd.Series(True, index=df.index)
        costs = df[["Late Rate", "Cost Proxy ($)"]].values

        for i in range(len(costs)):
            for j in range(len(costs)):
                if i == j:
                    continue
                if (costs[j] <= costs[i]).all() and (costs[j] < costs[i]).any():
                    is_pareto.iloc[i] = False
                    break

        return is_pareto

    def plot_pareto_frontier(self, pareto_df: pd.DataFrame,
                              group_col: str = "Market"):
        """Plot Pareto frontiers for each market/region."""
        groups = pareto_df[group_col].unique()
        n_cols = 3
        n_rows = int(np.ceil(len(groups) / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols,
                                  figsize=(14, n_rows * 4))
        axes = axes.flatten()

        colors = {"Standard Class": "#95a5a6", "Second Class": "#3498db",
                   "First Class": "#27ae60", "Same Day": "#e74c3c"}

        for ax, group_val in zip(axes, groups):
            subset = pareto_df[pareto_df[group_col] == group_val]

            for _, row in subset.iterrows():
                marker = "★" if row["Pareto Dominant"] else "o"
                ax.scatter(
                    row["Cost Proxy ($)"], row["Late Rate"],
                    c=colors.get(row["Shipping Mode"], "gray"),
                    s=200 if row["Pareto Dominant"] else 80,
                    label=row["Shipping Mode"],
                    zorder=3 if row["Pareto Dominant"] else 2
                )
                ax.annotate(
                    row["Shipping Mode"].split()[0],
                    (row["Cost Proxy ($)"], row["Late Rate"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8
                )

            ax.set_title(f"{group_val}", fontsize=11)
            ax.set_xlabel("Cost Proxy ($/order)")
            ax.set_ylabel("Late Delivery Rate")
            ax.grid(alpha=0.3)

        # Remove unused axes
        for i in range(len(groups), len(axes)):
            axes[i].set_visible(False)

        plt.suptitle("Pareto Frontier: Service Level vs. Cost by Market",
                      fontsize=13, y=1.02)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}pareto_frontier.png",
                     dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Pareto] Frontier plot saved.")
```

---

### Feature 5.3 — ESG-Augmented Profitability Scoring

**New file:** `financial/esg_scorer.py`

**What it does:**
- Layers sustainability proxy metrics onto profit scores using distance as a carbon proxy
- Assigns a composite ESG cost per order (distance × emissions factor × shipping mode weight)
- Produces ESG-adjusted profit rankings per Market region

**Implementation:**

```python
# financial/esg_scorer.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# Regional carbon intensity factors (kg CO2 per km, heuristic proxies)
# Based on regional grid emissions and logistics infrastructure quality
REGIONAL_EMISSIONS_FACTOR = {
    "Western Europe": 0.12,
    "Eastern Europe": 0.18,
    "North America": 0.15,
    "USCA": 0.15,
    "Southeast Asia": 0.22,
    "South Asia": 0.25,
    "West Africa": 0.30,
    "Central Africa": 0.30,
    "East Africa": 0.28,
    "Southern Africa": 0.26,
    "LATAM": 0.20,
    "South America": 0.20,
    "Pacific Asia": 0.20,
    "Eastern Asia": 0.18,
    "Oceania": 0.14,
    "default": 0.20
}

# Shipping mode transport type multiplier
# Air = highest emissions, ground = lowest
SHIPPING_MODE_MULTIPLIER = {
    "Same Day": 3.5,      # Likely air freight
    "First Class": 2.0,   # Mixed air/fast ground
    "Second Class": 1.2,  # Ground priority
    "Standard Class": 1.0 # Standard ground
}


class ESGScorer:
    """
    ESG-augmented profitability scorer for supply chain orders.

    Addresses Sattar et al. (2025) gap: ESG metrics integrated with
    operational data. Uses DataCo's Latitude/Longitude + shipping distance
    as a carbon proxy, combined with regional emissions factors.

    Output: ESG cost per order ($/tonne CO2 equivalent proxy)
            ESG-adjusted profit = Order Profit Per Order - ESG cost
    """

    CARBON_PRICE_PER_KG = 0.05  # $0.05/kg CO2 (conservative social cost estimate)

    def __init__(self, output_dir: str = "plots/financial/"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def compute_esg_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute ESG cost and adjusted profit for each order.

        Requires: shipping_distance_km column (from data_preprocessing.py)
        If not present, computes from Latitude/Longitude.
        """
        df = df.copy()

        # Ensure distance is available
        if "shipping_distance_km" not in df.columns:
            df["shipping_distance_km"] = np.sqrt(
                (df["Latitude"] - df["Latitude"].mean())**2 +
                (df["Longitude"] - df["Longitude"].mean())**2
            ) * 111

        # Get emissions factor per order region / market
        def get_emissions_factor(row):
            for key in [row.get("Market", ""), row.get("Order Region", "")]:
                for k, v in REGIONAL_EMISSIONS_FACTOR.items():
                    if k.lower() in str(key).lower():
                        return v
            return REGIONAL_EMISSIONS_FACTOR["default"]

        df["emissions_factor"] = df.apply(get_emissions_factor, axis=1)
        df["shipping_multiplier"] = df["Shipping Mode"].map(
            SHIPPING_MODE_MULTIPLIER
        ).fillna(1.0)

        # Carbon footprint estimate (kg CO2)
        df["carbon_kg"] = (
            df["shipping_distance_km"] *
            df["emissions_factor"] *
            df["shipping_multiplier"] *
            df["Order Item Quantity"]
        )

        # ESG cost ($)
        df["esg_cost"] = df["carbon_kg"] * self.CARBON_PRICE_PER_KG

        # ESG-adjusted profit
        df["esg_adjusted_profit"] = df["Order Profit Per Order"] - df["esg_cost"]

        # ESG efficiency score: profit per kg carbon
        df["profit_per_carbon_kg"] = (
            df["Order Profit Per Order"] /
            df["carbon_kg"].clip(lower=0.01)
        )

        return df

    def regional_esg_summary(self, df: pd.DataFrame,
                              group_col: str = "Market") -> pd.DataFrame:
        """
        Summarize ESG-adjusted performance by market region.
        """
        df_esg = self.compute_esg_score(df)

        summary = df_esg.groupby(group_col).agg(
            total_profit=("Order Profit Per Order", "sum"),
            total_esg_cost=("esg_cost", "sum"),
            total_esg_adjusted_profit=("esg_adjusted_profit", "sum"),
            avg_carbon_per_order=("carbon_kg", "mean"),
            avg_profit_per_carbon=("profit_per_carbon_kg", "mean"),
            order_count=("Order Id", "count")
        ).reset_index()

        summary["esg_cost_pct"] = (
            summary["total_esg_cost"] / summary["total_profit"] * 100
        ).round(2)

        summary = summary.sort_values("total_esg_adjusted_profit", ascending=False)
        print("[ESG] Regional ESG-Adjusted Profitability Summary:")
        print(summary.to_string(index=False))

        self._plot_esg_summary(summary, group_col)
        return summary

    def _plot_esg_summary(self, summary: pd.DataFrame, group_col: str):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Profit vs. ESG-adjusted profit
        x = range(len(summary))
        ax1.bar([i - 0.2 for i in x], summary["total_profit"],
                 width=0.4, label="Gross Profit", color="#3498db", alpha=0.8)
        ax1.bar([i + 0.2 for i in x], summary["total_esg_adjusted_profit"],
                 width=0.4, label="ESG-Adjusted Profit", color="#27ae60", alpha=0.8)
        ax1.set_xticks(x)
        ax1.set_xticklabels(summary[group_col], rotation=30, ha="right")
        ax1.set_ylabel("Total Profit ($)")
        ax1.set_title("Gross vs. ESG-Adjusted Profit by Market")
        ax1.legend()

        # Carbon efficiency scatter
        ax2.scatter(summary["avg_carbon_per_order"],
                     summary["avg_profit_per_carbon"],
                     s=summary["order_count"] / 10, alpha=0.7, color="#e67e22")
        for _, row in summary.iterrows():
            ax2.annotate(row[group_col],
                          (row["avg_carbon_per_order"], row["avg_profit_per_carbon"]),
                          fontsize=8, textcoords="offset points", xytext=(5, 5))
        ax2.set_xlabel("Avg Carbon per Order (kg CO2)")
        ax2.set_ylabel("Profit per Carbon kg ($/kg)")
        ax2.set_title("Carbon Efficiency by Market\n(bubble size = order count)")

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}esg_regional_summary.png", dpi=150)
        plt.close()
        print(f"[ESG] ESG summary plot saved.")
```

---

## Phase 6 — Dynamic Resilience & Cluster Enhancement

**Timeline:** Weeks 12–13  
**Complexity:** Medium  
**Research justification:** Yang et al. (2025) call for "dynamic/temporal modeling beyond static clusters." Singh et al. (2024) identify resilience as the core framework for supply chain AI/BDA. Your existing clustering (K-Means/DBSCAN) assigns static labels — this phase adds temporal drift detection and resilience scoring.

---

### Feature 6.1 — Cluster-Based Resilience Index

**New file:** `clustering_resilience.py`

**What it does:**
- Scores each geographic cluster on demand volatility, late delivery rate, cancellation rate, and profit variance
- Produces a composite resilience index (0–1, higher = more resilient) per cluster
- Overlays resilience scores on the existing cluster map

**Implementation:**

```python
# clustering_resilience.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

class ResilienceIndexCalculator:
    """
    Computes a composite supply chain resilience index per geographic cluster.

    Resilience dimensions (per Yang et al. 2025; Singh et al. 2024):
    1. Demand stability   — low volatility = high resilience
    2. Delivery reliability — low late delivery rate = high resilience
    3. Order integrity    — low cancellation rate = high resilience
    4. Profit consistency — low profit variance = high resilience

    Index = weighted average of normalized dimension scores (0 = fragile, 1 = resilient)
    """

    WEIGHTS = {
        "demand_stability": 0.30,
        "delivery_reliability": 0.35,
        "order_integrity": 0.20,
        "profit_consistency": 0.15
    }

    def __init__(self, cluster_col: str = "cluster_label",
                 output_dir: str = "plots/"):
        self.cluster_col = cluster_col
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.scaler = MinMaxScaler()

    def compute_cluster_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate operational metrics per cluster.
        Expects df to already have cluster_label from clustering.py.
        """
        cancel_mask = df["Order Status"].isin(["CANCELED", "ON_HOLD"])

        metrics = df.groupby(self.cluster_col).agg(
            # Demand stability: coefficient of variation of order quantity
            demand_cv=("Order Item Quantity",
                        lambda x: x.std() / x.mean() if x.mean() > 0 else 1),
            # Delivery reliability: late delivery rate
            late_rate=("Late_delivery_risk", "mean"),
            # Order integrity: cancellation rate (requires order status)
            order_count=("Order Id", "count"),
            # Profit consistency: profit standard deviation
            profit_std=("Order Profit Per Order", "std"),
            profit_mean=("Order Profit Per Order", "mean"),
            # Additional context
            avg_sales=("Sales", "mean"),
            dominant_region=("Order Region", lambda x: x.mode()[0]),
            dominant_market=("Market", lambda x: x.mode()[0]),
        ).reset_index()

        # Cancellation rate (merge separately)
        cancel_rates = (
            df.groupby(self.cluster_col)
            .apply(lambda g: g["Order Status"].isin(["CANCELED", "ON_HOLD"]).mean())
            .reset_index()
            .rename(columns={0: "cancel_rate"})
        )
        metrics = metrics.merge(cancel_rates, on=self.cluster_col)

        # Profit CoV
        metrics["profit_cv"] = (
            metrics["profit_std"] / metrics["profit_mean"].abs().clip(lower=0.01)
        )

        return metrics

    def compute_resilience_index(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert raw metrics to normalized resilience dimensions and composite index.

        For metrics where LOWER = MORE RESILIENT (late_rate, demand_cv, cancel_rate, profit_cv):
        resilience_dim = 1 - normalized_metric
        """
        df = metrics_df.copy()

        # Normalize each metric to [0, 1]
        for col in ["demand_cv", "late_rate", "cancel_rate", "profit_cv"]:
            col_min = df[col].min()
            col_max = df[col].max()
            if col_max > col_min:
                df[f"{col}_normalized"] = (df[col] - col_min) / (col_max - col_min)
            else:
                df[f"{col}_normalized"] = 0.5

        # Resilience dimensions (inverted: lower raw metric = higher resilience)
        df["demand_stability_score"] = 1 - df["demand_cv_normalized"]
        df["delivery_reliability_score"] = 1 - df["late_rate_normalized"]
        df["order_integrity_score"] = 1 - df["cancel_rate_normalized"]
        df["profit_consistency_score"] = 1 - df["profit_cv_normalized"]

        # Composite weighted index
        df["resilience_index"] = (
            self.WEIGHTS["demand_stability"] * df["demand_stability_score"] +
            self.WEIGHTS["delivery_reliability"] * df["delivery_reliability_score"] +
            self.WEIGHTS["order_integrity"] * df["order_integrity_score"] +
            self.WEIGHTS["profit_consistency"] * df["profit_consistency_score"]
        )

        # Resilience tier
        df["resilience_tier"] = pd.cut(
            df["resilience_index"],
            bins=[0, 0.4, 0.6, 0.8, 1.01],
            labels=["Fragile", "Vulnerable", "Stable", "Resilient"],
            include_lowest=True
        )

        df = df.sort_values("resilience_index", ascending=False)
        print("[Resilience] Cluster Resilience Index:")
        print(df[[self.cluster_col, "dominant_region", "dominant_market",
                   "resilience_index", "resilience_tier"]].to_string(index=False))

        return df

    def plot_resilience_radar(self, resilience_df: pd.DataFrame):
        """Radar chart comparing resilience dimensions per cluster."""
        categories = ["Demand Stability", "Delivery Reliability",
                       "Order Integrity", "Profit Consistency"]
        score_cols = ["demand_stability_score", "delivery_reliability_score",
                       "order_integrity_score", "profit_consistency_score"]

        N = len(categories)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})

        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(resilience_df)))

        for (_, row), color in zip(resilience_df.iterrows(), colors):
            values = row[score_cols].tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=1.5, color=color,
                     label=f"Cluster {int(row[self.cluster_col])} "
                           f"({row['resilience_tier']})")
            ax.fill(angles, values, alpha=0.1, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_title("Cluster Resilience Radar Chart", fontsize=13, pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}resilience_radar.png", dpi=150,
                     bbox_inches="tight")
        plt.close()
        print(f"[Resilience] Radar chart saved.")
```

---

### Feature 6.2 — Temporal Cluster Drift Detection

**New file:** `clustering_drift.py`

**What it does:**
- Tracks cluster membership per Order Region across quarterly windows
- Detects when a region migrates from a low-risk to a high-risk cluster
- Triggers model retraining alerts when drift exceeds a configurable threshold

**Implementation:**

```python
# clustering_drift.py

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import joblib
import os
from datetime import datetime

class ClusterDriftDetector:
    """
    Detects temporal drift in geographic cluster membership.

    For each quarter, re-clusters the data and computes cluster label stability
    per Order Region. A region is flagged as 'drifting' if its cluster assignment
    changes in a way that increases its risk profile.

    Addresses Yang et al. (2025) gap: 'dynamic/temporal modeling needed'
    for supply chain clustering.
    """

    def __init__(self, n_clusters: int = 5, output_dir: str = "plots/"):
        self.n_clusters = n_clusters
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.quarterly_cluster_history = {}

    def assign_quarterly_clusters(self, df: pd.DataFrame,
                                   feature_cols: list) -> pd.DataFrame:
        """
        Cluster data separately for each quarter and track cluster assignments
        per Order Region.

        Returns:
            DataFrame with columns: [quarter, Order Region, cluster_label,
                                     late_rate, demand_volatility]
        """
        df = df.copy()
        df["order_quarter"] = pd.to_datetime(
            df["order date (DateOrders)"]
        ).dt.to_period("Q")

        results = []

        for quarter, q_df in df.groupby("order_quarter"):
            if len(q_df) < self.n_clusters * 10:
                continue  # Skip quarters with too few orders

            # Aggregate to region level for clustering
            region_agg = q_df.groupby("Order Region").agg(
                late_rate=("Late_delivery_risk", "mean"),
                demand_vol=("Order Item Quantity",
                             lambda x: x.std() / x.mean() if x.mean() > 0 else 1),
                avg_sales=("Sales", "mean"),
                order_count=("Order Id", "count")
            ).reset_index()

            if len(region_agg) < self.n_clusters:
                continue

            # Fit KMeans for this quarter
            X = region_agg[["late_rate", "demand_vol", "avg_sales"]].fillna(0).values
            kmeans = KMeans(n_clusters=min(self.n_clusters, len(region_agg)),
                             random_state=42, n_init=10)
            region_agg["cluster_label"] = kmeans.fit_predict(X)

            region_agg["quarter"] = str(quarter)
            results.append(region_agg)

        if not results:
            print("[Drift] Warning: Insufficient quarterly data for drift analysis.")
            return pd.DataFrame()

        history_df = pd.concat(results, ignore_index=True)
        return history_df

    def detect_drift(self, history_df: pd.DataFrame,
                      drift_threshold: float = 0.6) -> pd.DataFrame:
        """
        Flag regions where cluster assignment is unstable across quarters.

        A region 'drifts' if its cluster label changes in more than
        drift_threshold fraction of consecutive quarter pairs.

        Returns:
            DataFrame of drifting regions with drift severity score
        """
        drift_results = []

        for region, region_df in history_df.groupby("Order Region"):
            region_df = region_df.sort_values("quarter")
            labels = region_df["cluster_label"].values
            quarters = region_df["quarter"].values

            if len(labels) < 2:
                continue

            # Count label changes between consecutive quarters
            changes = sum(1 for i in range(1, len(labels)) if labels[i] != labels[i-1])
            drift_rate = changes / (len(labels) - 1)

            # Late rate trend (increasing late rate = worsening)
            late_rates = region_df["late_rate"].values
            late_rate_trend = np.polyfit(range(len(late_rates)), late_rates, 1)[0]

            drift_results.append({
                "Order Region": region,
                "quarters_tracked": len(labels),
                "cluster_changes": changes,
                "drift_rate": round(drift_rate, 3),
                "is_drifting": drift_rate >= drift_threshold,
                "late_rate_trend": round(late_rate_trend, 4),
                "worsening": late_rate_trend > 0.01,
                "latest_cluster": int(labels[-1]),
                "latest_late_rate": round(late_rates[-1], 3)
            })

        drift_df = pd.DataFrame(drift_results)

        drifting = drift_df[drift_df["is_drifting"]]
        worsening = drift_df[drift_df["worsening"]]

        print(f"\n[Drift] Cluster drift analysis complete:")
        print(f"  Regions tracked: {len(drift_df)}")
        print(f"  Drifting regions (cluster unstable): {len(drifting)}")
        print(f"  Worsening regions (late rate trend ↑): {len(worsening)}")

        if len(drifting) > 0:
            print(f"\n[ALERT] Regions requiring model retraining attention:")
            print(drifting[["Order Region", "drift_rate",
                              "late_rate_trend", "latest_late_rate"]].to_string(index=False))

        return drift_df

    def plot_drift_timeline(self, history_df: pd.DataFrame,
                             regions_to_plot: list = None):
        """Plot cluster assignment timeline for drifting regions."""
        if regions_to_plot is None:
            regions_to_plot = history_df["Order Region"].unique()[:6]

        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes = axes.flatten()

        for ax, region in zip(axes, regions_to_plot):
            region_df = history_df[
                history_df["Order Region"] == region
            ].sort_values("quarter")

            ax.plot(region_df["quarter"], region_df["cluster_label"],
                     "o-", linewidth=2, markersize=8, color="#2c3e50")
            ax2 = ax.twinx()
            ax2.plot(region_df["quarter"], region_df["late_rate"],
                      "s--", linewidth=1.5, markersize=6, color="#e74c3c", alpha=0.7)
            ax2.set_ylabel("Late Rate", color="#e74c3c", fontsize=9)

            ax.set_title(region, fontsize=10)
            ax.set_ylabel("Cluster Label")
            ax.tick_params(axis="x", rotation=45)

        for i in range(len(regions_to_plot), len(axes)):
            axes[i].set_visible(False)

        plt.suptitle("Cluster Drift Timeline by Region\n"
                      "(blue = cluster label, red dashed = late rate)", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}cluster_drift_timeline.png",
                     dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Drift] Drift timeline saved.")
```

---

## Updated Project Structure

```text
├── index.py                          # Main orchestrator (extend to call new phases)
│
├── data_preprocessing.py             # Existing — add ESG distance features
├── geographic_analysis.py            # Existing
├── clustering.py                     # Existing — add cluster_label output for resilience
├── clustering_resilience.py          # NEW Phase 6.1 — Resilience Index
├── clustering_drift.py               # NEW Phase 6.2 — Drift Detection
│
├── demand_forecasting.py             # Existing XGBoost forecaster
├── demand_forecasting_lstm.py        # NEW Phase 4.1 — LSTM Forecaster
├── demand_forecasting_hybrid.py      # NEW Phase 4.2 — Hybrid Ensemble
│
├── late_delivery_classifier.py       # Existing
│
├── explainability/
│   ├── shap_explainer.py             # NEW Phase 1.1 — SHAP
│   ├── lime_explainer.py             # NEW Phase 1.2 — LIME
│   └── generate_xai_report.py        # NEW Phase 1.3 — XAI HTML Report
│
├── risk/
│   ├── fraud_detector.py             # NEW Phase 2.1 — Fraud Detection
│   ├── cancellation_predictor.py     # NEW Phase 2.2 — Cancellation Risk
│   └── composite_risk_scorer.py      # NEW Phase 2.3 — Unified Risk Score
│
├── causal/
│   ├── causal_inference.py           # NEW Phase 3.1 — DoWhy Causal Analysis
│   └── scenario_engine.py            # NEW Phase 3.2 — What-If Scenarios
│
├── financial/
│   ├── cost_sensitive_classifier.py  # NEW Phase 5.1 — Cost Threshold Optimization
│   ├── pareto_optimizer.py           # NEW Phase 5.2 — Multi-objective Pareto
│   └── esg_scorer.py                 # NEW Phase 5.3 — ESG Profitability Scoring
│
├── generate_report.py                # Existing — extend with XAI + financial sections
├── model_pipeline.py                 # Existing
├── model_verification.py             # Existing
│
├── models/                           # Existing + new model files
│   ├── fraud_detector.pkl
│   ├── cancellation_predictor.pkl
│   ├── lstm_demand_best.h5
│   ├── hybrid_meta_learner.pkl
│   └── ...
│
└── plots/
    ├── shap/                         # Phase 1 SHAP outputs
    ├── lime/                         # Phase 1 LIME outputs
    ├── fraud/                        # Phase 2 fraud plots
    ├── causal/                       # Phase 3 causal graphs
    ├── financial/                    # Phase 5 cost/ESG plots
    └── ...                           # Existing plots
```

---

## Dependencies Reference

```text
# Existing (from README)
scikit-learn>=1.3.0
xgboost>=2.0.0
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.11.0
plotly>=5.15.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Phase 1 — Explainability
shap>=0.44.0
lime>=0.2.0.1

# Phase 2 — Multi-Risk
catboost>=1.2.0

# Phase 3 — Causal Inference
dowhy>=0.11.0
econml>=0.15.0

# Phase 4 — Deep Learning
tensorflow>=2.13.0
# OR: torch>=2.0.0 torchvision>=0.15.0

# Phase 5 — Financial (no new deps — uses sklearn + matplotlib)

# Phase 6 — Drift Detection (no new deps — uses sklearn)
```

Install all at once:
```bash
pip install shap lime catboost dowhy econml tensorflow
```

---

*Implementation plan grounded in: Sattar et al. (2025), Zogaan et al. (2025), Lokanan & Maddhesia (2025), Wyrembek et al. (2025), Yang et al. (2025), Wang et al. (2024), Singh et al. (2024), Gonçalves & Cortez (2025), Camur et al. (2024), Mitra et al. (2025)*
