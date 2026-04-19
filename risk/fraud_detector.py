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
            random_seed=42,
            verbose=100
        )

        # Stratified cross-validation
        # Note: In sklearn 1.4+, fit_params was replaced by 'params' in cross_val_score
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_aucs = cross_val_score(self.model, X, y, cv=skf, scoring="roc_auc", 
                                 params={'cat_features': cat_feature_indices})
        print(f"[Fraud] 5-Fold CV AUC: {cv_aucs.mean():.4f} ± {cv_aucs.std():.4f}")

        self.model.fit(X, y, cat_features=cat_feature_indices)
        joblib.dump(self.model, os.path.join(self.model_dir, "fraud_detector.pkl"))
        joblib.dump(self.label_encoders, os.path.join(self.model_dir, "fraud_encoders.pkl"))
        print(f"[Fraud] Model saved.")

        self._plot_feature_importance(X.columns.tolist())
        self.plot_diagnostics(X, y)
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
        # Use feature_importances_ as the primary attribute for sklearn compatibility
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "get_feature_importance"):
            importances = self.model.get_feature_importance()
        else:
            print("[Fraud] Warning: Could not extract feature importance.")
            return

        sorted_idx = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.barh([feature_names[i] for i in sorted_idx],
                  [importances[i] for i in sorted_idx], color="#3498db")
        plt.xlabel("Feature Importance")
        plt.title("Fraud Detector — CatBoost Feature Importance")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "fraud_feature_importance.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[Fraud] Feature importance plot saved to {save_path}")

    def plot_diagnostics(self, X, y):
        """Generate ROC, PR curves and probability distribution plots."""
        from sklearn.metrics import roc_curve, precision_recall_curve, auc
        
        probas = self.model.predict_proba(X)[:, 1]
        
        # 1. ROC Curve
        fpr, tpr, _ = roc_curve(y, probas)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Fraud Detector — ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "fraud_roc_curve.png"), dpi=150)
        plt.close()

        # 2. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y, probas)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Fraud Detector — Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "fraud_pr_curve.png"), dpi=150)
        plt.close()

        # 3. Probability Distribution
        plt.figure(figsize=(8, 6))
        plt.hist(probas[y == 0], bins=50, alpha=0.5, label='Normal', color='blue', density=True)
        plt.hist(probas[y == 1], bins=50, alpha=0.5, label='Fraud', color='red', density=True)
        plt.xlabel('Fraud Probability')
        plt.ylabel('Density')
        plt.title('Fraud Detector — Score Distribution')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "fraud_score_distribution.png"), dpi=150)
        plt.close()
        
        print(f"[Fraud] Diagnostic plots (ROC, PR, Dist) saved to {self.output_dir}")
