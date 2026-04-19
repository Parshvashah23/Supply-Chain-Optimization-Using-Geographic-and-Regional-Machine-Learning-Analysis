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
