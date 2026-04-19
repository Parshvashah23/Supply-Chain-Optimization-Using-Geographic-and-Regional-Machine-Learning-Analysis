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

    def load_models(self, xgb_demand=None, rf_classifier=None):
        """Load existing trained models from the pipeline or use provided ones."""
        if xgb_demand is not None:
            self.xgb_demand = xgb_demand
        else:
            self.xgb_demand = joblib.load(f"{self.model_dir}regional_forecast/global_xgboost_forecast.pkl")
            
        if rf_classifier is not None:
            self.rf_classifier = rf_classifier
        else:
            self.rf_classifier = joblib.load(f"{self.model_dir}regional_forecast/global_rf_classifier.pkl")
            
        print("[SHAP] Models ready for explanation.")

    def compute_shap_demand(self, X_train: pd.DataFrame, X_sample: pd.DataFrame = None):
        """
        Compute SHAP values for the XGBoost demand forecaster.
        """
        if X_sample is None:
            X_sample = X_train.sample(min(50, len(X_train)), random_state=42)

        # Robust Feature Alignment: ensure we have the exact columns the model expects
        if hasattr(self.xgb_demand, "feature_names_in_"):
            expected_cols = list(self.xgb_demand.feature_names_in_)
            X_sample = X_sample.reindex(columns=expected_cols, fill_value=0)
        
        X_sample_np = X_sample.values.astype(np.float32)
        print(f"[SHAP] Computing KernelExplainer for demand model (sample size: {len(X_sample)})...")
        
        try:
            # We use a lambda to hide the model object from SHAP's introspection
            # This fixes the 'property feature_names_in_ of XGBRegressor object has no setter' error
            model_func = lambda x: self.xgb_demand.predict(x)
            background = shap.sample(X_sample_np, min(10, len(X_sample_np)))
            explainer = shap.KernelExplainer(model_func, background)
            print("[SHAP] Explainer initialized. Computing SHAP values...")
            shap_values = explainer.shap_values(X_sample_np)
            
            # Handle list vs array vs 3D array for Regressor
            if isinstance(shap_values, list) and len(shap_values) > 0:
                shap_values = shap_values[0]
            
            print("[SHAP] SHAP values computed successfully.")
        except Exception as e:
            print(f"[SHAP] SHAP computation failed: {e}")
            return None, None, None

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
        """
        if X_sample is None:
            X_sample = X_train.sample(min(50, len(X_train)), random_state=42)

        # Robust Feature Alignment: ensure we have the exact columns the model expects
        if hasattr(self.rf_classifier, "feature_names_in_"):
            expected_cols = list(self.rf_classifier.feature_names_in_)
            X_sample = X_sample.reindex(columns=expected_cols, fill_value=0)

        X_sample_np = X_sample.values.astype(np.float32)
        print(f"[SHAP] Computing KernelExplainer for classifier model (sample size: {len(X_sample)})...")
        try:
            # Use lambda to wrap predict_proba. KernelExplainer treats this as a generic function.
            model_func = lambda x: self.rf_classifier.predict_proba(x)
            background = shap.sample(X_sample_np, min(10, len(X_sample_np)))
            explainer = shap.KernelExplainer(model_func, background)
            print("[SHAP] Explainer initialized. Computing SHAP values...")
            shap_values = explainer.shap_values(X_sample_np)
            
            # Handle shape variants from KernelExplainer for classification
            # If shape is (samples, features, classes), class 1 is the goal (Late/Fraud)
            if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
                sv_class1 = shap_values[:, :, 1]
            elif isinstance(shap_values, list) and len(shap_values) > 1:
                sv_class1 = shap_values[1]
            else:
                sv_class1 = shap_values

            print("[SHAP] SHAP values computed successfully.")
            
            # Global summary plot
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

        except Exception as e:
            print(f"[SHAP] SHAP computation failed: {e}")
            return None, None, None

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
                               region_col: str = "Order Region", 
                               region_labels: pd.Series = None):
        """
        Aggregate mean |SHAP| values per Order Region and plot a heatmap.
        """
        import seaborn as sns

        if (shap_values is None) or (len(shap_values) == 0):
            print("[SHAP] No values to plot for regional heatmap.")
            return

        shap_df = pd.DataFrame(shap_values, columns=X.columns, index=X.index)
        
        # If we have external labels (e.g. from before dummies), use them
        if region_labels is not None:
            shap_df[region_col] = region_labels.values
        elif region_col in X.columns:
            shap_df[region_col] = X[region_col].values
        else:
            print(f"[SHAP] Warning: {region_col} not found in features or labels. Heatmap aborted.")
            return

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
