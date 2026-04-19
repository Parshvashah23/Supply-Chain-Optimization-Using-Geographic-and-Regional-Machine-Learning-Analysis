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
