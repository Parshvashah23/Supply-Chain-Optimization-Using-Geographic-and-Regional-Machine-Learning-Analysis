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
