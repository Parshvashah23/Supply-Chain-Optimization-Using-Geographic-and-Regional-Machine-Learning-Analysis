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
                f"{self.model_dir}lstm_demand_best.keras",
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
