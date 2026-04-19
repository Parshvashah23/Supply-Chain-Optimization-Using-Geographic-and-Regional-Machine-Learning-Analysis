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
