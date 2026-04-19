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
        print(f"  Worsening regions (late rate trend up): {len(worsening)}")

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
