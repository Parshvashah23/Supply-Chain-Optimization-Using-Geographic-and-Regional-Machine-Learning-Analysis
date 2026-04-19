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
