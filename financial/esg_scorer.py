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
