import pandas as pd
import numpy as np
import joblib
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

class CompositeRiskScorer:
    """
    Combines three risk signals into one composite score per order.

    Score formula (weighted combination):
        composite = w_late * P(late) + w_fraud * P(fraud) + w_cancel * P(cancel)

    Default weights reflect business priority: delay > fraud > cancellation
    These should be tuned based on actual cost data (see Phase 5).
    """

    DEFAULT_WEIGHTS = {
        "late_delivery": 0.45,
        "fraud": 0.35,
        "cancellation": 0.20
    }

    def __init__(self, weights: dict = None, model_dir: str = "models/"):
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.model_dir = model_dir

    def score(self,
              df: pd.DataFrame,
              p_late: pd.Series,
              p_fraud: pd.Series,
              p_cancel: pd.Series) -> pd.DataFrame:
        """
        Compute composite risk score.

        Args:
            df: Original order DataFrame
            p_late, p_fraud, p_cancel: Probability Series (same index as df)

        Returns:
            DataFrame with all component probabilities + composite score + risk tier
        """
        composite = (
            self.weights["late_delivery"] * p_late +
            self.weights["fraud"] * p_fraud +
            self.weights["cancellation"] * p_cancel
        )

        risk_tier = pd.cut(
            composite,
            bins=[0, 0.3, 0.5, 0.7, 1.01],
            labels=["Low", "Medium", "High", "Critical"],
            include_lowest=True
        )

        result = pd.DataFrame({
            "Order Id": df["Order Id"] if "Order Id" in df.columns else df.index,
            "Order Region": df.get("Order Region", pd.Series(["Unknown"] * len(df))),
            "Market": df.get("Market", pd.Series(["Unknown"] * len(df))),
            "p_late_delivery": p_late.round(4),
            "p_fraud": p_fraud.round(4),
            "p_cancellation": p_cancel.round(4),
            "composite_risk_score": composite.round(4),
            "risk_tier": risk_tier,
            "requires_review": (composite >= 0.7).astype(int)
        }, index=df.index)

        return result

    def get_flagged_orders(self, risk_df: pd.DataFrame,
                           output_path: str = "flagged_orders.csv") -> pd.DataFrame:
        """Export orders requiring manual review."""
        flagged = risk_df[risk_df["requires_review"] == 1].sort_values(
            "composite_risk_score", ascending=False
        )
        flagged.to_csv(output_path, index=False)
        print(f"[Risk] {len(flagged)} orders flagged for review → {output_path}")
        return flagged

    def print_risk_summary(self, risk_df: pd.DataFrame):
        """Print regional risk distribution summary and generate plot."""
        summary = risk_df.groupby(["Order Region", "risk_tier"]).size().unstack(fill_value=0)
        print("\n[Risk] Regional Risk Distribution:")
        print(summary.to_string())
        
        self.plot_distribution(risk_df)

    def plot_distribution(self, risk_df: pd.DataFrame):
        """Generates Fig 10: Histogram/KDE of 0-100 Composite Risk Index."""
        out_dir = "plots/risk"
        os.makedirs(out_dir, exist_ok=True)
        
        # Scale 0-1 probability to 0-100 index for plotting
        scores_100 = risk_df['composite_risk_score'] * 100
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create KDE / Histogram plot
        sns.histplot(scores_100, bins=40, kde=True, color='steelblue', edgecolor='white', ax=ax)
        
        # Mark threshold (70 on a 100 scale)
        threshold = 70
        ax.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Intervention Threshold ({threshold})')
        
        # Shading regions
        ax.axvspan(0, threshold, color='lightgreen', alpha=0.15, label='Safe Zone (<70)')
        ax.axvspan(threshold, 100, color='salmon', alpha=0.15, label='High-Risk Zone (>=70)')
        
        # Formatting
        ax.set_title('Composite Risk Score Distribution', fontsize=15, fontweight='bold')
        ax.set_xlabel('Composite Risk Index (0-100)', fontsize=12)
        ax.set_ylabel('Number of Orders', fontsize=12)
        ax.set_xlim([0, 100])
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        fname = f"{out_dir}/composite_risk_distribution.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"[Risk] Saved Composite Risk plot: {fname}")
