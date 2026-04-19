import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import joblib
import os

class CostSensitiveOptimizer:
    """
    Optimizes classifier threshold based on business cost of misclassification.

    Cost structure (Late Delivery Classifier):
    - False Negative (missed late delivery): HIGH COST
        → Order delivered late, customer dissatisfied, potential churn
        → Estimated cost = average Benefit per order (revenue at risk)
    - False Positive (flagged as late, was on time): LOW COST
        → Unnecessary expediting cost (e.g., express handling)
        → Estimated cost = fixed expediting overhead

    Per Gonçalves & Cortez (2025): optimizing for cost outperforms
    optimizing for F1 in real supply chain deployment.
    """

    def __init__(self, cost_fn: float = None, cost_fp: float = 5.0,
                 output_dir: str = "plots/financial/"):
        """
        Args:
            cost_fn: Cost of a False Negative (missed late delivery).
                     If None, computed from data as mean Benefit per order.
            cost_fp: Cost of a False Positive (unnecessary expediting). Default $5.
            output_dir: Directory for output plots.
        """
        self.cost_fn = cost_fn
        self.cost_fp = cost_fp
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.optimal_threshold = 0.5

    def compute_costs_from_data(self, df: pd.DataFrame) -> dict:
        """
        Compute realistic FN and FP costs from the DataCo dataset.

        FN cost = mean Benefit per order (revenue lost when late delivery causes return/churn)
        FP cost = fixed expediting cost (conservative $5 default)
        """
        self.cost_fn = df["Benefit per order"].clip(lower=0).mean()
        print(f"[Cost] FN cost (mean benefit at risk): ${self.cost_fn:.2f} per order")
        print(f"[Cost] FP cost (expediting overhead):  ${self.cost_fp:.2f} per order")
        return {"cost_fn": self.cost_fn, "cost_fp": self.cost_fp}

    def compute_total_cost(self, y_true: np.ndarray,
                            y_pred: np.ndarray) -> float:
        """Compute total business cost for a set of predictions."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        total_cost = fn * self.cost_fn + fp * self.cost_fp
        return total_cost

    def find_optimal_threshold(self, y_true: np.ndarray,
                                y_proba: np.ndarray) -> dict:
        """
        Sweep thresholds from 0.1 to 0.9 and find the one minimizing total cost.

        Returns:
            Dict with optimal threshold, cost comparison, and savings vs. default
        """
        thresholds = np.arange(0.1, 0.91, 0.01)
        costs = []

        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            costs.append(self.compute_total_cost(y_true, y_pred))

        costs = np.array(costs)
        opt_idx = np.argmin(costs)
        self.optimal_threshold = thresholds[opt_idx]

        # Default threshold cost
        y_pred_default = (y_proba >= 0.5).astype(int)
        default_cost = self.compute_total_cost(y_true, y_pred_default)

        # Optimal threshold cost
        optimal_cost = costs[opt_idx]
        savings = default_cost - optimal_cost

        print(f"[Cost] Default threshold (0.5) cost:          ${default_cost:,.2f}")
        print(f"[Cost] Optimal threshold ({self.optimal_threshold:.2f}) cost: ${optimal_cost:,.2f}")
        print(f"[Cost] Estimated savings:                      ${savings:,.2f} "
              f"({savings/default_cost*100:.1f}% reduction)")

        # Plot cost vs threshold
        self._plot_cost_curve(thresholds, costs, self.optimal_threshold, default_cost)

        return {
            "optimal_threshold": self.optimal_threshold,
            "optimal_cost": optimal_cost,
            "default_cost": default_cost,
            "savings": savings,
            "savings_pct": savings / default_cost * 100
        }

    def _plot_cost_curve(self, thresholds: np.ndarray, costs: np.ndarray,
                          optimal_t: float, default_cost: float):
        plt.figure(figsize=(10, 5))
        plt.plot(thresholds, costs, linewidth=2, color="#2c3e50", label="Total business cost")
        plt.axvline(x=0.5, linestyle="--", color="#e74c3c",
                     label=f"Default threshold (0.5) — ${default_cost:,.0f}")
        opt_cost = costs[np.argmin(costs)]
        plt.axvline(x=optimal_t, linestyle="--", color="#27ae60",
                     label=f"Optimal threshold ({optimal_t:.2f}) — ${opt_cost:,.0f}")
        plt.scatter([optimal_t], [opt_cost], color="#27ae60", s=100, zorder=5)
        plt.xlabel("Classification Threshold")
        plt.ylabel("Total Business Cost ($)")
        plt.title("Cost-Sensitive Threshold Optimization\n"
                  "(Late Delivery Classifier)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}cost_threshold_optimization.png", dpi=150)
        plt.close()
        print(f"[Cost] Cost curve saved.")
