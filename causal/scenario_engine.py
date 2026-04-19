import pandas as pd
import numpy as np
from typing import List, Dict, Any
import joblib

class ScenarioEngine:
    """
    Counterfactual scenario engine for supply chain decision support.

    Answers business questions of the form:
        'What if we change [variable X] to [value Y] for orders
         matching [condition Z]?'

    Uses trained ML models to estimate the downstream impact.
    """

    def __init__(self, late_delivery_model=None, model_dir: str = "models/"):
        self.model_dir = model_dir
        if late_delivery_model:
            self.late_model = late_delivery_model
        else:
            self.late_model = joblib.load(f"{model_dir}rf_late_delivery.pkl")

    def define_scenario(self,
                         name: str,
                         filter_condition: str,
                         intervention: Dict[str, Any]) -> Dict:
        """
        Define a what-if scenario.

        Args:
            name: Descriptive scenario name
            filter_condition: Pandas query string to select affected orders
                              e.g., "Market == 'Southeast Asia' and Sales > 200"
            intervention: Dict of column → new value to apply
                          e.g., {"Shipping Mode": "First Class"}

        Returns:
            Scenario definition dict
        """
        return {
            "name": name,
            "filter": filter_condition,
            "intervention": intervention
        }

    def simulate_scenario(self, df: pd.DataFrame,
                           scenario: Dict,
                           feature_columns: List[str]) -> pd.DataFrame:
        """
        Simulate a counterfactual scenario and measure impact.

        Returns:
            Summary DataFrame comparing baseline vs. intervention metrics
        """
        # Select affected orders
        affected = df.query(scenario["filter"]).copy()
        n_affected = len(affected)

        if n_affected == 0:
            print(f"[Scenario] Warning: No orders match filter: {scenario['filter']}")
            return pd.DataFrame()

        print(f"[Scenario] '{scenario['name']}': {n_affected} orders affected")

        # Baseline predictions
        X_baseline = self._prepare_features(affected, feature_columns)
        baseline_late_prob = self.late_model.predict_proba(X_baseline)[:, 1]

        # Apply intervention
        affected_intervention = affected.copy()
        for col, new_val in scenario["intervention"].items():
            affected_intervention[col] = new_val

        # Intervention predictions
        X_intervention = self._prepare_features(affected_intervention, feature_columns)
        intervention_late_prob = self.late_model.predict_proba(X_intervention)[:, 1]

        # Cost / profit impact
        baseline_profit = affected["Order Profit Per Order"].sum()
        # Shipping cost delta (heuristic: First Class = +$5/order, Same Day = +$12/order)
        shipping_cost_delta = self._estimate_shipping_cost_delta(
            affected, scenario["intervention"]
        )

        results = pd.DataFrame({
            "Scenario": scenario["name"],
            "Orders Affected": [n_affected],
            "Baseline Late Rate (%)": [baseline_late_prob.mean() * 100],
            "Intervention Late Rate (%)": [intervention_late_prob.mean() * 100],
            "Late Rate Delta (pp)": [(intervention_late_prob.mean() -
                                       baseline_late_prob.mean()) * 100],
            "Orders Saved from Delay": [
                int((baseline_late_prob - intervention_late_prob).clip(0).sum())
            ],
            "Shipping Cost Delta ($)": [shipping_cost_delta],
            "Baseline Total Profit ($)": [baseline_profit],
            "Net Profit Impact ($)": [baseline_profit - shipping_cost_delta],
        })

        return results

    def compare_scenarios(self, df: pd.DataFrame,
                           scenarios: List[Dict],
                           feature_columns: List[str]) -> pd.DataFrame:
        """Run multiple scenarios and produce a comparison table."""
        all_results = []
        for scenario in scenarios:
            result = self.simulate_scenario(df, scenario, feature_columns)
            if not result.empty:
                all_results.append(result)

        if all_results:
            comparison = pd.concat(all_results, ignore_index=True)
            print("\n[Scenario] Comparison Table:")
            print(comparison.to_string(index=False))
            return comparison
        return pd.DataFrame()

    def _prepare_features(self, df: pd.DataFrame,
                           feature_columns: List[str]) -> np.ndarray:
        """Encode and select features for model prediction."""
        X = df[feature_columns].copy()
        for col in X.select_dtypes(include="object").columns:
            X[col] = X[col].astype("category").cat.codes
        return X.values

    def _estimate_shipping_cost_delta(self, df: pd.DataFrame,
                                       intervention: Dict) -> float:
        """Heuristic shipping cost uplift for mode changes."""
        cost_map = {
            "First Class": 5.0,
            "Same Day": 12.0,
            "Second Class": 2.0,
            "Standard Class": 0.0
        }
        if "Shipping Mode" in intervention:
            new_mode = intervention["Shipping Mode"]
            old_mode_costs = df["Shipping Mode"].map(cost_map).fillna(0)
            new_cost = cost_map.get(new_mode, 0)
            return (new_cost - old_mode_costs).sum()
        return 0.0


# Example usage
def run_example_scenarios(df: pd.DataFrame, feature_columns: List[str]):
    engine = ScenarioEngine()

    scenarios = [
        engine.define_scenario(
            name="Upgrade Southeast Asia high-value orders to First Class",
            filter_condition="Market == 'Pacific Asia' and Sales > 200",
            intervention={"Shipping Mode": "First Class"}
        ),
        engine.define_scenario(
            name="Same Day for all West Africa orders",
            filter_condition="Order Region == 'West Africa'",
            intervention={"Shipping Mode": "Same Day"}
        ),
        engine.define_scenario(
            name="Remove discounts for Corporate segment",
            filter_condition="Customer Segment == 'Corporate'",
            intervention={"Order Item Discount Rate": 0.0}
        ),
    ]

    return engine.compare_scenarios(df, scenarios, feature_columns)
