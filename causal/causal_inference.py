import pandas as pd
import numpy as np
import dowhy
from dowhy import CausalModel
import matplotlib.pyplot as plt
import networkx as nx
import warnings
import os

warnings.filterwarnings("ignore")

class SupplyChainCausalAnalyzer:
    """
    Causal inference engine for supply chain decisions.

    Primary causal question:
        Does Shipping Mode causally reduce Late Delivery Risk,
        or is the relationship confounded by Order Region / Market?

    Causal graph (DAG):
        Order Region → Shipping Mode (region determines available modes)
        Order Region → Late_delivery_risk (region affects infrastructure)
        Shipping Mode → Late_delivery_risk (treatment effect of interest)
        Distance → Shipping Mode (longer distances use different modes)
        Distance → Late_delivery_risk (longer routes = more delays)
    """

    def __init__(self, output_dir: str = "plots/causal/"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare binary treatment: 1 = First Class / Same Day, 0 = Standard / Second Class.
        Outcome: Late_delivery_risk (binary).
        """
        df = df.copy()

        # Binary treatment variable
        df["fast_shipping"] = df["Shipping Mode"].isin(
            ["First Class", "Same Day"]
        ).astype(int)

        # Confounders: encode categoricals
        df["region_code"] = df["Order Region"].astype("category").cat.codes
        df["market_code"] = df["Market"].astype("category").cat.codes
        df["segment_code"] = df["Customer Segment"].astype("category").cat.codes

        # Distance proxy (if not already computed by data_preprocessing.py)
        if "shipping_distance_km" not in df.columns:
            df["shipping_distance_km"] = np.sqrt(
                (df["Latitude"] - df["Latitude"].mean())**2 +
                (df["Longitude"] - df["Longitude"].mean())**2
            ) * 111  # rough km conversion

        return df

    def build_causal_model(self, df: pd.DataFrame) -> CausalModel:
        """
        Define the causal DAG and build the DoWhy model.
        """
        # GML-format causal graph
        causal_graph = """
        graph [
            directed 1
            node [ id "fast_shipping" label "fast_shipping" ]
            node [ id "Late_delivery_risk" label "Late_delivery_risk" ]
            node [ id "region_code" label "region_code" ]
            node [ id "market_code" label "market_code" ]
            node [ id "shipping_distance_km" label "shipping_distance_km" ]
            node [ id "segment_code" label "segment_code" ]
            edge [ source "region_code" target "fast_shipping" ]
            edge [ source "region_code" target "Late_delivery_risk" ]
            edge [ source "market_code" target "fast_shipping" ]
            edge [ source "market_code" target "Late_delivery_risk" ]
            edge [ source "shipping_distance_km" target "fast_shipping" ]
            edge [ source "shipping_distance_km" target "Late_delivery_risk" ]
            edge [ source "segment_code" target "fast_shipping" ]
            edge [ source "fast_shipping" target "Late_delivery_risk" ]
        ]
        """

        model = CausalModel(
            data=df,
            treatment="fast_shipping",
            outcome="Late_delivery_risk",
            graph=causal_graph
        )

        print("[Causal] Model built. Displaying causal graph...")
        return model

    def estimate_causal_effect(self, model: CausalModel,
                                method: str = "backdoor.propensity_score_matching"):
        """
        Estimate Average Treatment Effect (ATE) of fast shipping on late delivery risk.

        Args:
            method: Estimation method. Options:
                'backdoor.propensity_score_matching'   — PSM (robust, interpretable)
                'backdoor.linear_regression'           — Linear regression adjustment
                'backdoor.propensity_score_weighting'  — IPW

        Returns:
            Causal estimate object with ATE value
        """
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        try:
            print(f"[Causal] Identified estimand:\n{identified_estimand}")
        except Exception:
            print("[Causal] Identified estimand: (Printed representation skipped due to character encoding issues)")

        estimate = model.estimate_effect(
            identified_estimand,
            method_name=method,
            target_units="ate",   # Average Treatment Effect
            confidence_intervals=True,
            test_significance=True
        )

        print(f"\n[Causal] Estimated ATE ({method}):")
        print(f"  Fast shipping reduces late delivery risk by: "
              f"{abs(estimate.value):.4f} (probability units)")
        print(f"  Interpretation: Upgrading from Standard to First Class reduces "
              f"P(late delivery) by ~{abs(estimate.value)*100:.1f} percentage points")

        return estimate

    def refute_estimate(self, model: CausalModel, estimate, identified_estimand):
        """
        Run refutation tests to check robustness of causal estimate.
        A good causal estimate should NOT change much under placebo/random treatment.
        """
        print("\n[Causal] Running refutation tests...")

        # Placebo treatment (should give ~0 effect)
        placebo = model.refute_estimate(
            identified_estimand, estimate,
            method_name="placebo_treatment_refuter",
            placebo_type="permute"
        )
        print(f"  Placebo test — new effect: {placebo.new_effect:.4f} "
              f"(should be near 0 if causal)")

        # Random common cause (robustness to unmeasured confounding)
        random_cc = model.refute_estimate(
            identified_estimand, estimate,
            method_name="random_common_cause"
        )
        print(f"  Random common cause — new effect: {random_cc.new_effect:.4f} "
              f"(should be similar to original)")

        return {"placebo": placebo, "random_common_cause": random_cc}

    def run_full_causal_analysis(self, df: pd.DataFrame) -> dict:
        """Run the complete causal analysis pipeline."""
        df_prepared = self.prepare_data(df)
        model = self.build_causal_model(df_prepared)
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

        estimate_psm = self.estimate_causal_effect(
            model, method="backdoor.propensity_score_matching"
        )
        estimate_lr = self.estimate_causal_effect(
            model, method="backdoor.linear_regression"
        )

        # Define edges for plotting (reflecting the GML graph)
        edges = [
            ("region_code", "fast_shipping"), ("region_code", "Late_delivery_risk"),
            ("market_code", "fast_shipping"), ("market_code", "Late_delivery_risk"),
            ("shipping_distance_km", "fast_shipping"), ("shipping_distance_km", "Late_delivery_risk"),
            ("segment_code", "fast_shipping"), ("fast_shipping", "Late_delivery_risk")
        ]
        
        # Plotting before refutations to ensure files are saved even if refutation crashes
        self.plot_causal_graph(edges, filename="late_delivery_causal_dag.png")
        self.plot_effect_results(estimate_psm, title="Effect of Fast Shipping on Late Delivery", filename="late_delivery_effect.png")

        # Refutations (wrapped to prevent skipping plots if they fail)
        refutation_results = {}
        try:
            refutation_results = self.refute_estimate(model, estimate_psm, identified_estimand)
        except Exception as e:
            print(f"[Causal] Refutation tests failed (skipping): {e}")

        return {
            "psm_ate": estimate_psm.value,
            "lr_ate": estimate_lr.value,
            "refutations": refutation_results,
            "model": model,
            "estimand": identified_estimand
        }

    def plot_causal_graph(self, edges, filename="causal_dag.png"):
        """Plot the causal DAG using networkx (fallback since dot/graphviz is missing)."""
        try:
            g = nx.DiGraph()
            g.add_edges_from(edges)
            
            if len(g.nodes) == 0:
                print("[Causal] Warning: No nodes found for plotting.")
                return

            plt.figure(figsize=(10, 6))
            pos = nx.shell_layout(g) # shell layout often looks better for small DAGs
            nx.draw(g, pos, with_labels=True, node_color='lightblue', 
                    edge_color='gray', node_size=3000, font_size=10, 
                    font_weight='bold', arrows=True, connectionstyle='arc3,rad=0.1')
            plt.title("Causal DAG Structure")
            plt.tight_layout()
            
            save_path = os.path.join(self.output_dir, filename)
            plt.savefig(save_path, dpi=150)
            plt.close()
            print(f"[Causal] Saved DAG plot to {save_path}")
        except Exception as e:
            print(f"[Causal] Failed to plot DAG: {e}")

    def plot_effect_results(self, estimate, title: str, filename: str):
        """Plot the estimated causal effect with confidence intervals."""
        try:
            effect_value = estimate.value
            ci = estimate.get_confidence_intervals()
            
            # ci might be a tuple or an array. Handle safely.
            if isinstance(ci, (tuple, list, np.ndarray)) and len(ci) == 2:
                lower, upper = ci[0], ci[1]
                # In dowhy, ci can be returned as ndarray of shape (2,) or tuple of arrays
                if isinstance(lower, np.ndarray): lower = lower[0]
                if isinstance(upper, np.ndarray): upper = upper[0]
            else:
                lower, upper = effect_value, effect_value # Fallback if CI not available
            
            plt.figure(figsize=(6, 4))
            plt.errorbar(x=["Effect"], y=[effect_value], yerr=[[effect_value - lower], [upper - effect_value]], 
                         fmt='o', color='blue', ecolor='red', capsize=5, markersize=8)
            plt.axhline(0, color='gray', linestyle='--')
            plt.title(title)
            plt.ylabel("Average Treatment Effect (ATE)")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, filename), dpi=150)
            plt.close()
            print(f"[Causal] Saved effect plot to {filename}")
        except Exception as e:
            print(f"[Causal] Failed to plot effect: {e}")

    def run_fraud_causal_analysis(self, df: pd.DataFrame) -> dict:
        """
        Analyze whether high discounts causally increase fraud suspicion.
        Treatment: High Discount (> 20%)
        Outcome: SUSPECTED_FRAUD
        Confounders: Order Region, Customer Segment 
        """
        print("\n[Causal] Running Fraud Causal Analysis...")
        df = df.copy()
        
        # Outcome variable
        df["fraud_suspected"] = (df["Order Status"] == "SUSPECTED_FRAUD").astype(int)
        
        # Treatment variable
        df["high_discount"] = (df["Order Item Discount Rate"] > 0.20).astype(int)
        
        # Confounders
        df["region_code"] = df["Order Region"].astype("category").cat.codes
        df["segment_code"] = df["Customer Segment"].astype("category").cat.codes
        
        causal_graph = """
        graph [
            directed 1
            node [ id "high_discount" label "high_discount" ]
            node [ id "fraud_suspected" label "fraud_suspected" ]
            node [ id "region_code" label "region_code" ]
            node [ id "segment_code" label "segment_code" ]
            edge [ source "region_code" target "high_discount" ]
            edge [ source "region_code" target "fraud_suspected" ]
            edge [ source "segment_code" target "high_discount" ]
            edge [ source "segment_code" target "fraud_suspected" ]
            edge [ source "high_discount" target "fraud_suspected" ]
        ]
        """
        
        try:
            model = CausalModel(
                data=df,
                treatment="high_discount",
                outcome="fraud_suspected",
                graph=causal_graph
            )
            
            identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
            estimate = model.estimate_effect(
                identified_estimand,
                method_name="backdoor.propensity_score_matching",
                target_units="ate"
            )
            
            print(f"[Causal] Fraud ATE (High Discount -> Fraud): {estimate.value:.4f}")
            
            # Plot graphs
            # Define edges for plotting (reflecting the GML graph)
            fraud_edges = [
                ("region_code", "high_discount"), ("region_code", "fraud_suspected"),
                ("segment_code", "high_discount"), ("segment_code", "fraud_suspected"),
                ("high_discount", "fraud_suspected")
            ]
            self.plot_causal_graph(fraud_edges, filename="fraud_causal_dag.png")
            
            # Plot effect (might not have CI if not requested during estimation)
            estimate_with_ci = model.estimate_effect(
                identified_estimand,
                method_name="backdoor.propensity_score_matching",
                target_units="ate",
                confidence_intervals=True
            )
            self.plot_effect_results(estimate_with_ci, 
                                     title="Effect of High Discount on Fraud Risk",
                                     filename="fraud_effect.png")
                                     
            return {"model": model, "estimate": estimate_with_ci}
            
        except Exception as e:
            print(f"[Causal] Fraud causal analysis failed: {e}")
            return {}
