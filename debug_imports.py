import time

print("Importing os/sys/time etc...")
import os
import sys

print("Importing pandas...")
import pandas as pd

print("Importing geographic_analysis/clustering...")
from geographic_analysis import run_eda
from clustering import run_clustering

print("Importing demand/classification/pipeline...")
from demand_forecasting import run_demand_forecasting
from late_delivery_classifier import run_late_delivery_classification
from model_pipeline import save_models, generate_insights, RegionalDemandPredictor

print("Importing explainability...")
from explainability.shap_explainer import SHAPExplainer
from explainability.lime_explainer import LIMEExplainer

print("Importing risk...")
from risk.fraud_detector import FraudDetector
from risk.cancellation_predictor import CancellationPredictor
from risk.composite_risk_scorer import CompositeRiskScorer

print("Importing causal...")
from causal.causal_inference import SupplyChainCausalAnalyzer

print("Importing DL...")
from demand_forecasting_lstm import LSTMDemandForecaster
from demand_forecasting_hybrid import HybridDemandForecaster

print("Importing financial...")
from financial.cost_sensitive_classifier import CostSensitiveOptimizer
from financial.pareto_optimizer import ParetoOptimizer
from financial.esg_scorer import ESGScorer

print("Importing cluster drift...")
from clustering_resilience import ResilienceIndexCalculator
from clustering_drift import ClusterDriftDetector

print("All imports successful!")
