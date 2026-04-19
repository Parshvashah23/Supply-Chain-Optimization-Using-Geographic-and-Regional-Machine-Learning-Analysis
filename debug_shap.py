
import os
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

def debug_shap():
    model_dir = "models/regional_forecast/"
    xgb_path = f"{model_dir}global_xgboost_forecast.pkl"
    rf_path = f"{model_dir}global_rf_classifier.pkl"
    
    if not os.path.exists(xgb_path) or not os.path.exists(rf_path):
        print("Models not found. Run index.py first to train them.")
        return

    print("Loading models...")
    xgb_demand = joblib.load(xgb_path)
    rf_classifier = joblib.load(rf_path)
    print("Models loaded.")

    # Create dummy data with expected shapes
    # (Assuming 47 features for classifier and some for regressor)
    # Actually, let's just make a small dataframe
    X_class = pd.DataFrame(np.random.rand(10, 47), columns=[f"feat_{i}" for i in range(47)]).astype(np.float32)
    X_demand = pd.DataFrame(np.random.rand(10, 25), columns=[f"feat_{i}" for i in range(25)]).astype(np.float32)

    print("Testing XGBoost TreeExplainer...")
    try:
        explainer = shap.TreeExplainer(xgb_demand)
        print("Explainer created.")
        shap_values = explainer.shap_values(X_demand, check_additivity=False)
        print("SHAP values computed for XGBoost.")
    except Exception as e:
        print(f"XGBoost SHAP Error: {e}")

    print("\nTesting Random Forest TreeExplainer...")
    try:
        explainer = shap.TreeExplainer(rf_classifier)
        print("Explainer created.")
        shap_values = explainer.shap_values(X_class, check_additivity=False)
        print("SHAP values computed for RF.")
    except Exception as e:
        print(f"RF SHAP Error: {e}")

if __name__ == "__main__":
    debug_shap()
