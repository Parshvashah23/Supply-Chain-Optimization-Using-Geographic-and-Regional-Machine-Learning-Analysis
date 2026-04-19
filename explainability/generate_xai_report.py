import os
import base64
import json
import pandas as pd
from datetime import datetime


def image_to_base64(image_path: str) -> str:
    """Convert a plot image to base64 for embedding in HTML."""
    if not os.path.exists(image_path):
        return ""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def generate_xai_html_section(shap_plot_dir: str = "plots/shap/",
                               lime_results: list = None,
                               output_path: str = "XAI_Report.html"):
    """
    Generate a standalone XAI report embedding all SHAP and LIME outputs.
    Appends to or creates an HTML file.
    """

    plots = {
        "Global Demand SHAP (Bar)": f"{shap_plot_dir}demand_shap_global.png",
        "Global Demand SHAP (Beeswarm)": f"{shap_plot_dir}demand_shap_beeswarm.png",
        "Late Delivery Classifier SHAP (Bar)": f"{shap_plot_dir}classifier_shap_global.png",
        "Late Delivery Classifier SHAP (Beeswarm)": f"{shap_plot_dir}classifier_shap_beeswarm.png",
        "Regional Feature Importance Heatmap": f"{shap_plot_dir}regional_shap_heatmap.png",
    }

    # Build HTML
    sections_html = ""
    for title, path in plots.items():
        b64 = image_to_base64(path)
        if b64:
            sections_html += f"""
            <div class="plot-card">
                <h3>{title}</h3>
                <img src="data:image/png;base64,{b64}" style="max-width:100%; border-radius:8px;">
            </div>
            """

    # LIME counterfactual table
    lime_table = ""
    if lime_results:
        rows = ""
        for res in lime_results[:20]:  # Show top 20
            prob = round(res["predicted_probability"] * 100, 1)
            top_feat = res["top_features"][0][0] if res["top_features"] else "N/A"
            rows += f"<tr><td>{res['order_id']}</td><td>{prob}%</td><td>{top_feat}</td></tr>"

        lime_table = f"""
        <h2>LIME — High-Risk Order Explanations (Top 20)</h2>
        <table>
            <thead><tr><th>Order ID</th><th>Late Risk %</th><th>Top Driver</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>
        """

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>XAI Report — Supply Chain ML</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1100px; margin: auto; padding: 40px; background: #fafafa; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 40px; }}
        h3 {{ color: #555; font-size: 14px; margin: 10px 0; }}
        .plot-card {{ background: white; border-radius: 8px; padding: 20px; margin: 20px 0;
                      box-shadow: 0 1px 4px rgba(0,0,0,0.1); }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th {{ background: #3498db; color: white; padding: 10px 14px; text-align: left; }}
        td {{ border: 1px solid #ddd; padding: 9px 14px; }}
        tr:nth-child(even) {{ background: #f5f5f5; }}
        .meta {{ color: #888; font-size: 13px; margin-bottom: 30px; }}
    </style>
</head>
<body>
    <h1>Explainability & Trust Layer — XAI Report</h1>
    <p class="meta">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} |
       Dataset: DataCo Smart Supply Chain</p>

    <h2>SHAP Analysis — Global Model Explanations</h2>
    <p>SHAP (SHapley Additive exPlanations) decomposes each prediction into per-feature
       contributions. Bar charts show mean absolute impact; beeswarm plots show direction
       and distribution of each feature's effect.</p>

    {sections_html}
    {lime_table}
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)
    print(f"[XAI Report] Saved to {output_path}")
