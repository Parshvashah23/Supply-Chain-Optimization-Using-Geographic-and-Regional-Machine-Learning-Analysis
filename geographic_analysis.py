"""
Geographic Exploratory Data Analysis & Statistical Tests
DataCo Supply Chain Dataset - Geographic & Regional Demand Analysis
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.neighbors import NearestNeighbors
import os
import warnings
warnings.filterwarnings('ignore')

PLOT_DIR = 'plots'


def ensure_plot_dir():
    os.makedirs(PLOT_DIR, exist_ok=True)


def plot_regional_late_delivery(df):
    """Bar chart: Late delivery rate by region."""
    ensure_plot_dir()
    region_late = df.groupby('Macro_Region')['Late_delivery_risk'].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    region_late.plot(kind='bar', color='salmon', edgecolor='darkred', ax=ax)
    ax.set_title('Late Delivery Rate by Region', fontsize=14, fontweight='bold')
    ax.set_ylabel('Late Delivery Rate')
    ax.set_xlabel('Region')
    ax.axhline(y=df['Late_delivery_risk'].mean(), color='black', linestyle='--', label='Global Average')
    ax.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/01_late_delivery_by_region.png', dpi=150)
    plt.close()
    print("  Saved: 01_late_delivery_by_region.png")


def plot_sales_by_region(df):
    """Bar chart: Total and average sales by region."""
    ensure_plot_dir()
    region_sales = df.groupby('Macro_Region')['Sales'].agg(['sum', 'mean', 'count']).sort_values('sum', ascending=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    region_sales['sum'].plot(kind='bar', color='steelblue', ax=axes[0])
    axes[0].set_title('Total Sales by Region', fontweight='bold')
    axes[0].set_ylabel('Total Sales ($)')
    axes[0].tick_params(axis='x', rotation=45)

    region_sales['mean'].plot(kind='bar', color='mediumseagreen', ax=axes[1])
    axes[1].set_title('Average Order Value by Region', fontweight='bold')
    axes[1].set_ylabel('Avg Sales ($)')
    axes[1].tick_params(axis='x', rotation=45)

    region_sales['count'].plot(kind='bar', color='darkorange', ax=axes[2])
    axes[2].set_title('Order Volume by Region', fontweight='bold')
    axes[2].set_ylabel('Number of Orders')
    axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/02_sales_by_region.png', dpi=150)
    plt.close()
    print("  Saved: 02_sales_by_region.png")


def plot_shipping_days_boxplot(df):
    """Box plot: Shipping days distribution by region."""
    ensure_plot_dir()
    fig, ax = plt.subplots(figsize=(14, 6))
    top_regions = df['Macro_Region'].value_counts().head(15).index
    subset = df[df['Macro_Region'].isin(top_regions)]
    sns.boxplot(data=subset, x='Macro_Region', y='Days for shipping (real)', ax=ax, palette='Set2')
    ax.set_title('Shipping Days Distribution by Region', fontsize=14, fontweight='bold')
    ax.set_xlabel('Region')
    ax.set_ylabel('Days for Shipping (Real)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/03_shipping_days_boxplot.png', dpi=150)
    plt.close()
    print("  Saved: 03_shipping_days_boxplot.png")


def plot_seasonal_demand(df):
    """Line chart: Monthly demand trends by top regions."""
    ensure_plot_dir()
    if 'Order_Month' not in df.columns:
        print("  Skipped seasonal demand plot (no Order_Month column)")
        return

    top_regions = df['Macro_Region'].value_counts().head(6).index
    monthly = df[df['Macro_Region'].isin(top_regions)].groupby(
        ['Macro_Region', 'Order_Month'])['Sales'].sum().reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))
    for region in top_regions:
        r_data = monthly[monthly['Macro_Region'] == region]
        ax.plot(r_data['Order_Month'], r_data['Sales'], marker='o', label=region)

    ax.set_title('Monthly Sales Trend by Region', fontsize=14, fontweight='bold')
    ax.set_xlabel('Month')
    ax.set_ylabel('Total Sales ($)')
    ax.set_xticks(range(1, 13))
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/04_seasonal_demand.png', dpi=150)
    plt.close()
    print("  Saved: 04_seasonal_demand.png")


def plot_correlation_matrix(df):
    """Heatmap: Correlation among key numeric features."""
    ensure_plot_dir()
    numeric_features = [
        'Days for shipping (real)', 'Days for shipment (scheduled)',
        'Sales', 'Order Item Quantity', 'Order Item Discount Rate',
        'Benefit per order', 'Late_delivery_risk', 'Shipping_Distance_KM',
        'Weather_Risk_Score', 'Region_Avg_Late_Delivery_Rate',
        'Region_Avg_Shipping_Days', 'Region_Avg_Order_Value'
    ]
    available = [c for c in numeric_features if c in df.columns]
    if len(available) < 3:
        print("  Skipped correlation matrix (insufficient numeric features)")
        return

    corr = df[available].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, ax=ax)
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/05_correlation_matrix.png', dpi=150)
    plt.close()
    print("  Saved: 05_correlation_matrix.png")


def plot_delivery_status_by_region(df):
    """Stacked bar: Delivery status distribution per region."""
    ensure_plot_dir()
    if 'Delivery Status' not in df.columns:
        return
    top_regions = df['Macro_Region'].value_counts().head(10).index
    ct = pd.crosstab(df[df['Macro_Region'].isin(top_regions)]['Macro_Region'],
                     df['Delivery Status'], normalize='index') * 100

    fig, ax = plt.subplots(figsize=(14, 6))
    ct.plot(kind='bar', stacked=True, ax=ax, colormap='Set2')
    ax.set_title('Delivery Status Distribution by Region (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Percentage')
    ax.legend(title='Status', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/06_delivery_status_by_region.png', dpi=150)
    plt.close()
    print("  Saved: 06_delivery_status_by_region.png")


def plot_customer_segment_by_region(df):
    """Grouped bar: Customer segment breakdown by region."""
    ensure_plot_dir()
    if 'Customer Segment' not in df.columns:
        return
    top_regions = df['Macro_Region'].value_counts().head(10).index
    ct = pd.crosstab(df[df['Macro_Region'].isin(top_regions)]['Macro_Region'],
                     df['Customer Segment'])

    fig, ax = plt.subplots(figsize=(12, 6))
    ct.plot(kind='bar', ax=ax, colormap='Pastel1', edgecolor='gray')
    ax.set_title('Customer Segment Distribution by Region', fontsize=14, fontweight='bold')
    ax.set_ylabel('Count')
    ax.legend(title='Segment')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/07_customer_segment_by_region.png', dpi=150)
    plt.close()
    print("  Saved: 07_customer_segment_by_region.png")


def plot_morans_i_scatter(df):
    """Scatter plot: Moran's I Spatial Autocorrelation for Late Delivery Risk."""
    ensure_plot_dir()
    
    if 'Late_delivery_risk' not in df.columns or 'Latitude' not in df.columns or 'Longitude' not in df.columns:
        print("  Skipped Moran's I plot (missing required spatial columns)")
        return

    # Aggregate by Order City to get spatial points
    if 'Order City' in df.columns:
        spatial_df = df.groupby('Order City').agg({
            'Latitude': 'mean',
            'Longitude': 'mean',
            'Late_delivery_risk': 'mean'
        }).dropna()
    else:
        # Fallback to pure coordinates if city is missing
        spatial_df = df.groupby(['Latitude', 'Longitude']).agg({
            'Late_delivery_risk': 'mean'
        }).reset_index().dropna()

    if len(spatial_df) < 10:
        print("  Skipped Moran's I plot (not enough spatial points)")
        return

    coords = spatial_df[['Latitude', 'Longitude']].values
    y = spatial_df['Late_delivery_risk'].values

    # Standardize the variable (z-score)
    y_mean = y.mean()
    y_std = y.std()
    
    if y_std == 0:
        print("  Skipped Moran's I plot (Zero variance in Late_delivery_risk)")
        return
        
    z = (y - y_mean) / y_std

    # Build K-Nearest Neighbors Spatial Weight Matrix (W)
    k = min(8, len(coords) - 1)
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    # Compute spatial lag (average of neighbors' z-scores)
    w_z = np.zeros(len(z))
    for i in range(len(z)):
        # indices[i][1:] excludes the point itself (index 0)
        neighbors = indices[i][1:]
        w_z[i] = np.mean(z[neighbors])

    # Calculate Moran's I (slope of regression w_z vs z)
    # I = sum(z * w_z) / sum(z^2)
    morans_i = np.sum(z * w_z) / np.sum(z**2)
    
    # Calculate approximate p-value using scipy
    slope, intercept, r_value, p_value, std_err = stats.linregress(z, w_z)

    # Visualization
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Scatter points based on quadrants
    colors = []
    labels = []
    for zi, wzi in zip(z, w_z):
        if zi >= 0 and wzi >= 0:
            colors.append('red')      # High-High
            labels.append('HH')
        elif zi < 0 and wzi < 0:
            colors.append('blue')     # Low-Low
            labels.append('LL')
        elif zi >= 0 and wzi < 0:
            colors.append('lightblue')# High-Low
            labels.append('HL')
        else:
            colors.append('pink')     # Low-High
            labels.append('LH')
            
    ax.scatter(z, w_z, c=colors, alpha=0.6, edgecolors='white', s=50)
    
    # Regression line
    x_range = np.linspace(z.min(), z.max(), 100)
    ax.plot(x_range, intercept + slope * x_range, color='black', linestyle='--', 
            linewidth=2, label=f"Moran's I: {morans_i:.2f} (p={p_value:.3f})")

    # Quadrant lines
    ax.axhline(0, color='gray', linestyle=':')
    ax.axvline(0, color='gray', linestyle=':')

    # Labels and Titles
    ax.set_title("Moran's I Spatial Autocorrelation\nLate Delivery Risk", fontsize=14, fontweight='bold')
    ax.set_xlabel('Local Late Delivery Risk (Standardized)', fontsize=12)
    ax.set_ylabel('Spatial Lag of Late Delivery Risk', fontsize=12)
    
    # Quadrant text annotations
    lim_val = max(abs(z.min()), abs(z.max()), abs(w_z.min()), abs(w_z.max())) * 0.8
    ax.text(lim_val, lim_val, 'HH', fontsize=16, alpha=0.4, ha='center', va='center')
    ax.text(-lim_val, -lim_val, 'LL', fontsize=16, alpha=0.4, ha='center', va='center')
    ax.text(lim_val, -lim_val, 'HL', fontsize=16, alpha=0.4, ha='center', va='center')
    ax.text(-lim_val, lim_val, 'LH', fontsize=16, alpha=0.4, ha='center', va='center')

    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/08_morans_i_scatter.png', dpi=150)
    plt.close()
    print("  Saved: 08_morans_i_scatter.png")


# ─────────────────────────────────────────────────────────────
# STATISTICAL TESTS
# ─────────────────────────────────────────────────────────────

def run_statistical_tests(df):
    """Run ANOVA, Chi-Square, Kruskal-Wallis, and correlation tests."""
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)

    results = {}

    # 1. ANOVA - Shipping days across regions
    print("\n--- ANOVA: Shipping Days across Regions ---")
    regions = df['Macro_Region'].unique()
    groups = [df[df['Macro_Region'] == r]['Days for shipping (real)'].dropna() for r in regions]
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) >= 2:
        f_stat, p_val = stats.f_oneway(*groups)
        results['ANOVA_shipping_days'] = {'F-statistic': f_stat, 'p-value': p_val}
        print(f"  F-statistic: {f_stat:.4f}")
        print(f"  p-value: {p_val:.6f}")
        print(f"  {'Significant' if p_val < 0.05 else 'Not significant'} difference across regions")

    # 2. Chi-Square - Late delivery vs Region
    print("\n--- Chi-Square: Late Delivery Risk vs Region ---")
    if 'Late_delivery_risk' in df.columns:
        contingency = pd.crosstab(df['Macro_Region'], df['Late_delivery_risk'])
        chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
        results['ChiSquare_late_delivery'] = {'chi2': chi2, 'p-value': p_val, 'dof': dof}
        print(f"  Chi-square: {chi2:.4f}")
        print(f"  p-value: {p_val:.6f}")
        print(f"  Degrees of freedom: {dof}")
        print(f"  {'Significant' if p_val < 0.05 else 'Not significant'} association")

    # 3. Kruskal-Wallis - Non-parametric test for Sales across regions
    print("\n--- Kruskal-Wallis: Sales across Regions ---")
    groups_sales = [df[df['Macro_Region'] == r]['Sales'].dropna() for r in regions]
    groups_sales = [g for g in groups_sales if len(g) >= 2]
    if len(groups_sales) >= 2:
        h_stat, p_val = stats.kruskal(*groups_sales)
        results['KruskalWallis_sales'] = {'H-statistic': h_stat, 'p-value': p_val}
        print(f"  H-statistic: {h_stat:.4f}")
        print(f"  p-value: {p_val:.6f}")
        print(f"  {'Significant' if p_val < 0.05 else 'Not significant'} difference")

    # 4. Correlation - Distance vs Delivery Time
    print("\n--- Correlation: Shipping Distance vs Delivery Time ---")
    if 'Shipping_Distance_KM' in df.columns:
        valid = df[['Shipping_Distance_KM', 'Days for shipping (real)']].dropna()
        valid = valid[valid['Shipping_Distance_KM'] > 0]
        if len(valid) > 10:
            r, p_val = stats.pearsonr(valid['Shipping_Distance_KM'], valid['Days for shipping (real)'])
            results['Correlation_distance_shipping'] = {'Pearson_r': r, 'p-value': p_val}
            print(f"  Pearson r: {r:.4f}")
            print(f"  p-value: {p_val:.6f}")

    # 5. T-test - Weekend vs Weekday delivery performance
    print("\n--- T-test: Weekend vs Weekday Shipping Days ---")
    if 'Is_Weekend' in df.columns:
        weekend = df[df['Is_Weekend'] == 1]['Days for shipping (real)'].dropna()
        weekday = df[df['Is_Weekend'] == 0]['Days for shipping (real)'].dropna()
        if len(weekend) > 10 and len(weekday) > 10:
            t_stat, p_val = stats.ttest_ind(weekend, weekday)
            results['Ttest_weekend_weekday'] = {'t-statistic': t_stat, 'p-value': p_val}
            print(f"  t-statistic: {t_stat:.4f}")
            print(f"  p-value: {p_val:.6f}")
            print(f"  Weekend avg: {weekend.mean():.2f} days, Weekday avg: {weekday.mean():.2f} days")

    print("\n" + "=" * 70)
    return results


def run_eda(df):
    """Run all EDA visualizations and statistical tests."""
    print("\n" + "=" * 70)
    print("EXPLORATORY DATA ANALYSIS & VISUALIZATIONS")
    print("=" * 70)

    print("\nGenerating plots...")
    plot_regional_late_delivery(df)
    plot_sales_by_region(df)
    plot_shipping_days_boxplot(df)
    plot_seasonal_demand(df)
    plot_correlation_matrix(df)
    plot_delivery_status_by_region(df)
    plot_customer_segment_by_region(df)
    plot_morans_i_scatter(df)

    stat_results = run_statistical_tests(df)

    print(f"\nAll plots saved to '{PLOT_DIR}/' directory.")
    return stat_results


if __name__ == '__main__':
    from data_preprocessing import prepare_full_dataset
    df = prepare_full_dataset()
    run_eda(df)
