"""
Data Preprocessing & Geographic Feature Engineering
DataCo Supply Chain Dataset - Geographic & Regional Demand Analysis
"""

import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
import warnings
warnings.filterwarnings('ignore')


def haversine(lon1, lat1, lon2, lat2):
    """Calculate the great-circle distance (km) between two points using Haversine formula."""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * 6371 * asin(sqrt(a))


def load_data(filepath='DataCoSupplyChainDataset.csv'):
    """Load and return the raw dataset."""
    print("[1/6] Loading dataset...")
    df = pd.read_csv(filepath, encoding='latin-1')
    print(f"  Loaded {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def clean_data(df):
    """Basic cleaning: parse dates, drop duplicates, handle missing values."""
    print("[2/6] Cleaning data...")
    df = df.copy()

    # Parse date columns
    for col in ['order date (DateOrders)', 'shipping date (DateOrders)']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format='mixed', dayfirst=False, errors='coerce')

    # Drop fully-duplicate rows
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"  Removed {before - len(df)} duplicate rows")

    # Numeric columns: fill NaN with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    # Categorical columns: fill NaN with mode
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df[col].isna().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)

    print(f"  Remaining NaN: {df.isna().sum().sum()}")
    return df


def engineer_geographic_features(df):
    """Create geographic and regional features from the dataset."""
    print("[3/6] Engineering geographic features...")
    df = df.copy()

    # --- 1. Shipping Distance (Haversine) ---
    # The dataset has Latitude/Longitude for order origin. We'll use Order Region
    # as a proxy for destination geography since destination coords aren't explicit.
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        # Compute per-Order-Country average lat/lon as destination proxy
        dest_coords = df.groupby('Order Country')[['Latitude', 'Longitude']].transform('mean')
        df['Dest_Lat'] = dest_coords['Latitude']
        df['Dest_Lon'] = dest_coords['Longitude']

        df['Shipping_Distance_KM'] = df.apply(
            lambda r: haversine(r['Longitude'], r['Latitude'], r['Dest_Lon'], r['Dest_Lat'])
            if pd.notna(r['Latitude']) and pd.notna(r['Longitude']) else 0,
            axis=1
        )
    else:
        df['Shipping_Distance_KM'] = 0

    # --- 2. Macro Region mapping (use existing Order Region or Market) ---
    if 'Order Region' in df.columns:
        df['Macro_Region'] = df['Order Region'].str.strip()
    elif 'Market' in df.columns:
        df['Macro_Region'] = df['Market'].str.strip()
    else:
        df['Macro_Region'] = 'Unknown'

    # --- 3. Is Cross-Border ---
    if 'Customer Country' in df.columns and 'Order Country' in df.columns:
        df['Is_Cross_Border'] = (df['Customer Country'].str.strip() != df['Order Country'].str.strip()).astype(int)
    else:
        df['Is_Cross_Border'] = 0

    print(f"  Unique Macro Regions: {df['Macro_Region'].nunique()}")
    print(f"  Cross-border orders: {df['Is_Cross_Border'].sum():,}")
    return df


def engineer_temporal_features(df):
    """Create temporal and seasonal features."""
    print("[4/6] Engineering temporal features...")
    df = df.copy()

    date_col = 'order date (DateOrders)'
    if date_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df['Order_Month'] = df[date_col].dt.month
        df['Order_Year'] = df[date_col].dt.year
        df['Order_DayOfWeek'] = df[date_col].dt.dayofweek
        df['Order_Quarter'] = df[date_col].dt.quarter
        df['Is_Weekend'] = df['Order_DayOfWeek'].isin([5, 6]).astype(int)
        df['Is_Month_End'] = df[date_col].dt.is_month_end.astype(int)
    else:
        # Fallback: try to derive from existing columns
        if 'Order Month' not in df.columns:
            df['Order_Month'] = 1
        else:
            df['Order_Month'] = df['Order Month'] if 'Order Month' in df.columns else 1

    # --- Season by hemisphere hint ---
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'

    df['Season'] = df['Order_Month'].apply(get_season)

    # --- Weather risk proxy (higher in winter months for Northern hemisphere) ---
    weather_risk = {12: 0.8, 1: 0.9, 2: 0.7, 3: 0.5, 4: 0.3, 5: 0.2,
                    6: 0.3, 7: 0.2, 8: 0.3, 9: 0.5, 10: 0.6, 11: 0.7}
    df['Weather_Risk_Score'] = df['Order_Month'].map(weather_risk).fillna(0.5)

    # --- Holiday proximity flag (US-centric major shopping periods) ---
    df['Holiday_Flag'] = 0
    if 'Order_Month' in df.columns:
        df.loc[(df['Order_Month'] == 11) | (df['Order_Month'] == 12), 'Holiday_Flag'] = 1

    print(f"  Temporal features created: Order_Month, Order_Year, Order_DayOfWeek, Season, etc.")
    return df


def engineer_regional_aggregates(df):
    """Compute aggregated regional statistics and merge back."""
    print("[5/6] Computing regional aggregate features...")
    df = df.copy()

    agg_dict = {}
    if 'Late_delivery_risk' in df.columns:
        agg_dict['Late_delivery_risk'] = 'mean'
    if 'Days for shipping (real)' in df.columns:
        agg_dict['Days for shipping (real)'] = ['mean', 'std']
    if 'Sales' in df.columns:
        agg_dict['Sales'] = ['mean', 'sum', 'count']
    if 'Order Item Discount Rate' in df.columns:
        agg_dict['Order Item Discount Rate'] = 'mean'

    if not agg_dict:
        print("  No aggregatable columns found, skipping.")
        return df

    regional_stats = df.groupby('Macro_Region').agg(agg_dict)
    regional_stats.columns = ['_'.join(col).strip('_') for col in regional_stats.columns]
    regional_stats = regional_stats.rename(columns={
        'Late_delivery_risk_mean': 'Region_Avg_Late_Delivery_Rate',
        'Days for shipping (real)_mean': 'Region_Avg_Shipping_Days',
        'Days for shipping (real)_std': 'Region_Std_Shipping_Days',
        'Sales_mean': 'Region_Avg_Order_Value',
        'Sales_sum': 'Region_Total_Sales',
        'Sales_count': 'Region_Total_Orders',
        'Order Item Discount Rate_mean': 'Region_Avg_Discount_Rate'
    })
    regional_stats = regional_stats.reset_index()

    df = df.merge(regional_stats, on='Macro_Region', how='left')
    print(f"  Added {len(regional_stats.columns) - 1} regional aggregate features")
    return df


def prepare_full_dataset(filepath='DataCoSupplyChainDataset.csv'):
    """Full preprocessing pipeline: load -> clean -> feature engineer."""
    print("=" * 70)
    print("DATA PREPROCESSING PIPELINE")
    print("=" * 70)

    df = load_data(filepath)
    df = clean_data(df)
    df = engineer_geographic_features(df)
    df = engineer_temporal_features(df)
    df = engineer_regional_aggregates(df)

    print("[6/6] Preprocessing complete!")
    print(f"  Final dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print("=" * 70)
    return df


if __name__ == '__main__':
    df = prepare_full_dataset()
    print("\nSample columns:")
    print(df.columns.tolist())
