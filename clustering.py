"""
Geographic Clustering Analysis
K-Means, DBSCAN, and Hierarchical Clustering
DataCo Supply Chain Dataset
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
import os
import warnings
warnings.filterwarnings('ignore')

PLOT_DIR = 'plots'


def ensure_plot_dir():
    os.makedirs(PLOT_DIR, exist_ok=True)


def prepare_clustering_features(df):
    """Select and prepare features for clustering."""
    feature_candidates = [
        'Latitude', 'Longitude',
        'Shipping_Distance_KM',
        'Days for shipping (real)',
        'Sales', 'Order Item Quantity',
        'Late_delivery_risk',
        'Weather_Risk_Score',
        'Region_Avg_Late_Delivery_Rate',
        'Region_Avg_Order_Value',
    ]
    available = [c for c in feature_candidates if c in df.columns]
    print(f"  Clustering features ({len(available)}): {available}")

    X = df[available].copy()
    # Drop rows with NaN
    X = X.dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X, X_scaled, available, scaler


def run_kmeans(df, X_scaled, feature_names):
    """K-Means clustering with elbow method and silhouette analysis."""
    ensure_plot_dir()
    print("\n--- K-Means Clustering ---")

    # Elbow method
    K_range = range(2, 11)
    inertias = []
    silhouette_scores = []

    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_scaled, labels, sample_size=min(10000, len(X_scaled)))
        silhouette_scores.append(sil)
        print(f"  K={k}: Inertia={km.inertia_:.0f}, Silhouette={sil:.4f}")

    # Pick optimal K: highest silhouette score
    optimal_k = list(K_range)[np.argmax(silhouette_scores)]
    print(f"\n  Optimal K (by silhouette): {optimal_k}")

    # Plot elbow + silhouette
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(list(K_range), inertias, 'b-o')
    axes[0].axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal K={optimal_k}')
    axes[0].set_title('Elbow Method', fontweight='bold')
    axes[0].set_xlabel('Number of Clusters (K)')
    axes[0].set_ylabel('Inertia')
    axes[0].legend()

    axes[1].plot(list(K_range), silhouette_scores, 'g-o')
    axes[1].axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal K={optimal_k}')
    axes[1].set_title('Silhouette Score', fontweight='bold')
    axes[1].set_xlabel('Number of Clusters (K)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/08_kmeans_elbow_silhouette.png', dpi=150)
    plt.close()
    print("  Saved: 08_kmeans_elbow_silhouette.png")

    # Fit final model
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans_final.fit_predict(X_scaled)

    db_score = davies_bouldin_score(X_scaled, cluster_labels)
    sil_score = silhouette_score(X_scaled, cluster_labels, sample_size=min(10000, len(X_scaled)))
    print(f"  Final K-Means: Silhouette={sil_score:.4f}, Davies-Bouldin={db_score:.4f}")

    return cluster_labels, optimal_k, {'silhouette': sil_score, 'davies_bouldin': db_score}


def run_dbscan(df, X_scaled):
    """DBSCAN density-based clustering."""
    ensure_plot_dir()
    print("\n--- DBSCAN Clustering ---")

    # Use a sample for parameter tuning if dataset is large
    n = min(20000, len(X_scaled))
    sample_idx = np.random.RandomState(42).choice(len(X_scaled), n, replace=False)
    X_sample = X_scaled[sample_idx]

    # Try different eps values
    best_sil = -1
    best_eps = 0.5
    best_labels = None

    for eps in [0.3, 0.5, 0.7, 1.0, 1.5]:
        db = DBSCAN(eps=eps, min_samples=10)
        labels = db.fit_predict(X_sample)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()

        if n_clusters >= 2:
            mask = labels != -1
            if mask.sum() > 10:
                sil = silhouette_score(X_sample[mask], labels[mask],
                                       sample_size=min(5000, mask.sum()))
                if sil > best_sil:
                    best_sil = sil
                    best_eps = eps
                    best_labels = labels
                print(f"  eps={eps}: clusters={n_clusters}, noise={n_noise}, silhouette={sil:.4f}")
            else:
                print(f"  eps={eps}: clusters={n_clusters}, noise={n_noise} (too few non-noise)")
        else:
            print(f"  eps={eps}: clusters={n_clusters}, noise={n_noise} (insufficient clusters)")

    # Apply best to full dataset
    dbscan_final = DBSCAN(eps=best_eps, min_samples=10)
    full_labels = dbscan_final.fit_predict(X_scaled)
    n_clusters = len(set(full_labels)) - (1 if -1 in full_labels else 0)
    n_noise = (full_labels == -1).sum()

    print(f"\n  Best DBSCAN (eps={best_eps}): {n_clusters} clusters, {n_noise} noise points")
    return full_labels, {'n_clusters': n_clusters, 'n_noise': n_noise, 'best_eps': best_eps}


def run_hierarchical(df, X_scaled, feature_names):
    """Agglomerative / Hierarchical Clustering on regional aggregates."""
    ensure_plot_dir()
    print("\n--- Hierarchical Clustering ---")

    # Aggregate to the Macro_Region level for dendrogram readability
    agg_cols = [c for c in feature_names if c in df.columns]
    if 'Macro_Region' not in df.columns or len(agg_cols) == 0:
        print("  Skipped hierarchical clustering (missing Macro_Region or features)")
        return None, {}

    regional_agg = df.groupby('Macro_Region')[agg_cols].mean().dropna()

    if len(regional_agg) < 3:
        print(f"  Skipped: only {len(regional_agg)} regions (need >= 3)")
        return None, {}

    scaler = StandardScaler()
    X_regional = scaler.fit_transform(regional_agg)

    # Dendrogram
    linkage_matrix = linkage(X_regional, method='ward')

    fig, ax = plt.subplots(figsize=(14, 7))
    dendrogram(linkage_matrix, labels=regional_agg.index.tolist(), ax=ax,
               leaf_rotation=45, leaf_font_size=9)
    ax.set_title('Regional Hierarchical Clustering (Ward Linkage)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Distance')
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/09_hierarchical_dendrogram.png', dpi=150)
    plt.close()
    print("  Saved: 09_hierarchical_dendrogram.png")

    # Agglomerative clustering (4 macro-clusters)
    n_clusters = min(4, len(regional_agg) - 1)
    agg_model = AgglomerativeClustering(n_clusters=n_clusters)
    regional_agg['Hierarchical_Cluster'] = agg_model.fit_predict(X_regional)

    print(f"  Hierarchical clusters ({n_clusters}):")
    for c in range(n_clusters):
        members = regional_agg[regional_agg['Hierarchical_Cluster'] == c].index.tolist()
        print(f"    Cluster {c}: {members}")

    return regional_agg, {'n_clusters': n_clusters}


def plot_cluster_scatter(df, cluster_col, title_suffix=''):
    """Scatter plot of clusters using lat/lon if available."""
    ensure_plot_dir()
    if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
        return
    if cluster_col not in df.columns:
        return

    sample = df.sample(min(15000, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(12, 7))
    scatter = ax.scatter(sample['Longitude'], sample['Latitude'],
                         c=sample[cluster_col], cmap='tab10', s=3, alpha=0.5)
    ax.set_title(f'Geographic Clusters {title_suffix}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.colorbar(scatter, ax=ax, label='Cluster')
    plt.tight_layout()
    fname = f"10_cluster_scatter_{cluster_col.lower().replace(' ', '_')}.png"
    plt.savefig(f'{PLOT_DIR}/{fname}', dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


def run_clustering(df):
    """Run all clustering methods and return enriched dataframe."""
    print("\n" + "=" * 70)
    print("GEOGRAPHIC CLUSTERING ANALYSIS")
    print("=" * 70)

    X_raw, X_scaled, feature_names, scaler = prepare_clustering_features(df)

    # Align df to rows that survived NaN drop
    df_clean = df.loc[X_raw.index].copy()

    # K-Means
    kmeans_labels, optimal_k, kmeans_metrics = run_kmeans(df_clean, X_scaled, feature_names)
    df_clean['KMeans_Cluster'] = kmeans_labels

    # Cluster profiles
    print("\n  K-Means Cluster Profiles:")
    profile_cols = ['Sales', 'Late_delivery_risk', 'Days for shipping (real)',
                    'Shipping_Distance_KM']
    profile_cols = [c for c in profile_cols if c in df_clean.columns]
    if profile_cols:
        profiles = df_clean.groupby('KMeans_Cluster')[profile_cols].mean()
        profiles['Count'] = df_clean.groupby('KMeans_Cluster').size()
        print(profiles.to_string())

    # DBSCAN
    dbscan_labels, dbscan_metrics = run_dbscan(df_clean, X_scaled)
    df_clean['DBSCAN_Cluster'] = dbscan_labels

    # Hierarchical
    regional_hier, hier_metrics = run_hierarchical(df_clean, X_scaled, feature_names)

    # Visualizations
    plot_cluster_scatter(df_clean, 'KMeans_Cluster', '(K-Means)')

    # Merge cluster labels back to full df
    df['KMeans_Cluster'] = np.nan
    df.loc[df_clean.index, 'KMeans_Cluster'] = df_clean['KMeans_Cluster']
    df['KMeans_Cluster'] = df['KMeans_Cluster'].fillna(0).astype(int)
    
    # Phase 6 compatibility
    df['cluster_label'] = df['KMeans_Cluster']

    df['DBSCAN_Cluster'] = np.nan
    df.loc[df_clean.index, 'DBSCAN_Cluster'] = df_clean['DBSCAN_Cluster']
    df['DBSCAN_Cluster'] = df['DBSCAN_Cluster'].fillna(-1).astype(int)

    all_metrics = {
        'kmeans': kmeans_metrics,
        'dbscan': dbscan_metrics,
        'hierarchical': hier_metrics
    }

    print("\n" + "=" * 70)
    print(f"Clustering complete. Columns added: KMeans_Cluster, DBSCAN_Cluster")
    print("=" * 70)
    return df, all_metrics


if __name__ == '__main__':
    from data_preprocessing import prepare_full_dataset
    df = prepare_full_dataset()
    df, metrics = run_clustering(df)
    print(metrics)
