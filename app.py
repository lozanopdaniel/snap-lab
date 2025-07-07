import streamlit as st
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import silhouette_score
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="ML Clustering Dashboard",
    page_icon="🔬",
    layout="wide"
)

# Title and description
st.title("🔬 Machine Learning Clustering Dashboard")
st.markdown("Explore different clustering algorithms using scikit-learn, HDBSCAN, and more!")

# Sidebar for controls
st.sidebar.header("Settings")

# Dataset selection
dataset_type = st.sidebar.selectbox(
    "Choose Dataset",
    ["Blobs", "Moons", "Circles", "Random"]
)

# Number of samples
n_samples = st.sidebar.slider("Number of samples", 100, 1000, 300)

# Algorithm selection
algorithm = st.sidebar.selectbox(
    "Clustering Algorithm",
    ["K-Means", "DBSCAN", "HDBSCAN"]
)

# Generate data based on selection
@st.cache_data
def generate_data(dataset_type, n_samples):
    if dataset_type == "Blobs":
        X, y = make_blobs(n_samples=n_samples, centers=4, cluster_std=0.60, random_state=42)
    elif dataset_type == "Moons":
        X, y = make_moons(n_samples=n_samples, noise=0.1, random_state=42)
    elif dataset_type == "Circles":
        X, y = make_blobs(n_samples=n_samples, centers=[[0, 0], [0, 0]], cluster_std=[0.2, 0.5], random_state=42)
        # Create circular pattern
        X = X * np.array([1, 1]) + np.array([0, 0])
        y = (np.linalg.norm(X, axis=1) > 0.3).astype(int)
    else:  # Random
        X = np.random.randn(n_samples, 2)
        y = np.zeros(n_samples)
    
    return X, y

# Generate the data
X, y_true = generate_data(dataset_type, n_samples)

# Create two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Original Data")
    
    # Plot original data
    fig_original = px.scatter(
        x=X[:, 0], 
        y=X[:, 1],
        title=f"Original {dataset_type} Dataset ({n_samples} samples)",
        labels={'x': 'Feature 1', 'y': 'Feature 2'}
    )
    fig_original.update_layout(height=400)
    st.plotly_chart(fig_original, use_container_width=True)

# Clustering parameters
with st.sidebar:
    st.subheader("Algorithm Parameters")
    
    if algorithm == "K-Means":
        n_clusters = st.slider("Number of clusters", 2, 10, 4)
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    elif algorithm == "DBSCAN":
        eps = st.slider("Epsilon", 0.1, 2.0, 0.5, 0.1)
        min_samples = st.slider("Min samples", 2, 20, 5)
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
    else:  # HDBSCAN
        min_cluster_size = st.slider("Min cluster size", 5, 50, 10)
        min_samples = st.slider("Min samples", 1, 20, 5)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)

# Perform clustering
y_pred = clusterer.fit_predict(X)

# Calculate metrics
if algorithm == "K-Means":
    inertia = clusterer.inertia_
    silhouette = silhouette_score(X, y_pred) if len(np.unique(y_pred)) > 1 else 0
else:
    inertia = None
    silhouette = silhouette_score(X, y_pred) if len(np.unique(y_pred)) > 1 else 0

with col2:
    st.subheader(f"🎯 {algorithm} Clustering Results")
    
    # Create color map for clusters
    unique_labels = np.unique(y_pred)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    # Plot clustered data
    fig_clustered = go.Figure()
    
    for i, label in enumerate(unique_labels):
        mask = y_pred == label
        if label == -1:  # Noise points for DBSCAN/HDBSCAN
            fig_clustered.add_trace(go.Scatter(
                x=X[mask, 0], y=X[mask, 1],
                mode='markers',
                marker=dict(color='black', size=8, symbol='x'),
                name=f'Noise (Cluster {label})',
                showlegend=True
            ))
        else:
            fig_clustered.add_trace(go.Scatter(
                x=X[mask, 0], y=X[mask, 1],
                mode='markers',
                marker=dict(color=colors[i], size=8),
                name=f'Cluster {label}',
                showlegend=True
            ))
    
    fig_clustered.update_layout(
        title=f"{algorithm} Clustering",
        xaxis_title="Feature 1",
        yaxis_title="Feature 2",
        height=400
    )
    st.plotly_chart(fig_clustered, use_container_width=True)

# Metrics section
st.subheader("📈 Clustering Metrics")

metric_col1, metric_col2, metric_col3 = st.columns(3)

with metric_col1:
    st.metric("Number of Clusters", len(unique_labels) - (1 if -1 in unique_labels else 0))
    
with metric_col2:
    if silhouette is not None:
        st.metric("Silhouette Score", f"{silhouette:.3f}")
    else:
        st.metric("Silhouette Score", "N/A")
        
with metric_col3:
    if inertia is not None:
        st.metric("Inertia", f"{inertia:.2f}")
    else:
        st.metric("Inertia", "N/A")

# Additional analysis
st.subheader("🔍 Detailed Analysis")

analysis_col1, analysis_col2 = st.columns(2)

with analysis_col1:
    st.subheader("Cluster Statistics")
    
    # Calculate cluster statistics
    cluster_stats = []
    for label in unique_labels:
        if label != -1:  # Skip noise points
            mask = y_pred == label
            cluster_size = np.sum(mask)
            cluster_mean = np.mean(X[mask], axis=0)
            cluster_std = np.std(X[mask], axis=0)
            
            cluster_stats.append({
                'Cluster': label,
                'Size': cluster_size,
                'Mean_X': cluster_mean[0],
                'Mean_Y': cluster_mean[1],
                'Std_X': cluster_std[0],
                'Std_Y': cluster_std[1]
            })
    
    if cluster_stats:
        stats_df = pd.DataFrame(cluster_stats)
        st.dataframe(stats_df, use_container_width=True)
    else:
        st.info("No valid clusters found")

with analysis_col2:
    st.subheader("Feature Distribution")
    
    # Create feature distribution plot
    fig_dist = go.Figure()
    
    for i, label in enumerate(unique_labels):
        if label != -1:  # Skip noise points
            mask = y_pred == label
            fig_dist.add_trace(go.Histogram(
                x=X[mask, 0],
                name=f'Cluster {label} - Feature 1',
                opacity=0.7,
                marker_color=colors[i]
            ))
    
    fig_dist.update_layout(
        title="Feature 1 Distribution by Cluster",
        xaxis_title="Feature 1",
        yaxis_title="Count",
        barmode='overlay'
    )
    st.plotly_chart(fig_dist, use_container_width=True)

# Information section
st.subheader("ℹ️ About the Algorithms")

with st.expander("Learn more about the clustering algorithms"):
    st.markdown("""
    **K-Means**: 
    - Partitions data into k clusters by minimizing within-cluster variance
    - Works well with spherical clusters
    - Requires specifying the number of clusters beforehand
    
    **DBSCAN**: 
    - Density-based clustering that groups together points in high-density regions
    - Can find clusters of arbitrary shapes
    - Automatically identifies noise points
    - Requires tuning epsilon and min_samples parameters
    
    **HDBSCAN**: 
    - Hierarchical version of DBSCAN
    - More robust to parameter selection
    - Can find clusters of varying densities
    - Automatically determines the optimal number of clusters
    """)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit, scikit-learn, HDBSCAN, and Plotly") 