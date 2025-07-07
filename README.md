# Machine Learning Clustering Dashboard

A comprehensive Streamlit application that demonstrates various clustering algorithms using scikit-learn, HDBSCAN, and other machine learning libraries.

## Features

- **Multiple Clustering Algorithms**: K-Means, DBSCAN, and HDBSCAN
- **Interactive Data Generation**: Blobs, Moons, Circles, and Random datasets
- **Real-time Parameter Tuning**: Adjust algorithm parameters through the sidebar
- **Visual Analytics**: Interactive plots showing original data and clustering results
- **Performance Metrics**: Silhouette score, inertia, and cluster statistics
- **Responsive Design**: Wide layout with organized sections

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

3. Use the sidebar to:
   - Select different dataset types
   - Adjust the number of samples
   - Choose clustering algorithms
   - Tune algorithm parameters

## Available Algorithms

### K-Means
- Partitions data into k clusters by minimizing within-cluster variance
- Works well with spherical clusters
- Requires specifying the number of clusters beforehand

### DBSCAN
- Density-based clustering that groups together points in high-density regions
- Can find clusters of arbitrary shapes
- Automatically identifies noise points
- Requires tuning epsilon and min_samples parameters

### HDBSCAN
- Hierarchical version of DBSCAN
- More robust to parameter selection
- Can find clusters of varying densities
- Automatically determines the optimal number of clusters

## Dataset Types

- **Blobs**: Gaussian clusters with controlled variance
- **Moons**: Two interleaving half circles
- **Circles**: Concentric circles pattern
- **Random**: Randomly distributed points

## Libraries Used

- **Streamlit**: Web application framework
- **scikit-learn**: Machine learning algorithms (K-Means, DBSCAN)
- **HDBSCAN**: Hierarchical density-based clustering
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **Plotly**: Interactive visualizations
- **Matplotlib**: Additional plotting capabilities
- **SciPy**: Scientific computing

## Screenshots

The app provides:
- Side-by-side comparison of original data vs clustering results
- Real-time parameter adjustment
- Performance metrics dashboard
- Detailed cluster statistics
- Feature distribution analysis

Enjoy exploring different clustering algorithms and datasets! 