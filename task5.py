import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import seaborn as sns
from itertools import product

#load dataset
def load_data(filename):
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, :2] 
    y_true = data[:, 2].astype(int)
    return X, y_true

# Activity 1: Implement the Purity measure
def purity(a, b, outliers=False):
    # Handle outliers (cluster label -1 in Python)
    if -1 in a:
        outlier_mask = a == -1
        non_outlier_mask = ~outlier_mask
        a_clean = a[non_outlier_mask]
        b_clean = b[non_outlier_mask]
        outlier_percentage = np.sum(outlier_mask) / len(a)
    else:
        a_clean = a
        b_clean = b
        outlier_percentage = 0.0
    
    if len(a_clean) == 0:
        return (0.0, outlier_percentage) if outliers else 0.0
    
    # Get unique clusters
    clusters = np.unique(a_clean)
    
    total_majority = 0
    total_points = len(a_clean)
    
    for cluster in clusters:
        if cluster == -1:  # Skip outliers
            continue
            
        cluster_mask = a_clean == cluster
        if np.sum(cluster_mask) == 0:
            continue
            
        cluster_labels = b_clean[cluster_mask]
        
        # Find majority class in this cluster
        unique, counts = np.unique(cluster_labels, return_counts=True)
        majority_count = np.max(counts)
        total_majority += majority_count
    
    purity_value = total_majority / total_points if total_points > 0 else 0.0
    
    if outliers:
        return (purity_value, outlier_percentage)
    else:
        return purity_value

# Activity 2: DBSCAN parameter search
def find_best_dbscan(X, y_true, max_evaluations=5000):
    """
    Find best DBSCAN parameters that maximize purity with constraints:
    - 3 to 13 clusters
    - <= 15% outliers
    """
    best_purity = 0
    best_params = None
    best_labels = None
    evaluations = 0
    
    # Parameter ranges to search
    eps_values = np.linspace(0.5, 10.0, 50)
    min_samples_values = range(2, 20)
    
    for eps, min_samples in product(eps_values, min_samples_values):
        if evaluations >= max_evaluations:
            break
            
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        
        # Count clusters (excluding outliers)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        # Check constraints
        if 3 <= n_clusters <= 13:
            purity_val, outlier_perc = purity(labels, y_true, outliers=True)
            
            if outlier_perc <= 0.15 and purity_val > best_purity:
                best_purity = purity_val
                best_params = (eps, min_samples, n_clusters, outlier_perc)
                best_labels = labels
                
        evaluations += 1
    
    return best_params, best_labels, best_purity

# Activity 3: K-means clustering with elbow method
def kmeans_analysis(X, y_true, k_range=range(2, 14)):
    """
    Run K-means for different k values and compute SSE and purity
    """
    sse_values = []
    purity_values = []
    cluster_centers_list = []
    labels_list = []
    
    k_values = list(k_range)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        # Calculate SSE
        sse = kmeans.inertia_
        sse_values.append(sse)
        
        # Calculate purity
        purity_val = purity(labels, y_true)
        purity_values.append(purity_val)
        
        cluster_centers_list.append(kmeans.cluster_centers_)
        labels_list.append(labels)
    
    return k_values, sse_values, purity_values, cluster_centers_list, labels_list

# Visualization functions
def plot_clustering_results(X, labels, title, true_labels=None):
    plt.figure(figsize=(12, 5))
    
    if true_labels is not None:
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(X[:, 0], X[:, 1], c=true_labels, cmap='tab10', alpha=0.7)
        plt.title('True Classes')
        plt.colorbar(scatter)
    
    plt.subplot(1, 2, int(true_labels is not None) + 1)
    
    # Handle outliers (label -1)
    outlier_mask = labels == -1
    non_outlier_mask = ~outlier_mask
    
    if np.any(outlier_mask):
        plt.scatter(X[outlier_mask, 0], X[outlier_mask, 1], 
                   c='black', marker='x', alpha=0.5, label='Outliers')
    
    scatter = plt.scatter(X[non_outlier_mask, 0], X[non_outlier_mask, 1], 
                         c=labels[non_outlier_mask], cmap='tab10', alpha=0.7)
    plt.title(title)
    plt.colorbar(scatter)
    if np.any(outlier_mask):
        plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_elbow_curve(k_values, sse_values, purity_values):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # SSE plot
    ax1.plot(k_values, sse_values, 'bo-')
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('SSE')
    ax1.set_title('Elbow Method - SSE vs Number of Clusters')
    ax1.grid(True, alpha=0.3)
    
    # Purity plot
    ax2.plot(k_values, purity_values, 'ro-')
    ax2.set_xlabel('Number of clusters (k)')
    ax2.set_ylabel('Purity')
    ax2.set_title('Purity vs Number of Clusters')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Main execution
def main():
    # Load data
    X, y_true = load_data('complex9_gn8.txt')
    
    print("Dataset shape:", X.shape)
    print("Unique true classes:", np.unique(y_true))
    
    # Standardize the data (important for DBSCAN)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Activity 1: Test purity function
    print("\n" + "="*50)
    print("ACTIVITY 1: Purity Measure")
    print("="*50)
    
    # Test with some dummy data
    test_labels = np.array([0, 0, 1, 1, 0, 2, 2, 2])
    test_true = np.array([0, 0, 0, 1, 1, 2, 2, 2])
    test_purity = purity(test_labels, test_true)
    print(f"Test purity: {test_purity:.3f}")
    
    # Activity 2: DBSCAN clustering
    print("\n" + "="*50)
    print("ACTIVITY 2: DBSCAN Clustering")
    print("="*50)
    
    best_params, best_dbscan_labels, best_dbscan_purity = find_best_dbscan(X_scaled, y_true)
    
    if best_params:
        eps, min_samples, n_clusters, outlier_perc = best_params
        print(f"Best DBSCAN parameters:")
        print(f"  epsilon: {eps:.3f}")
        print(f"  min_samples: {min_samples}")
        print(f"  Number of clusters: {n_clusters}")
        print(f"  Outlier percentage: {outlier_perc:.3f}")
        print(f"  Purity: {best_dbscan_purity:.3f}")
        
        # Visualize best DBSCAN clustering
        plot_clustering_results(X, best_dbscan_labels, 
                              f'Best DBSCAN Clustering (Purity: {best_dbscan_purity:.3f})',
                              y_true)
    else:
        print("No valid DBSCAN parameters found within constraints")
    
    # Activity 3: K-means clustering
    print("\n" + "="*50)
    print("ACTIVITY 3: K-means Clustering")
    print("="*50)
    
    k_values, sse_values, purity_values, centers_list, labels_list = kmeans_analysis(X, y_true)
    
    # Plot elbow curve and purity
    plot_elbow_curve(k_values, sse_values, purity_values)
    
    # Find best k using elbow method (simple gradient-based approach)
    gradients = np.diff(sse_values)
    gradient_changes = np.diff(gradients)
    elbow_points = []
    
    for i in range(1, len(gradient_changes)-1):
        if gradient_changes[i] * gradient_changes[i-1] < 0:  # Sign change
            elbow_points.append(k_values[i+1])
    
    print("Potential elbow points (k values):", elbow_points)
    
    # Select best k (you can choose based on your observation)
    best_k = elbow_points[0] if elbow_points else 9  # Default to 9 if no clear elbow
    best_k_idx = k_values.index(best_k)
    
    print(f"\nSelected k: {best_k}")
    print(f"SSE for k={best_k}: {sse_values[best_k_idx]:.3f}")
    print(f"Purity for k={best_k}: {purity_values[best_k_idx]:.3f}")
    print(f"Cluster centroids for k={best_k}:")
    for i, center in enumerate(centers_list[best_k_idx]):
        print(f"  Cluster {i}: ({center[0]:.3f}, {center[1]:.3f})")
    
    # Visualize best K-means clustering
    best_kmeans_labels = labels_list[best_k_idx]
    plot_clustering_results(X, best_kmeans_labels,
                          f'Best K-means Clustering (k={best_k}, Purity: {purity_values[best_k_idx]:.3f})',
                          y_true)
    
    # Activity 4: Comparison
    print("\n" + "="*50)
    print("ACTIVITY 4: Algorithm Comparison")
    print("="*50)
    
    if best_params:
        dbscan_purity = best_dbscan_purity
    else:
        # If no valid DBSCAN found, run a default one for comparison
        dbscan = DBSCAN(eps=2.0, min_samples=5)
        dbscan_labels = dbscan.fit_predict(X_scaled)
        dbscan_purity = purity(dbscan_labels, y_true)
    
    kmeans_purity = purity_values[best_k_idx]
    
    print(f"DBSCAN Purity: {dbscan_purity:.3f}")
    print(f"K-means Purity: {kmeans_purity:.3f}")
    
    if dbscan_purity > kmeans_purity:
        print("DBSCAN performed better for this dataset")
    else:
        print("K-means performed better for this dataset")

if __name__ == "__main__":
    main()