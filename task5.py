import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from itertools import product

#load data
data = np.loadtxt('complex9_gn8.txt', delimiter=',')
X = data[:, :2]
y_true = data[:, 2].astype(int)

# Activity 1: Implement the Purity measure
def purity(a, b, outliers=False):
    if -1 in a:
        outlier_mask = a == -1
        a_clean = a[~outlier_mask]
        b_clean = b[~outlier_mask]
        outlier_perc = np.sum(outlier_mask) / len(a)
    else:
        a_clean = a
        b_clean = b
        outlier_perc = 0.0
    
    if len(a_clean) == 0:
        return (0.0, outlier_perc) if outliers else 0.0
    
    clusters = np.unique(a_clean)
    total_majority = 0
    
    for cluster in clusters:
        if cluster == -1:
            continue
        cluster_mask = a_clean == cluster
        cluster_labels = b_clean[cluster_mask]
        unique, counts = np.unique(cluster_labels, return_counts=True)
        total_majority += np.max(counts)
    
    purity_val = total_majority / len(a_clean)
    return (purity_val, outlier_perc) if outliers else purity_val

# Activity 2: Clustering with DBSCAN
def find_best_dbscan(X, y_true, max_evaluations=5000):
    best_purity = 0
    best_params = None
    best_labels = None
    evaluations = 0
    
    #parameter ranges
    eps_values = np.linspace(0.1, 15.0, 80) 
    min_samples_values = range(2, 30)
    for eps, min_samples in product(eps_values, min_samples_values):
        if evaluations >= max_evaluations:
            break
            
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        if 3 <= n_clusters <= 13:
            purity_val, outlier_perc = purity(labels, y_true, outliers=True)
            if outlier_perc <= 0.15 and purity_val > best_purity:
                best_purity = purity_val
                best_params = (eps, min_samples, n_clusters, outlier_perc)
                best_labels = labels
                print(f"New best: eps={eps:.3f}, min_samples={min_samples}, clusters={n_clusters}, outliers={outlier_perc:.3f}, purity={purity_val:.3f}")
                
        evaluations += 1
    
    return best_params, best_labels

# Activity 3: Clustering with K-means
def kmeans_analysis(X, y_true, k_range=range(2, 14)):
    k_values = list(k_range)
    sse_values = []
    purity_values = []
    cluster_centers_list = []
    labels_list = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=20) 
        labels = kmeans.fit_predict(X)
        
        sse_values.append(kmeans.inertia_)
        purity_values.append(purity(labels, y_true))
        cluster_centers_list.append(kmeans.cluster_centers_)
        labels_list.append(labels)
    
    return k_values, sse_values, purity_values, cluster_centers_list, labels_list

def main():
    #load data
    data = np.loadtxt('complex9_gn8.txt', delimiter=',')
    X = data[:, :2]
    y_true = data[:, 2].astype(int)
    
    # Activity 2: DBSCAN
    best_dbscan_params, best_dbscan_labels = find_best_dbscan(X, y_true)
    
    # Activity 3: K-means
    k_values, sse_values, purity_values, cluster_centers_list, labels_list = kmeans_analysis(X, y_true)
    
    # Find best k
    best_k = 9 
    best_k_idx = k_values.index(best_k)
    
    #generate plots and print results
    generate_plots_and_output(X, y_true, best_dbscan_params, best_dbscan_labels, 
                            k_values, sse_values, purity_values, cluster_centers_list, 
                            labels_list, best_k, best_k_idx)

def generate_plots_and_output(X, y_true, best_dbscan_params, best_dbscan_labels,
                           k_values, sse_values, purity_values, cluster_centers_list,
                           labels_list, best_k, best_k_idx):
    
    # DBSCAN plot
    if best_dbscan_params:
        eps, min_samples, n_clusters, outlier_perc = best_dbscan_params
        dbscan_purity = purity(best_dbscan_labels, y_true)
        
        plt.figure(figsize=(8, 6))
        outlier_mask = best_dbscan_labels == -1
        plt.scatter(X[outlier_mask, 0], X[outlier_mask, 1], c='black', marker='x', alpha=0.5, label='Outliers')
        plt.scatter(X[~outlier_mask, 0], X[~outlier_mask, 1], c=best_dbscan_labels[~outlier_mask], cmap='tab10')
        plt.title(f'Best DBSCAN: eps={eps:.2f}, min_samples={min_samples}\nClusters: {n_clusters}, Purity: {dbscan_purity:.3f}')
        plt.colorbar()
        plt.savefig('figure1_dbscan.png', dpi=300, bbox_inches='tight')
        plt.show()
    else:
        dbscan_purity = 0
        print("No valid DBSCAN parameters found")
    
    # K-means plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(k_values, sse_values, 'bo-')
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('SSE')
    ax1.set_title('SSE vs Number of Clusters')
    ax1.grid(True)

    ax2.plot(k_values, purity_values, 'ro-')
    ax2.set_xlabel('Number of clusters (k)')
    ax2.set_ylabel('Purity')
    ax2.set_title('Purity vs Number of Clusters')
    ax2.grid(True)
    plt.savefig('figure2_elbow_purity.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Best K-means clustering
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels_list[best_k_idx], cmap='tab10')
    plt.scatter(cluster_centers_list[best_k_idx][:, 0], cluster_centers_list[best_k_idx][:, 1], 
            marker='x', s=200, linewidths=3, color='black', label='Centroids')
    cbar = plt.colorbar(scatter, ticks=range(best_k))
    cbar.set_ticklabels(range(best_k))

    plt.title(f'Best K-means: k={best_k}, Purity: {purity_values[best_k_idx]:.3f}')
    plt.legend()
    plt.savefig('figure3_kmeans.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print results
    print("Activity 2 - Best DBSCAN Results:")
    if best_dbscan_params:
        print(f"eps: {best_dbscan_params[0]:.3f}, min_samples: {best_dbscan_params[1]}")
        print(f"Clusters: {best_dbscan_params[2]}, Outliers: {best_dbscan_params[3]:.3f}")
        print(f"Purity: {dbscan_purity:.3f}")

    print("\nActivity 3 - Best K-means Results:")
    print(f"Selected k: {best_k}")
    print(f"SSE: {sse_values[best_k_idx]:.3f}")
    print(f"Purity: {purity_values[best_k_idx]:.3f}")
    print("Cluster centroids:")
    for i, center in enumerate(cluster_centers_list[best_k_idx]):
        print(f"  Cluster {i}: ({center[0]:.3f}, {center[1]:.3f})")

    print("\nActivity 3c - Algorithm Comparison:")
    print(f"DBSCAN Purity: {dbscan_purity:.3f}")
    print(f"K-means Purity: {purity_values[best_k_idx]:.3f}")

if __name__ == "__main__":
    main()