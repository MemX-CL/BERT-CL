import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import distance


def sample_by_cluster(X_pca, k, n_samples):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit_predict(X_pca)
    
    centers = kmeans.cluster_centers_
    
    sample_indexes = []
    
    for c in centers:
        dist = [distance.euclidean(c, point) for point in X_pca]
        nearest_indices = np.argsort(dist)[:n_samples]
        
        sample_indexes = np.concatenate((sample_indexes, nearest_indices))
        
    return sample_indexes