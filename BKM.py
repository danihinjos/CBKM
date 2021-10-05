# BISECTING K-MEANS ALGORITHM

# Imports
from utils import *


class BKM:

    def __init__(self, n_clusters, mode=False):
        self.k = n_clusters
        self.mode = mode
        self.labels_ = []
        self.cluster_centers_ = []

    def fit(self, data, subclusters=None):
        # Initialization of the initial general cluster that will be bisected
        # Verify it it's the first iteration (whole data) or not (cooperative subclusters)
        if self.mode == False:
            clusters = []  # array for clusters
            centroids = []  # array for centroids
            clusters.append(data)
            centroids.append(recalculate_centroids(clusters))
        else:
            clusters = [c for c in subclusters]
            centroids = recalculate_centroids(clusters)

        while len(clusters) < self.k:
            # Bisection criteria: cluster with highest MSE
            searchmse = []
            for i in range(len(clusters)):
                searchmse.append(self.msecluster(centroids[i], clusters[i]))
            max_index = np.argmax(searchmse)

            # Removal of cluster and its corresponding centroids from lists
            cluster = clusters.pop(max_index)
            centroids.pop(max_index)

            # Setting the error high so that a minimal error is found
            error = 10 ** 10
            for i in range(10):  # 10 is an arbitrary value
                km = KMeans(n_clusters=2, n_init=1).fit(cluster)
                if (- km.score(cluster)) < error:  # keeping all relevant data from the cluster partition with lowest error
                    error = - km.score(cluster)
                    best_centroids = km.cluster_centers_
                    best_labels = km.labels_

            best_clusters = labels_to_clusters(cluster, best_labels)

            for c in best_clusters:
                clusters.append(c)
            for c in best_centroids:
                centroids.append(c)

        self.labels_ = clusters_to_labels(clusters, data)
        self.cluster_centers_ = centroids
        return self

    def predict(self, data_test):
        prediction = []
        if not any(isinstance(l, np.ndarray) for l in data_test):
            data_test = [data_test]
        for new_sample in data_test:
            sample_distances = []
            for c in self.cluster_centers_:
                sample_distances.append(euclidean_distances(new_sample.reshape(1, -1), c.reshape(1, -1)))
            prediction.append(np.argmin(sample_distances))
        return prediction


    # Compute MSE of a specific cluster
    def msecluster(self, centroid, values):
        summation = 0  # variable to store the summation of differences
        for v in values:
            difference = distance.euclidean(centroid,v)
            summation += difference ** 2  # taking a sum of all the distances of a cluster
        mse = summation / len(values) # dividing it between the number of data points of that cluster
        return mse



