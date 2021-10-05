from BKM import *


class CBKM:

    def __init__(self, n_clusters, bin_threshold):
        self.data = None
        self.k = n_clusters
        self.bin_threshold = bin_threshold
        self.cluster_centers_ = []
        self.labels_ = []

    def fit(self, data):
        self.data = data
        mode = False
        final_subclusters = []

        for l in range(2, self.k+1):
            print('\n(l = ' + str(l) + ')')
            # PHASE 1: GLOBAL CLUSTERING
            s_cooperative = self.global_clustering(l, final_subclusters, mode)
            # PHASE 2: COOPERATIVE CLUSTERING
            subclusters = self.cooperative_clustering(s_cooperative, l)
            # PHASE 3: MERGING
            final_subclusters = self.merging(subclusters, l)
            mode = True

        self.labels_ = clusters_to_labels(final_subclusters, self.data)
        self.cluster_centers_ = recalculate_centroids(final_subclusters)
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

    def get_params(self):
        return {'n_clusters': self.k, 'bin_threshold': self.bin_threshold}

    def set_params(self, dict_params):
        items = dict_params.items()
        self.k = items[0][1]
        self.bin_threshold = items[1][1]
        return self

    # Phase 1: Global clustering
    def global_clustering(self, l, final_subclusters = None, mode = False):
        # K-Means
        kmeans = KMeans(n_clusters=l, n_init = 1).fit(self.data)
        km_clusters = labels_to_clusters(self.data, kmeans.labels_)

        # Bisecting K-Means
        bkmeans = BKM(n_clusters=l, mode=mode)
        bkmeans = bkmeans.fit(self.data, final_subclusters)
        bkm_clusters = labels_to_clusters(self.data, bkmeans.labels_)

        return km_clusters, bkm_clusters

    # Phase 2: Cooperative Clustering
    def cooperative_clustering(self, s_coop, l):
        km_clusters, bkm_clusters = s_coop
        subclusters = []

        # Generating Cooperative Contingency Matrix (CCM)
        # Obtaining disjoint sub-clusters from CCM
        contingency_matrix = []
        for idx, km_c in enumerate(km_clusters):
            bkm_list = []
            for idy, bkm_c in enumerate(bkm_clusters):
                intersection = 0
                subclusters_local = []
                for km_sample in km_c:
                    for bkm_sample in bkm_c:
                        if (km_sample == bkm_sample).all():
                            intersection += 1
                            subclusters_local.append(km_sample)
                            break
                bkm_list.append(intersection)
                if subclusters_local:
                    subclusters.append(subclusters_local)
            contingency_matrix.append(bkm_list)

        # Plot CCM Matrix
        ccm_matrix_numpy = np.array(contingency_matrix)
        rowcolnames = ['S'+str(i) for i in range(l)]
        df = pd.DataFrame(ccm_matrix_numpy, index=rowcolnames, columns=rowcolnames)
        print("Cooperative Contingency Matrix (CCM)")
        print(df)

        return subclusters

    # Phasse 3: Merging
    def merging(self, subclusters, l):
        # STEP 1: For each subcluster, build similarity histogram
        histograms_subclusters = []
        for id, sc in enumerate(subclusters):
            histograms_subclusters.append(self.similarity_histogram(sc, ids=id))

        # STEP 2: Using the subclusters histograms, a merging cohesiveness factor of each pair of subclusters
        # is computed. This factor represents the coherency (quality) of merging two subclusters into
        # a new coherent cluster.

        # This factor is calculated by the coherency of merging the corresponding histograms
        # The process of merging two subclusters reveals a new histogram; this histogram is constructed by adding the
        # corresponding counts of each bin from the two merged histogram and also by adding the additional pair-wise
        # similarities that are generated as a result of merging the two subclusters together
        merging_matrix = []
        for idx, sx in enumerate(subclusters):
            cmm_list = []
            for idy, sy in enumerate(subclusters):
                if idx == idy:
                    cmm_list.append(-1)
                    continue
                new_hist = self.merge_histograms(histograms_subclusters[idx], histograms_subclusters[idy], sx, sy)
                mcf = self.compute_mcf(sx, sy, new_hist)
                cmm_list.append(mcf)
            merging_matrix.append(cmm_list)

        # Plot CMM Matrix
        cmm_matrix_numpy = np.array(merging_matrix)
        rowcolnames = ['Sb' + str(i) for i in range(len(subclusters))]
        df = pd.DataFrame(cmm_matrix_numpy, index=rowcolnames, columns=rowcolnames)
        print("\nCooperative Merging Matrix (CMM)")
        print(df)

        # STEP 3: Merge two sub-clusters with the highest mcf in CMM; reduce the number of subclusters and update CMM
        # Until len(subclusters) == l
        subclusters_ids = [str(i) for i in range(len(subclusters))]
        while len(subclusters) != l:
            # Get indices of the pair of subclusters with maximum mcf
            row, col = np.unravel_index(np.argmax(cmm_matrix_numpy), cmm_matrix_numpy.shape)
            #rows, cols = np.unravel_index(np.argsort(-cmm_matrix_numpy, axis = None)[:2], cmm_matrix_numpy.shape)

            # Merge resulting subclusters and updating structures
            merged_cluster = subclusters[row] + subclusters[col]
            subclusters.pop(row)
            histograms_subclusters.pop(row)
            cmm_matrix_numpy = np.delete(cmm_matrix_numpy, row, axis=0)
            cmm_matrix_numpy = np.delete(cmm_matrix_numpy, row, axis=1)
            pop_row = subclusters_ids.pop(row)
            if row < col:
                col_used = col-1
            else:
                col_used = col
            subclusters.pop(col_used)
            histograms_subclusters.pop(col_used)
            cmm_matrix_numpy = np.delete(cmm_matrix_numpy, col_used, axis=0)
            cmm_matrix_numpy = np.delete(cmm_matrix_numpy, col_used, axis=1)
            pop_col = subclusters_ids.pop(col_used)
            subclusters.append(merged_cluster)
            histograms_subclusters.append(self.similarity_histogram(merged_cluster, ids=str(pop_row)+'+'+str(pop_col),color='purple'))
            subclusters_ids.append(str(pop_row)+'+'+str(pop_col))
            print("\nSub-clusters merged: Sb" + str(pop_row)+ " and Sb" + str(pop_col))

            # Compute mcf of new merged subcluster with the rest of subclusters
            cmm_list = []
            idx = len(subclusters) - 1
            for idy, sy in enumerate(subclusters):
                if idx == idy:
                    cmm_list.append(-1)
                    continue
                new_hist = self.merge_histograms(histograms_subclusters[idx],
                                                 histograms_subclusters[idy], subclusters[idx], sy)
                mcf = self.compute_mcf(subclusters[idx], sy, new_hist)
                cmm_list.append(mcf)

            # Updating CMM
            cmm_matrix_numpy = np.append(cmm_matrix_numpy, [cmm_list[:-1]], axis=0)
            cmm_matrix_numpy = np.insert(cmm_matrix_numpy, len(cmm_list[:-1]), cmm_list, axis=1)
            rowcolnames = ['Sb' + str(i) for i in subclusters_ids]
            df = pd.DataFrame(cmm_matrix_numpy, index=rowcolnames, columns=rowcolnames)
            print("Cooperative Merging Matrix (CMM)")
            print(df)

        print("\nMerge phase finished with l={} sub-clusters".format(len(subclusters)))
        return subclusters

    # Compute similarity histogram of a subcluster
    def similarity_histogram(self, sc, ids, nbins=21, color='red'):
        histogram = np.zeros(nbins, dtype=int)
        binsize = 2/nbins

        for x in sc:
            for y in sc:
                if (x == y).all():
                    continue
                # Calculate cosine similarity between samples
                sim = cosine_similarity(x.reshape(1, -1), y.reshape(1, -1))[0][0]
                # Decide the binid of the histogram in which the similarity falls
                if sim == -1:
                    binid = 0
                elif sim == 1:
                    binid = nbins-1
                else:
                    binid = -1 + int(sim/binsize) + (nbins/2)
                # Increment histogram[binid]
                histogram[int(binid)] += 1

        # Plot histogram
        # COMMENT THE FOLLOWING LINES IF YOU DON'T WANT THE HISTOGRAMS TO BE PLOTTED
        plt.figure(figsize = (9,5))
        plt.bar(np.arange(-1.0, 1.1, 0.1), histogram, width=0.1, align='edge', color=color)
        xmin, xmax = plt.xlim()
        plt.xlim(xmin * 0.8, xmax * 0.8)
        plt.xticks(np.arange(-1.0, 1.1, 0.1))
        plt.ylabel('Number of pair-wise similarities')
        plt.title("Similarity Histogram S"+str(ids))
        plt.show()

        return histogram

    # Create new merging histogram
    def merge_histograms(self, histx, histy, sx, sy):
        additional_similarities_histogram = self.get_additional_similarities_histogram(sx, sy)
        new_hist = [(histx[i] + histy[i] + additional_similarities_histogram[i]) for i in range(len(histx))]
        return new_hist

    # Computing merging cohesiveness factor
    def compute_mcf(self, sx, sy, new_hist):
        binsize = 2 / len(new_hist)
        bin_threshold = self.bin_threshold

        nsim = (len(sx) + len(sy)) * (len(sx) + len(sy) - 1) / 2
        num_mcf = sum([(((binid * binsize) - 1 + (binsize / 2)) * new_hist[binid])
                       for binid in range(bin_threshold, len(new_hist))])

        # An mcf between two sub-clusters is computed by calculating the ration of the count of similarities weighted
        # by the bin similarity above aa certain similarity threshold to the total count of similarities in the new
        # merged histogram. The higher this ration, the more cohesive the new generated cluster
        mcf = num_mcf / nsim
        return mcf

    # Compute histogram of the additional similarities resulting from the merging of two subclusters
    def get_additional_similarities_histogram(self, sx, sy, nbins=21):
        histogram = np.zeros(nbins, dtype=int)
        binsize = 2/nbins

        # Compute new similarities and its histogram
        for x in sx:
            for y in sy:
                # Calculate cosine similarity between samples
                sim = cosine_similarity(x.reshape(1, -1), y.reshape(1, -1))[0][0]
                # Decide the binid of the histogram in which the similarity falls
                if sim == -1:
                    binid = 0
                elif sim == 1:
                    binid = nbins-1
                else:
                    binid = -1 + int(sim/binsize) + (nbins/2)
                # Increment histogram[binid]
                histogram[int(binid)] += 1

        return histogram







