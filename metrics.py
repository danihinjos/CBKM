from utils import *


# Compute Entropy
def compute_entropy(clusters, datax, datay):
    total_entropy = 0
    for c in clusters:
        # Labels
        cluster_labels = get_cluster_labels(c, datax, datay)
        # Probabilities
        probs = [cluster_labels.count(i)/len(c) for i in set(cluster_labels)]
        total_entropy += ((len(c)/len(datax)) * entropy(probs, base=2))
    return total_entropy / math.log(len(clusters), 2)


# Compute Recall
def recall(cluster, datax, datay, class_value):
    cluster_labels = get_cluster_labels(cluster, datax, datay)
    recall_num = cluster_labels.count(class_value)
    recall_den = list(datay).count(class_value)
    recall = recall_num / recall_den
    return recall


# Compute Precision
def precision(cluster, datax, datay, class_value):
    cluster_labels = get_cluster_labels(cluster, datax, datay)
    precision_num = cluster_labels.count(class_value)
    precision_den = len(cluster)
    precision = precision_num / precision_den
    return precision


# Compute F-Measure
def compute_fmeasure(clusters, datax, datay):
    f_measure, fm_num, fm_den = 0, 0, 0
    for class_value in set(datay):
        fm_class = max([((2 * precision(c, datax, datay, class_value) * recall(c, datax, datay, class_value)) /
                     (precision(c, datax, datay, class_value) + recall(c, datax, datay, class_value)))
                        for c in clusters
                        if (precision(c, datax, datay, class_value) + recall(c, datax, datay, class_value)) > 0])
        fm_num += (list(datay).count(class_value) * fm_class)
        fm_den += list(datay).count(class_value)
    f_measure = fm_num / fm_den
    return f_measure


# Compute Purity
def compute_purity(clusters, datax, datay):
    purity = 0
    for c in clusters:
        cluster_labels = get_cluster_labels(c, datax, datay)
        maxcount = max([cluster_labels.count(class_label) for class_label in set(datay)])
        purity += maxcount
    purity = purity / len(datax)
    return purity

'''
from sklearn import metrics
def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    print(contingency_matrix)
    print(np.amax(contingency_matrix, axis=0))
    print(np.sum(np.amax(contingency_matrix, axis=0)))
    print(np.sum(contingency_matrix))
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
'''


# Compute Separation Index
def compute_separation_index(clusters, datax):
    centroids = recalculate_centroids(clusters)
    si_num = sum([sum([(1 - cosine_similarity(x.reshape(1, -1), centroids[i].reshape(1, -1))[0][0])
               for x in clusters[i]]) for i in range(len(clusters))])

    si_den = []
    for idx, cx in enumerate(centroids):
        for idy, cy in enumerate(centroids):
            if idx == idy:
                continue
            si_den.append((1 - cosine_similarity(cx.reshape(1, -1), cy.reshape(1, -1))[0][0]))
    si_den = len(datax) * min(si_den)

    si = si_num / si_den
    return si


# Compute all metrics
def compute_metrics(clusters, datax, datay):
    pred = clusters_to_labels(clusters, datax)

    fscore = compute_fmeasure(clusters, datax, datay)
    entropy = compute_entropy(clusters, datax, datay)
    purity = compute_purity(clusters, datax, datay)
    nmi = normalized_mutual_info_score(datay, pred)
    si = compute_separation_index(clusters, datax)

    return fscore, entropy, purity, nmi, si


# Verify if null hypothesis is accepted or rejected
def rejected_hipothesis(t_value):
    if t_value > 2.024:
        return "H0: Rejected"
    else:
        return "H0: Accepted"