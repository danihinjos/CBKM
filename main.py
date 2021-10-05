from CBKM import *
import time
from scipy import stats
from metrics import *
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs

desired_width = 320
pd.set_option('display.width', desired_width)

outputpath = "results/"
filepath = "datasets/"

# Hyperparameters tested
k = 4
bin_threshold = 12
ITER = 10


# User selection of dataset
dataset_dict = {'1':'DS1 (Artificial)', '2':'DS2 (Artificial)', '3':'Heart-C', '4':'Breast Cancer', '5':'CMC', '6':'Other datasets', '7':'Exit'}
selection = make_selection('Datasets', dataset_dict)
dataset_selected = dataset_dict[selection]

print('\nSelected ' + selection + ' - ' + str(dataset_selected) + '\n')

filepath_correct = True
while filepath_correct:
    artificial = False
    if selection == "1":
        dataset_name = "DS1"
        artificial = True
    if selection == "2":
        dataset_name = "DS2"
        artificial = True
    if selection == "3":
        dataset_name = "heart-c.arff"
    elif selection == "4":
        dataset_name = "breast-w.arff"
    elif selection == "5":
        dataset_name = "cmc.arff"
    elif selection == "6":
        dataset_name = input("Introduce the filename of your dataset"  + ': ')
    elif selection == "7":
        exit()

    if artificial == False:
        outputpath = outputpath + dataset_name.split('.')[0].upper() + "_RESULTS.txt"
        # Load chosen dataset
        flag, dataset = load_data(filepath+dataset_name)
        if flag == True:
            filepath_correct = False
        else:
            filepath_correct = True
    else:
        outputpath = outputpath + dataset_name + "_RESULTS.txt"
        if dataset_name == "DS1":
            X, Y = make_blobs(n_samples=811, centers=3, n_features=2)
        elif dataset_name == "DS2":
            X, Y = make_blobs(n_samples=2530, centers=6, n_features=2)
        filepath_correct = False


# Opening .txt files which will contain results
resultsfile = open(outputpath, "w+")

if artificial == False:
    # Preprocessing step
    dataset = preprocessing_step(dataset)

    # Data/Label split
    dx = dataset.drop(dataset.columns[-1], axis=1)
    dy = dataset[dataset.columns[-1]]
    X = dx.values
    Y = dy

# CBKM Algorithm
cbkm_fmeasures, cbkm_entropies, \
cbkm_purities, cbkm_nmis, cbkm_sis, cbkm_times = [], [], [], [], [], []
for i in range(ITER):
    print("\nExecuting CBKM....")
    start_time = time.time()
    cbkm = CBKM(k, bin_threshold)
    cbkm = cbkm.fit(X)
    cbkm_clusters = labels_to_clusters(X, cbkm.labels_)
    #pred = cbkm.predict(X)
    fscore, entropy, purity, nmi, si = compute_metrics(cbkm_clusters, X, Y)
    end_time = time.time()
    cbkm_fmeasures.append(np.round(fscore,4))
    cbkm_entropies.append(np.round(entropy,4))
    cbkm_purities.append(np.round(purity,4))
    cbkm_nmis.append(np.round(nmi,4))
    cbkm_sis.append(np.round(si,4))
    cbkm_times.append(np.round(end_time-start_time, 4))

cbkm_fmeasures_mean = sum(cbkm_fmeasures) / len(cbkm_fmeasures)
cbkm_fmeasures_std = np.std(cbkm_fmeasures)

cbkm_entropies_mean = sum(cbkm_entropies) / len(cbkm_entropies)
cbkm_entropies_std = np.std(cbkm_entropies)

cbkm_purities_mean = sum(cbkm_purities) / len(cbkm_purities)
cbkm_purities_std = np.std(cbkm_purities)

cbkm_nmis_mean = sum(cbkm_nmis) / len(cbkm_nmis)
cbkm_nmis_std = np.std(cbkm_nmis)

cbkm_sis_mean = sum(cbkm_sis) / len(cbkm_sis)
cbkm_sis_std = np.std(cbkm_sis)

cbkm_times_mean = sum(cbkm_times) / len(cbkm_times)
cbkm_times_std = np.std(cbkm_times)

# KM Algorithm
km_fmeasures, km_entropies, \
km_purities, km_nmis, km_sis, km_times = [], [], [], [], [], []
for i in range(ITER):
    print("\nExecuting KM....")
    start_time = time.time()
    kmeans = KMeans(n_clusters=k, n_init = 1).fit(X)
    km_clusters = labels_to_clusters(X, kmeans.labels_)
    fscore, entropy, purity, nmi, si = compute_metrics(km_clusters, X, Y)
    end_time = time.time()
    km_fmeasures.append(np.round(fscore,4))
    km_entropies.append(np.round(entropy,4))
    km_purities.append(np.round(purity,4))
    km_nmis.append(np.round(nmi,4))
    km_sis.append(np.round(si,4))
    km_times.append(np.round(end_time-start_time, 4))

km_fmeasures_mean = sum(km_fmeasures) / len(km_fmeasures)
km_fmeasures_std = np.std(km_fmeasures)
t_km_fmeasure, _ = stats.ttest_ind(cbkm_fmeasures, km_fmeasures)
h_km_fmeasure = rejected_hipothesis(t_km_fmeasure)

km_entropies_mean = sum(km_entropies) / len(km_entropies)
km_entropies_std = np.std(km_entropies)
t_km_entropy, _ = stats.ttest_ind(cbkm_entropies, km_entropies)
h_km_entropy = rejected_hipothesis(t_km_entropy)

km_purities_mean = sum(km_purities) / len(km_purities)
km_purities_std = np.std(km_purities)
t_km_purity, _ = stats.ttest_ind(cbkm_purities, km_purities)
h_km_purity = rejected_hipothesis(t_km_purity)

km_nmis_mean = sum(km_nmis) / len(km_nmis)
km_nmis_std = np.std(km_nmis)
t_km_nmi, _ = stats.ttest_ind(cbkm_nmis, km_nmis)
h_km_nmi = rejected_hipothesis(t_km_nmi)

km_sis_mean = sum(km_sis) / len(km_sis)
km_sis_std = np.std(km_sis)
t_km_si, _ = stats.ttest_ind(cbkm_sis, km_sis)
h_km_si = rejected_hipothesis(t_km_si)

km_times_mean = sum(km_times) / len(km_times)
km_times_std = np.std(km_times)
t_km_time, _ = stats.ttest_ind(cbkm_times, km_times)
h_km_time = rejected_hipothesis(t_km_time)

# BKM Algorithm
bkm_fmeasures, bkm_entropies, \
bkm_purities, bkm_nmis, bkm_sis, bkm_times = [], [], [], [], [], []
for i in range(ITER):
    # BISECTING KMEANS
    print("\nExecuting BKM....")
    start_time = time.time()
    bkmeans = BKM(n_clusters=k).fit(X)
    bkm_clusters = labels_to_clusters(X, bkmeans.labels_)
    fscore, entropy, purity, nmi, si = compute_metrics(bkm_clusters, X, Y)
    end_time = time.time()
    bkm_fmeasures.append(np.round(fscore,4))
    bkm_entropies.append(np.round(entropy,4))
    bkm_purities.append(np.round(purity,4))
    bkm_nmis.append(np.round(nmi,4))
    bkm_sis.append(np.round(si,4))
    bkm_times.append(np.round(end_time-start_time, 4))

bkm_fmeasures_mean = sum(bkm_fmeasures) / len(bkm_fmeasures)
bkm_fmeasures_std = np.std(bkm_fmeasures)
t_bkm_fmeasure, _ = stats.ttest_ind(cbkm_fmeasures, bkm_fmeasures)
h_bkm_fmeasure = rejected_hipothesis(t_bkm_fmeasure)

bkm_entropies_mean = sum(bkm_entropies) / len(bkm_entropies)
bkm_entropies_std = np.std(bkm_entropies)
t_bkm_entropy, _ = stats.ttest_ind(cbkm_entropies, bkm_entropies)
h_bkm_entropy = rejected_hipothesis(t_bkm_entropy)

bkm_purities_mean = sum(bkm_purities) / len(bkm_purities)
bkm_purities_std = np.std(bkm_purities)
t_bkm_purity, _ = stats.ttest_ind(cbkm_purities, bkm_purities)
h_bkm_purity = rejected_hipothesis(t_bkm_purity)

bkm_nmis_mean = sum(bkm_nmis) / len(bkm_nmis)
bkm_nmis_std = np.std(bkm_nmis)
t_bkm_nmi, _ = stats.ttest_ind(cbkm_nmis, bkm_nmis)
h_bkm_nmi = rejected_hipothesis(t_bkm_nmi)

bkm_sis_mean = sum(bkm_sis) / len(bkm_sis)
bkm_sis_std = np.std(bkm_sis)
t_bkm_si, _ = stats.ttest_ind(cbkm_sis, bkm_sis)
h_bkm_si = rejected_hipothesis(t_bkm_si)

bkm_times_mean = sum(bkm_times) / len(bkm_times)
bkm_times_std = np.std(bkm_times)
t_bkm_time, _ = stats.ttest_ind(cbkm_times, bkm_times)
h_bkm_time = rejected_hipothesis(t_bkm_time)

# Single Link Agglomerative Clustering Algorithm
print("\nExecuting SL....")
start_time = time.time()
sl = AgglomerativeClustering(n_clusters = k, linkage = 'single').fit(X)
sl_clusters = labels_to_clusters(X, sl.labels_)
sl_fscore, sl_entropy, sl_purity, sl_nmi, sl_si = compute_metrics(sl_clusters,X, Y)
end_time = time.time()
sl_time = end_time-start_time

resultsfile.write("COOPERATIVE BISECTING K-MEANS (k = {})\n".format(k))
resultsfile.write("-----------------------------------\n")
resultsfile.write("F-measure: mean {}, std {}\n".format(np.round(cbkm_fmeasures_mean, 4), np.round(cbkm_fmeasures_std, 4)))
resultsfile.write("Entropy: mean {}, std {}\n".format(np.round(cbkm_entropies_mean, 4), np.round(cbkm_entropies_std, 4)))
resultsfile.write("Purity: mean {}, std {}\n".format(np.round(cbkm_purities_mean, 4), np.round(cbkm_purities_std, 4)))
resultsfile.write("Normalized Mutual Information: mean {}, std {}\n".format(np.round(cbkm_nmis_mean, 4), np.round(cbkm_nmis_std, 4)))
resultsfile.write("Separation Index: mean {}, std {}\n".format(np.round(cbkm_sis_mean, 4), np.round(cbkm_sis_std, 4)))
resultsfile.write("Execution Time: mean {}, std {}\n".format(np.round(cbkm_times_mean, 4), np.round(cbkm_times_std, 4)))

resultsfile.write("\nK-MEANS (k = {})\n".format(k))
resultsfile.write("-----------------------------------\n")
resultsfile.write("F-measure: mean {}, std {} | t = {}, {}\n".format(np.round(km_fmeasures_mean, 4), np.round(km_fmeasures_std, 4), np.round(t_km_fmeasure, 4), h_km_fmeasure))
resultsfile.write("Entropy: mean {}, std {} | t = {}, {}\n".format(np.round(km_entropies_mean, 4), np.round(km_entropies_std, 4), np.round(t_km_entropy, 4), h_km_entropy))
resultsfile.write("Purity: mean {}, std {} | t = {}, {}\n".format(np.round(km_purities_mean, 4), np.round(km_purities_std, 4), np.round(t_km_purity, 4), h_km_purity))
resultsfile.write("Normalized Mutual Information: mean {}, std {} | t = {}, {}\n".format(np.round(km_nmis_mean, 4), np.round(km_nmis_std, 4), np.round(t_km_nmi, 4), h_km_nmi))
resultsfile.write("Separation Index: mean {}, std {} | t = {}, {}\n".format(np.round(km_sis_mean, 4), np.round(km_sis_std, 4), np.round(t_km_si, 4), h_km_si))
resultsfile.write("Execution Time: mean {}, std {} | t = {}, {}\n".format(np.round(km_times_mean, 4), np.round(km_times_std, 4), np.round(t_km_time, 4), h_km_time))

resultsfile.write("\nBISECTING K-MEANS (k = {})\n".format(k))
resultsfile.write("-----------------------------------\n")
resultsfile.write("F-measure: mean {}, std {} | t = {}, {}\n".format(np.round(bkm_fmeasures_mean, 4), np.round(bkm_fmeasures_std, 4), np.round(t_bkm_fmeasure, 4), h_bkm_fmeasure))
resultsfile.write("Entropy: mean {}, std {} | t = {}, {}\n".format(np.round(bkm_entropies_mean, 4), np.round(bkm_entropies_std, 4), np.round(t_bkm_entropy, 4), h_bkm_entropy))
resultsfile.write("Purity: mean {}, std {} | t = {}, {}\n".format(np.round(bkm_purities_mean, 4), np.round(bkm_purities_std, 4), np.round(t_bkm_purity, 4), h_bkm_purity))
resultsfile.write("Normalized Mutual Information: mean {}, std {} | t = {}, {}\n".format(np.round(bkm_nmis_mean, 4), np.round(bkm_nmis_std, 4), np.round(t_bkm_nmi, 4), h_bkm_nmi))
resultsfile.write("Separation Index: mean {}, std {} | t = {}, {}\n".format(np.round(bkm_sis_mean, 4), np.round(bkm_sis_std, 4), np.round(t_bkm_si, 4), h_bkm_si))
resultsfile.write("Execution Time: mean {}, std {} | t = {}, {}\n".format(np.round(bkm_times_mean, 4), np.round(bkm_times_std, 4), np.round(t_bkm_time, 4), h_bkm_time))

resultsfile.write("\nAGGLOMERATIVE SINGLE LINK (k = {})\n".format(k))
resultsfile.write("-----------------------------------\n")
resultsfile.write("F-measure: mean {}\n".format(np.round(sl_fscore, 4)))
resultsfile.write("Entropy: mean {}\n".format(np.round(sl_entropy, 4)))
resultsfile.write("Purity: mean {}\n".format(np.round(sl_purity, 4)))
resultsfile.write("Normalized Mutual Information: mean {}\n".format(np.round(sl_nmi, 4)))
resultsfile.write("Separation Index: mean {}\n".format(np.round(sl_si, 4)))
resultsfile.write("Execution Time: mean {}\n".format(np.round(sl_time, 4)))


resultsfile.close()