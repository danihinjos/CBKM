# Imports
from scipy.io import arff
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from scipy.stats import entropy
import math


# Method to load and examine dataset
def load_data(filepath):
    if ".csv" in filepath:
        data = pd.read_csv(filepath)
    elif ".arff" in filepath:
        data, _ = arff.loadarff(filepath)
    elif ".txt" in filepath:
        data = pd.read_csv(filepath)
    else:
        print("File not found.\n")
        return False, 0
    df = pd.DataFrame(data)
    # Dataset dimension
    print("Dimensions of the dataset: " + str(df.shape))
    # There are 1473 instances with 9 attributes each and the corresponding class.

    print("\nInformation about the attributes:")
    print(df.info())
    # This dataset has numerical and categorical attributes but the advantage
    # is that nominal attributes are numerically labelled so all the dataset
    # can be transform into a numerical dataset.
    # Moreover, there aren't missing values

    # Peek at the data
    print("\nFirst 5 rows of the dataset:")
    print(df.head(5))

    print("\nStatistical summary:")
    print(df.describe())
    # The values of the numeral attributes has different ranges. This indicates that a normalization or
    # standardization will be necessary.

    print("\nClasses: " + str(df[df.columns[-1]].unique()))
    print("--------------------------------------------------------------------------------\n")

    return True, df


# Preprocess dataset
def preprocessing_step(dataset):
    for attr in dataset.columns:
        if is_categorical(dataset[attr]):
            # Decoding
            dataset[attr] = dataset[attr].str.decode('utf-8')
            # Missing categorical values: fill with mode
            dataset[attr].fillna(dataset[attr].mode(), inplace=True)
            # Discretize
            # Label Encoding
            le = preprocessing.LabelEncoder()
            dataset[attr] = le.fit_transform(dataset[attr])
        else:
            dataset[attr].fillna(dataset[attr].mean(), inplace=True)
            # Normalization with MinMaxScaler
            scaler = MinMaxScaler()
            dataset[attr] = scaler.fit_transform(dataset[attr].values.reshape(-1,1))
    return dataset


# Verify if an attribute is categorical
def is_categorical(array_like):
    return array_like.dtype.name == 'category' or array_like.dtype.name == 'object'


# Verify if an attribute is numerical
def is_numerical(array_like):
    return array_like.dtype.name == 'int64' or array_like.dtype.name == 'float64'


# Utility function to request user input
def make_selection(title, choices, prompt='Select one of the choices above'):
    print(title)
    print('-'*len(title))
    for choice in choices:
        print(str(choice) + ' - ' + choices[choice])

    selection_valid = False

    while (not selection_valid):
        selection = input(prompt  + ': ')

        if(selection in choices):
            selection_valid = True
        else:
            print('Error: Unrecognized option. Try again.')

    return selection


# Method to recalculate cluster centroids, used mainly by BKM class
def recalculate_centroids(clusters):
    centroids = [np.mean(c, 0) for c in clusters]
    return centroids


# Method to obtain the clusters from the clustering labels
def labels_to_clusters(data, labels):
     clustering = []
     for value in set(labels):
        clustering.append([row for row, label in zip(data, labels) if label == value])
     return clustering


# Method to obtain the labels from the clustering
def clusters_to_labels(clusters, datax):
    labels = []
    for instance in datax:
        for id, c in enumerate(clusters):
            for csample in c:
                if (instance == csample).all():
                    labels.append(id)
                    break
    return labels


# Method to obtain the labels from a specific cluster
def get_cluster_labels(cluster, datax, datay):
    cluster_labels = []
    for cinstance in cluster:
        for instance, label in zip(datax, datay):
            if (cinstance == instance).all():
                cluster_labels.append(label)
                break
    return cluster_labels
