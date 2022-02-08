from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import preprocessing

from sklearn.cluster import KMeans
from kmodes.kmodes import KModes

import pandas as pd
import numpy as np

from utility.binary_encodings.binary_econding import base_encode


def _transform(X, y, n_features_pca=None, normalize=False, standardize=False, test_size=0.3, enc_type=None, n_bits=0):

    X_new = X
    y_new = y
    if n_features_pca != None: 
        pca_features = PCA(n_components=n_features_pca)
        X_new = pca_features.fit_transform(X_new)

    if standardize:
        scaler = StandardScaler().fit(X_new)
        X_new = scaler.transform(X_new)

    if normalize:
        X_new = preprocessing.normalize(X_new, axis=1)

    if enc_type != None:
        if n_bits <= 0:
            raise Exception('Invalid Number of n_bits')
        X_new, y_new = base_encode(X_new, y_new, n_features=n_features_pca, n_bits=n_bits, enc_type=enc_type)
        temp_df = pd.DataFrame(data=X_new, index=X.index)
        temp_df['target'] = y_new
        temp_df['target'] = temp_df.groupby([0])['target'].transform(lambda x: pd.Series.mode(x)[0])
        X_new = temp_df[0]
        y_new = temp_df['target']


    X_new = pd.DataFrame(X_new, index=X.index)
    X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=test_size, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test 



def _normalize(X_train, X_test):
    X_train = preprocessing.normalize(X_train, axis=1)
    X_test = preprocessing.normalize(X_test, axis=1)
    return X_train, X_test

def _standardize(X_train, X_test):
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def _apply_pca(X_train, X_test, n_features_pca):
    pca_features = PCA(n_components=n_features_pca)
    X_train = pd.DataFrame(pca_features.fit_transform(X_train), index=X_train.index)
    X_test = pd.DataFrame(pca_features.fit_transform(X_test), index=X_test.index)
    return X_train, X_test

def _select_test_sample(X_test, y_test, n_records_per_class=1, random_state=1):
    selected_X_test = []
    selected_y_test = []
    selected_test_indexes = y_test.sample(n=1, random_state=random_state)
    for index, _ in selected_test_indexes.items():
        selected_X_test.append(X_test.loc[index])
        selected_y_test.append(y_test.loc[index])
    return selected_X_test, selected_y_test, selected_test_indexes

def basis_selection_kmeans(X_train, X_test, y_train, y_test, n_records_per_class=1, random_state_test=1, discard_index_test=[]):
    selected_X_train = []
    selected_y_train = []
    df = pd.DataFrame(X_train, index=X_train.index)
    df['target'] = y_train
    for cl in df.target.unique():
        groupby_class = df.loc[df['target'] == cl]
        kmodes_on_class = KModes(n_clusters=n_records_per_class, random_state=0)
        kmodes_on_class.fit_predict(groupby_class.drop('target', 1))
        [selected_X_train.append(centroid) for centroid in kmodes_on_class.cluster_centroids_]
        [selected_y_train.append(cl) for i in range(len(kmodes_on_class.cluster_centroids_))]

    selected_X_test, selected_y_test, selected_test_indexes = _select_test_sample(X_test, y_test, random_state=random_state_test)

    return selected_X_train, selected_X_test, selected_y_train, selected_y_test, selected_test_indexes.index.values


def selection_kmeans(X_train, X_test, y_train, y_test, n_records_per_class=1, random_state_test=1, discard_index_test=[]):
    selected_X_train = []
    selected_y_train = []
    df = pd.DataFrame(X_train, index=X_train.index)
    df['target'] = y_train
    for cl in df.target.unique():
        groupby_class = df.loc[df['target'] == cl]
        kmeans_on_class = KMeans(n_clusters=n_records_per_class, random_state=0)
        kmeans_on_class.fit(groupby_class.drop('target', 1))
        [selected_X_train.append(centroid) for centroid in kmeans_on_class.cluster_centers_]
        [selected_y_train.append(cl) for i in range(n_records_per_class)]

    selected_X_test, selected_y_test, selected_test_indexes = _select_test_sample(X_test, y_test, random_state=random_state_test)

    return selected_X_train, selected_X_test, selected_y_train, selected_y_test, selected_test_indexes.index.values

def basis_selection_mean(X_train, X_test, y_train, y_test, random_state_test=1, discard_index_test=[]):
    selected_X_train = []
    selected_y_train = []
    if len(discard_index_test):
        X_test.drop(discard_index_test, inplace=True)
        y_test.drop(discard_index_test, inplace=True)


    df = pd.DataFrame(X_train, index=X_train.index)
    df['target'] = y_train
    highest_per_class = df.groupby('target')[0].apply(lambda x: x.value_counts().head(1))
    for multi_index,_ in highest_per_class.items():
        elem = multi_index[1]
        row = df[df[0]==elem].head(1)[0]
        row = pd.Series(row.squeeze()) 
        target = df[df[0]==elem].head(1)['target']
        selected_X_train.append(np.array([row[0]]))
        selected_y_train.append(target.squeeze())
    selected_X_test, selected_y_test, selected_test_indexes = _select_test_sample(X_test, y_test, random_state=random_state_test)
    return selected_X_train, selected_X_test, selected_y_train, selected_y_test, selected_test_indexes.index.values

def selection_mean(X_train, X_test, y_train, y_test, random_state_test=1, discard_index_test=[]):
    selected_X_train = []
    selected_y_train = []
    if len(discard_index_test):
        X_test.drop(discard_index_test, inplace=True)
        y_test.drop(discard_index_test, inplace=True)


    df = pd.DataFrame(X_train, index=X_train.index)
    df['target'] = y_train
    means = df.groupby('target').mean()
    for target, row in means.iterrows():
        selected_X_train.append(row)
        selected_y_train.append(target)

    selected_X_test, selected_y_test, selected_test_indexes = _select_test_sample(X_test, y_test, random_state=random_state_test)
    return selected_X_train, selected_X_test, selected_y_train, selected_y_test, selected_test_indexes.index.values


def selection_random(X_train, X_test, y_train, y_test, n_records_per_class=1, random_state_training=1, random_state_test=1, discard_index_train=[], discard_index_test=[]): 
    selected_X_train = []
    selected_y_train = []
    if len(discard_index_train):
        X_train.drop(discard_index_train, inplace=True)
        y_train.drop(discard_index_train, inplace=True)
    selected_X_test, selected_y_test, selected_test_indexes = _select_test_sample(X_test, y_test, random_state=random_state_test)
    selected_train_indexes = y_train.groupby(y_train).sample(n=n_records_per_class, random_state=random_state_training)
    for index, _ in selected_train_indexes.items():
        selected_X_train.append(X_train.loc[index])
        selected_y_train.append(y_train.loc[index])
    return selected_X_train, selected_X_test, selected_y_train, selected_y_test, selected_train_indexes.index.values, selected_test_indexes.index.values
