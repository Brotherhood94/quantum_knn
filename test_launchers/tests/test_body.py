from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

from knn.amplitude_quantum_knn import AmplitudeQKNeighborsClassifier
from knn.basis_quantum_knn import BasisQKNeighborsClassifier
from knn.basis_threshold import BasisRuan

from sklearn.model_selection import train_test_split

import time
import math
import numpy as np

from collections import Counter

def check_iters(X, y, n_records_per_class, n_classes, test_iters, train_iters, test_size):
    #this is redundad but I need to know here how many element for each class are in the training set. 
    #I'm assuming that the split here is the very same of the one defined in the selection.py //random_state=42
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y) 
    test_size = int(math.floor(len(X)*test_size))

    C = Counter(y_train)
    minimum = min(C, key=C.get)
    training_size = int(math.floor((C[minimum]*n_classes)/(n_records_per_class*n_classes)))

    if test_iters > test_size:
        test_iters = test_size

    if train_iters > training_size:
        train_iters = training_size

    return test_iters, train_iters


def execute_body(knn_k, selected_X_train, selected_X_test, selected_y_train, selected_y_test,
                 dataset,
                 n_classes,
                 n_features_real,
                 training_type,
                 training_size,
                 nbr_features,
                 pca,
                 test_id,
                 exp_id
                 ):

    #Classical KNN
    
    classical_knn = KNeighborsClassifier(n_neighbors=knn_k)

    start_fit = time.time()
    classical_knn.fit(selected_X_train, selected_y_train)
    end_fit = time.time()
    c_train_time = end_fit - start_fit

    start_pred = time.time()
    classic_pred = classical_knn.predict(selected_X_test)[0]
    end_pred = time.time()
    c_test_time = end_pred - start_pred

    KNN = { 'dataset': dataset,
           'n_neighbors':knn_k,
                   'n_classes': n_classes,
                   'n_features_real': n_features_real,
                   'method':'KNN',
                   'methodology': 'amplitude',
                   'n_bits': -1,
                   'enc_type': -1,
                   'training_type': training_type,
                   'training_size': training_size,
                   'nbr_features': nbr_features,
                   'pca': pca,
                   'test_id': test_id,
                   'exp_id': exp_id,
                   'accuracy_test': int(selected_y_test[0]==classic_pred),
                   'train_time': c_train_time,
                   'test_time': c_test_time,
                   'y_true': selected_y_test[0],
                   'y_pred': classic_pred}

    ############################################################
    
    #Quantum KNN

    aqknn = AmplitudeQKNeighborsClassifier(n_neighbors=knn_k)

    start_fit = time.time()
    aqknn.fit(selected_X_train, selected_y_train)
    end_fit = time.time()
    q_train_time = end_fit - start_fit

    start_pred = time.time()
    quantum_pred = aqknn.predict(selected_X_test)  #majority vote of first elem
    end_pred = time.time()
    q_test_time = end_pred - start_pred


    aQKNN = {'dataset': dataset,
            'n_neighbors':knn_k,
                     'n_classes': n_classes,
                     'n_features_real': n_features_real,
                     'method':'aQKNN',
                     'methodology': 'amplitude',
                     'n_bits': -1,
                     'enc_type': -1,
                     'training_type': training_type,
                     'training_size': training_size,
                     'nbr_features': nbr_features,
                     'pca': pca,
                     'test_id': test_id,
                     'exp_id': exp_id,
                     'accuracy_test': int(selected_y_test[0]==quantum_pred),
                     'train_time': q_train_time,
                     'test_time': q_test_time,
                     'y_true': selected_y_test[0],
                     'y_pred': quantum_pred}

    return KNN, aQKNN


def execute_body_basis(knn_k, selected_X_train, selected_X_test, selected_y_train, selected_y_test,
                 dataset,
                 n_classes,
                 n_features_real,
                 training_type,
                 training_size,
                 nbr_features,
                 pca,
                 test_id,
                 exp_id,
                 n_bits,
                 enc_type
                 ):


    #extracting from list of series
    selected_X_train = [x_train[0] for x_train in selected_X_train]
    #selected_y_train = [y_train[0] for y_train in selected_y_train]
    selected_X_test = [x_test[0] for x_test in selected_X_test]

    #############################################################

    classical_knn = KNeighborsClassifier(n_neighbors=knn_k, metric='hamming')
    encoded_selected_X_train = []
    for i in range(len(selected_X_train)):
        encoded_selected_X_train.append(np.array([int(t) for t in selected_X_train[i]]))

    start_fit = time.time()
    classical_knn.fit(encoded_selected_X_train, selected_y_train)
    end_fit = time.time()
    c_train_time = end_fit - start_fit

    test_instance = np.array([int(b) for b in selected_X_test[0]])
    test_instance = test_instance.reshape(1,-1)
    start_pred = time.time()
    classic_pred = classical_knn.predict(test_instance)
    end_pred = time.time()
    c_test_time = end_pred - start_pred

    bKNN = { 'dataset': dataset,
           'n_neighbors':knn_k,
                   'n_classes': n_classes,
                   'n_features_real': n_features_real,
                   'method':'bKNN',
                   'methodology': 'basis',
                   'n_bits': n_bits,
                   'enc_type': enc_type,
                   'training_type': training_type,
                   'training_size': training_size,
                   'nbr_features': nbr_features,
                   'pca': pca,
                   'test_id': test_id,
                   'exp_id': exp_id,
                   'accuracy_test': int(selected_y_test[0]==classic_pred),
                   'train_time': c_train_time,
                   'test_time': c_test_time,
                   'y_true': selected_y_test[0],
                   'y_pred': classic_pred[0]}

    ############################################################

    #Basis Quantum KNN

    x_to_y = dict(zip(selected_X_train, selected_y_train))
    #removing duplicates otherwise amplitudes problem with basis encoding -- mantenere associazione classe
    selected_X_train = list(x_to_y.keys())
    selected_y_train = list(x_to_y.values())

    bqknn = BasisQKNeighborsClassifier(precision=n_bits)

    start_fit = time.time()
    bqknn.fit(selected_X_train)
    end_fit = time.time()
    q_train_time = end_fit - start_fit

    start_pred = time.time()
    x_quantum_pred = bqknn.predict(selected_X_test[0])  #majority vote of first elem
    if x_quantum_pred == -1:
        quantum_pred = -1
        accuracy_test = -1
    else:
        quantum_pred = x_to_y[x_quantum_pred]
        accuracy_test = int(selected_y_test[0]==quantum_pred),
    end_pred = time.time()
    q_test_time = end_pred - start_pred


    bQKNN = {'dataset': dataset,
           'n_neighbors':knn_k,
                     'n_classes': n_classes,
                     'n_features_real': n_features_real,
                     'method':'bQKNN',
                     'methodology': 'basis',
                     'n_bits': n_bits,
                     'enc_type': enc_type,
                     'training_type': training_type,
                     'training_size': training_size,
                     'nbr_features': nbr_features,
                     'pca': pca,
                     'test_id': test_id,
                     'exp_id': exp_id,
                     'accuracy_test': accuracy_test,
                     'train_time': q_train_time,
                     'test_time': q_test_time,
                     'y_true': selected_y_test[0],
                     'y_pred': quantum_pred}
    #return bQKNN
    return bKNN, bQKNN

def execute_body_basis_threshold(knn_k, selected_X_train, selected_X_test, selected_y_train, selected_y_test,
                 dataset,
                 n_classes,
                 n_features_real,
                 training_type,
                 training_size,
                 nbr_features,
                 pca,
                 test_id,
                 exp_id,
                 n_bits,
                 enc_type
                 ):


    #extracting from list of series
    selected_X_train = [x_train[0] for x_train in selected_X_train]
    #selected_y_train = [y_train[0] for y_train in selected_y_train]
    selected_X_test = [x_test[0] for x_test in selected_X_test]

    #############################################################

    classical_knn = KNeighborsClassifier(n_neighbors=knn_k, metric='hamming')
    encoded_selected_X_train = []
    for i in range(len(selected_X_train)):
        encoded_selected_X_train.append(np.array([int(t) for t in selected_X_train[i]]))

    start_fit = time.time()
    classical_knn.fit(encoded_selected_X_train, selected_y_train)
    end_fit = time.time()
    c_train_time = end_fit - start_fit

    test_instance = np.array([int(b) for b in selected_X_test[0]])
    test_instance = test_instance.reshape(1,-1)
    start_pred = time.time()
    classic_pred = classical_knn.predict(test_instance)
    end_pred = time.time()
    c_test_time = end_pred - start_pred

    bKNN = { 'dataset': dataset,
           'n_neighbors':knn_k,
                   'n_classes': n_classes,
                   'n_features_real': n_features_real,
                   'method':'bKNN',
                   'methodology': 'basis_threshold',
                   'n_bits': n_bits,
                   'enc_type': enc_type,
                   'training_type': training_type,
                   'training_size': training_size,
                   'nbr_features': nbr_features,
                   'pca': pca,
                   'test_id': test_id,
                   'exp_id': exp_id,
                   'accuracy_test': int(selected_y_test[0]==classic_pred),
                   'train_time': c_train_time,
                   'test_time': c_test_time,
                   'y_true': selected_y_test[0],
                   'y_pred': classic_pred[0]}

    ############################################################

    #Basis Ruan Quantum KNN
    # convert integer classes to binary class
    selected_y_train = [np.binary_repr(y_train, width=int(math.ceil(math.log2(n_classes)))) for y_train in selected_y_train]

    x_to_y = dict(zip(selected_X_train, selected_y_train))
    #removing duplicates otherwise amplitudes problem with basis encoding -- keep class mapping
    selected_X_train = list(x_to_y.keys())
    selected_y_train = list(x_to_y.values())

    btqknn = BasisRuan(threshold=knn_k)

    #print(selected_X_train)
    #print(selected_y_train)
    start_fit = time.time()
    btqknn.fit(selected_X_train, selected_y_train)
    end_fit = time.time()
    q_train_time = end_fit - start_fit

    start_pred = time.time()
    y_quantum_pred = btqknn.predict(selected_X_test[0]) 
    if y_quantum_pred == -1:
        quantum_pred = -1
        accuracy_test = -1
    else:
        quantum_pred = int(y_quantum_pred, 2) #binary value to int value.
        accuracy_test = int(selected_y_test[0]==quantum_pred),
    end_pred = time.time()
    q_test_time = end_pred - start_pred


    btQKNN = {'dataset': dataset,
           'n_neighbors':knn_k,
                     'n_classes': n_classes,
                     'n_features_real': n_features_real,
                     'method':'btQKNN',
                     'methodology': 'basis_threshold',
                     'n_bits': n_bits,
                     'enc_type': enc_type,
                     'training_type': training_type,
                     'training_size': training_size,
                     'nbr_features': nbr_features,
                     'pca': pca,
                     'test_id': test_id,
                     'exp_id': exp_id,
                     'accuracy_test': accuracy_test,
                     'train_time': q_train_time,
                     'test_time': q_test_time,
                     'y_true': selected_y_test[0],
                     'y_pred': quantum_pred}
    #return bQKNN
    return bKNN, btQKNN
