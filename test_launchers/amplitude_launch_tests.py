from tkinter import W
from sklearn.datasets import load_iris, load_breast_cancer, load_digits
from datasets.load_external_dataset import load_sonar, load_soybean_small
import numpy as np

import tests.amplitude_tests.test_1 as t1
import tests.amplitude_tests.test_2 as t2
import tests.amplitude_tests.test_3 as t3
import tests.amplitude_tests.test_4 as t4


#datasets = {load_iris:'iris', load_breast_cancer:'breast_cancer', load_digits:'digits', load_sonar:'sonar', load_soybean_small:'soybean_small'}
datasets = {load_sonar:'sonar', load_soybean_small:'soybean_small'}

for d_call, d_name in datasets.items():
    X, y = d_call(return_X_y=True, as_frame=True)
    knn_k = 1

    dataset = d_name
    n_classes = len(np.unique(y))
    n_features_real = len(X.columns)
    training_type = 'sample'
    test_size = 0.3


    print("[Started Test 1]")
    # 1 test istance, 1 istance per class training
    # 100 Random Selection on Training, 100 Random Selection on Test
    test_iters_t1 = 50
    train_iters_t1 = 50

    t1.test_1_no_pca(X, y, knn_k, test_iters_t1, train_iters_t1, dataset, n_classes, n_features_real, training_type, test_size)


    print("[Started Test 1 PCA]")
    t1.test_1_pca(X, y, knn_k, test_iters_t1, train_iters_t1, dataset, n_classes, n_features_real, training_type, test_size)

    ##############################################################################

    print("[Started Test 2]")
    # 1 test instance, n_records_per_class=2,4,8,16,32 in training
    # 20 Random Selection on Training, 100 Random Selection on Test, variable number of elements per class

    test_iters_t2 = 50
    train_iters_t2 = 50
    exp_records_per_class = 6 #2,4,8,16,32 (2^5)

    t2.test_2_no_pca(X, y, knn_k, test_iters_t2, train_iters_t2, exp_records_per_class,dataset, n_classes, n_features_real, training_type, test_size)

    print("[Started Test 2 PCA]")
    t2.test_2_pca(X, y, knn_k, test_iters_t2, train_iters_t2, exp_records_per_class, dataset, n_classes, n_features_real, training_type, test_size)

    ##############################################################################

    training_type = 'proto'
    print("[Started Test 3]")
    # 1 test instance, training fixed (column mean for each class)
    # 20 Random Selection on Training

    test_iters_t3 = 100
    t3.test_3_no_pca(X, y, knn_k, test_iters_t3, dataset, n_classes, n_features_real, training_type, test_size)

    print("[Started Test 3 PCA]")
    t3.test_3_pca(X, y, knn_k, test_iters_t3, dataset, n_classes, n_features_real, training_type, test_size)

    ##############################################################################

    print("[Started Test 4]")
    # 1 test instance, k kmeans
    # 100 Random Selection on k

    test_iters_t4 = 100

    t4.test_4_no_pca(X, y, knn_k, test_iters_t4, exp_records_per_class, dataset, n_classes, n_features_real, training_type, test_size)

    print("[Started Test 4 PCA]")
    t4.test_4_pca(X, y, knn_k, test_iters_t4, exp_records_per_class, dataset, n_classes, n_features_real, training_type, test_size)

    ##############################################################################
