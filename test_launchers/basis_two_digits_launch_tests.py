from sklearn.datasets import load_digits
import numpy as np

import tests.basis_two_digits_tests.test_1 as t1
import tests.basis_two_digits_tests.test_2 as t2
import tests.basis_two_digits_tests.test_3 as t3
import tests.basis_two_digits_tests.test_4 as t4


encodings = ['entropy', 'hash']
datasets = {load_digits:'digits'}

classes_comb = [ [0,8], [1,7], [4,5], [4,6], [7,9] ]


for comb in classes_comb:
    for enc_type in encodings:
        for d_call, d_name in datasets.items():
            X, y = d_call(return_X_y=True, as_frame=True)
            knn_k = 1

            n_classes = 2 
            training_type = 'sample'
            test_size = 0.3


            print("[Started Test 1 PCA]")
            # 1 test istance, 1 istance per class training
            # 100 Random Selection on Training, 100 Random Selection on Test
            test_iters_t1 = 100
            train_iters_t1 = 100
            t1.test_1_pca(X, y, knn_k, test_iters_t1, train_iters_t1, d_name, n_classes, 3, training_type, test_size, enc_type, 2, comb)
            t1.test_1_pca(X, y, knn_k, test_iters_t1, train_iters_t1, d_name, n_classes, 5, training_type, test_size, enc_type, 4, comb)

            ##############################################################################

            print("[Started Test 2 PCA]")
            # 1 test instance, n_records_per_class=2,4,8,16,32 in training
            # 20 Random Selection on Training, 100 Random Selection on Test, variable number of elements per class

            test_iters_t2 = 100
            train_iters_t2 = 100
            exp_records_per_class = 5 #2,4,8,16,32 (2^5)
            t2.test_2_pca(X, y, knn_k, test_iters_t2, train_iters_t2, exp_records_per_class, d_name, n_classes, 3, training_type, test_size, enc_type, 2, comb)
            t2.test_2_pca(X, y, knn_k, test_iters_t2, train_iters_t2, exp_records_per_class, d_name, n_classes, 5, training_type, test_size, enc_type, 4, comb)

            ##############################################################################

            print("[Started Test 3 PCA]")
            training_type = 'proto'
            # 1 test instance, training fixed (column mean for each class)
            # 20 Random Selection on Training

            test_iters_t3 = 100
            t3.test_3_pca(X, y, knn_k, test_iters_t3, d_name, n_classes, 3, training_type, test_size, enc_type, 2, comb)
            t3.test_3_pca(X, y, knn_k, test_iters_t3, d_name, n_classes, 5, training_type, test_size, enc_type, 4, comb)

            ##############################################################################


            print("[Started Test 4 PCA]")
            # 1 test instance, k kmeans
            # 100 Random Selection on k

            test_iters_t4 = 100

            t4.test_4_pca(X, y, knn_k, test_iters_t4, exp_records_per_class, d_name, n_classes, 3, training_type, test_size, enc_type, 2, comb)
            t4.test_4_pca(X, y, knn_k, test_iters_t4, exp_records_per_class, d_name, n_classes, 5, training_type, test_size, enc_type, 4, comb)

            ##############################################################################
