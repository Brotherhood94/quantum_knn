from sklearn.datasets import load_iris, load_breast_cancer, load_digits
from datasets.load_external_dataset import load_sonar, load_soybean_small, load_real_estate
import numpy as np
#import multiprocessing

import tests.basis_tests.test_1 as t1
import tests.basis_tests.test_2 as t2
import tests.basis_tests.test_3 as t3
import tests.basis_tests.test_4 as t4

encodings = ['entropy', 'hash']
#datasets = {load_iris:'iris', load_breast_cancer:'breast_cancer', load_digits:'digits'}
datasets = {load_sonar:'sonar', load_soybean_small:'soybean_small', load_real_estate:'real_estate'}

if __name__ == '__main__':
    for enc_type in encodings:
        for d_call, d_name in datasets.items():
            X, y = d_call(return_X_y=True, as_frame=True)
            knn_k = 1

            n_classes = len(np.unique(y))
            training_type = 'sample'
            test_size = 0.3

            print("N° classes: {}".format(n_classes))

            print("[Started Test 1 PCA]")
            # 1 test istance, 1 istance per class training
            # 100 Random Selection on Training, 100 Random Selection on Test

            test_iters_t1 = 100
            train_iters_t1 = 100

            processes = []

            #p0 = multiprocessing.Process(target=t1.test_1_pca , args=(X, y, knn_k, test_iters_t1, train_iters_t1, d_name, n_classes, 3, training_type, test_size, enc_type, 2))
            #processes.append(p0)
            t1.test_1_pca(X, y, knn_k, test_iters_t1, train_iters_t1, d_name, n_classes, 3, training_type, test_size, enc_type, 2)


            #p1 = multiprocessing.Process(target=t1.test_1_pca , args=(X, y, knn_k, test_iters_t1, train_iters_t1, d_name, n_classes, 5, training_type, test_size, enc_type, 4))
            #processes.append(p1)
            t1.test_1_pca(X, y, knn_k, test_iters_t1, train_iters_t1, d_name, n_classes, 5, training_type, test_size, enc_type, 4)

            ##############################################################################



            print("[Started Test 2 PCA]")
            # 1 test instance, n_records_per_class=2,4,8,16,32 in training
            # 20 Random Selection on Training, 100 Random Selection on Test, variable number of elements per class

            test_iters_t2 = 100
            train_iters_t2 = 100
            exp_records_per_class = 5 #2,4,8,16,32 (2^5)

            #p2 = multiprocessing.Process(target=t2.test_2_pca , args=(X, y, knn_k, test_iters_t2, train_iters_t2, exp_records_per_class, d_name, n_classes, 3, training_type, test_size, enc_type, 2))
            #processes.append(p2)
            t2.test_2_pca(X, y, knn_k, test_iters_t2, train_iters_t2, exp_records_per_class, d_name, n_classes, 3, training_type, test_size, enc_type, 2)

            #p3 = multiprocessing.Process(target=t2.test_2_pca , args=(X, y, knn_k, test_iters_t2, train_iters_t2, exp_records_per_class, d_name, n_classes, 5, training_type, test_size, enc_type, 4))
            #processes.append(p3)
            t2.test_2_pca(X, y, knn_k, test_iters_t2, train_iters_t2, exp_records_per_class, d_name, n_classes, 5, training_type, test_size, enc_type, 4)

            ##############################################################################



            print("[Started Test 3 PCA]")
            # 1 test instance, training fixed (column mean for each class)
            # 20 Random Selection on Training

            training_type = 'proto'
            test_iters_t3 = 100

            #p4 = multiprocessing.Process(target=t3.test_3_pca, args=(X, y, knn_k, test_iters_t3, d_name, n_classes, 3, training_type, test_size, enc_type, 2))
            #processes.append(p4)
            t3.test_3_pca(X, y, knn_k, test_iters_t3, d_name, n_classes, 3, training_type, test_size, enc_type, 2)

            #p5 = multiprocessing.Process(target=t3.test_3_pca , args=(X, y, knn_k, test_iters_t3, d_name, n_classes, 5, training_type, test_size, enc_type, 4))
            #processes.append(p5)
            t3.test_3_pca(X, y, knn_k, test_iters_t3, d_name, n_classes, 5, training_type, test_size, enc_type, 4)

            ##############################################################################


            print("[Started Test 4 PCA]")
            test_iters_t4 = 100
            # 1 test instance, k kmeans
            # 100 Random Selection on k

            #p6 = multiprocessing.Process(target=t4.test_4_pca , args=(X, y, knn_k, test_iters_t4, exp_records_per_class, d_name, n_classes, 3, training_type, test_size, enc_type, 2))
            #processes.append(p6)
            t4.test_4_pca(X, y, knn_k, test_iters_t4, exp_records_per_class, d_name, n_classes, 3, training_type, test_size, enc_type, 2)

            #p7 = multiprocessing.Process(target=t4.test_4_pca , args=(X, y, knn_k, test_iters_t4, exp_records_per_class, d_name, n_classes, 5, training_type, test_size, enc_type, 4))
            #processes.append(p7)
            t4.test_4_pca(X, y, knn_k, test_iters_t4, exp_records_per_class, d_name, n_classes, 5, training_type, test_size, enc_type, 4)

            ##############################################################################

            '''
            for process in processes:
                process.start()

            for process in processes:
                process.join()
            '''
