import utility.selections as sel
import math
from tqdm import trange
from utility.save_results import print_to_file
from tests.test_body import execute_body, check_iters


def test_4_no_pca(X, y, knn_k, test_iters, exp_records_per_class, dataset, n_classes, n_features_real, training_type, test_size):

    #TEST 4
    # 1 test instance, k kmeans
    # 100 Random Selection on k

    ##### TEST 4 No PCA #####

    aQKNN_exps = []
    KNN_exps = []

    di_test = []

    pca = 'false'
    for n_records_per_class in (2**e for e in trange(1, exp_records_per_class, desc='n_records_per_class')):
        test_iters, _ = check_iters(X, y, n_records_per_class, n_classes, test_iters, 1, test_size)
        X_train, X_test, y_train, y_test = sel._transform(X, y, n_features_pca=None, normalize=True, standardize=True, test_size=test_size)
        for random_state_test in trange(0, test_iters, desc='test', leave=False): #random state test
              selected_X_train, selected_X_test, selected_y_train, selected_y_test, di_test =  sel.selection_kmeans(
                                                                                                       X_train, X_test, y_train, y_test,
                                                                                                       n_records_per_class=n_records_per_class,
                                                                                                       random_state_test=random_state_test,
                                                                                                       discard_index_test=di_test)


              KNN, aQKNN = execute_body(knn_k, selected_X_train, selected_X_test, selected_y_train, selected_y_test,
                                                    dataset,
                                                    n_classes,
                                                    n_features_real,
                                                    training_type,
                                                    len(selected_X_train),#training_size
                                                    len(selected_X_train[0]),
                                                    pca,
                                                    di_test[0], #test_id
                                                    0 #exp_id
                                                    )

              aQKNN_exps.append(aQKNN)
              KNN_exps.append(KNN)


        print_to_file(dataset+"_TEST_4_No_PCA_n_records_per_class"+str(n_records_per_class), KNN_exps, aQKNN_exps)
        aQKNN_exps = []
        KNN_exps = []
        di_test = []



def test_4_pca(X, y, knn_k, test_iters, exp_records_per_class, dataset, n_classes, n_features_real, training_type, test_size):

    #### TEST_4 PCA


    aQKNN_exps = []
    KNN_exps = []

    di_test = []

    pca = 'true'
    features_range = trange(1,  int(math.ceil(math.log2(n_features_real))), desc='pca')
    for n_features_pca in (2**e for e in features_range): #start from 2, exponential step
        for n_records_per_class in (2**e for e in trange(1, exp_records_per_class, desc='n_records_per_class', leave=False)):
            test_iters, _ = check_iters(X, y, n_records_per_class, n_classes, test_iters, 1, test_size)
            X_train, X_test, y_train, y_test = sel._transform(X, y, n_features_pca=n_features_pca, normalize=True, standardize=True, test_size=test_size)
            for random_state_test in trange(0, test_iters, desc='test', leave=False): #random state test
                  selected_X_train, selected_X_test, selected_y_train, selected_y_test, di_test = sel.selection_kmeans(
                                                                                                           X_train, X_test, y_train, y_test,
                                                                                                           n_records_per_class=n_records_per_class,
                                                                                                           random_state_test=random_state_test,
                                                                                                           discard_index_test=di_test)

                  KNN, aQKNN = execute_body(knn_k, selected_X_train, selected_X_test, selected_y_train, selected_y_test,
                                                    dataset,
                                                    n_classes,
                                                    n_features_real,
                                                    training_type,
                                                    len(selected_X_train),#training_size
                                                    len(selected_X_train[0]),
                                                    pca,
                                                    di_test[0], #test_id
                                                    0 #exp_id
                                                    )


                  aQKNN_exps.append(aQKNN)
                  KNN_exps.append(KNN)


            print_to_file(dataset+"_TEST_4_PCA:"+str(n_features_pca)+"_n_records_per_class_"+str(n_records_per_class), KNN_exps, aQKNN_exps)
            aQKNN_exps = []
            KNN_exps = []
            di_test = []



