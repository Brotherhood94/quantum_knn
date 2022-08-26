import utility.selections as sel
from tqdm import trange
from utility.save_results import print_to_file
from tests.test_body import execute_body, check_iters

def test_3_no_pca(X, y, knn_k, test_iters, dataset, n_classes, n_features_real, training_type, test_size):
    #TEST 3
    # 1 test instance, training fixed (column mean for each class)
    # 20 Random Selection on Training

    ##### TEST 3 No PCA #####

    aQKNN_exps = []
    KNN_exps = []

    di_test = []

    pca = 'false'

    test_iters, _ = check_iters(X, y, 1, 1, test_iters, 1, test_size)
    X_train, X_test, y_train, y_test = sel._transform(X, y, n_features_pca=8, normalize=True, standardize=True, test_size=test_size)
    for random_state_test in trange(0, test_iters, desc='test'): #random state test
        selected_X_train, selected_X_test, selected_y_train, selected_y_test, di_test = sel.selection_mean(
                                                                                              X_train, X_test, y_train, y_test,
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


    print_to_file(dataset+"_TEST_3_No_PCA", KNN_exps, aQKNN_exps)
    aQKNN_exps = []
    KNN_exps = []

### TEST_3_PCA

def test_3_pca(X, y, knn_k, test_iters, dataset, n_classes, n_features_real, training_type, test_size):

    aQKNN_exps = []
    KNN_exps = []

    di_test = []

    pca = 'true'

    test_iters, _ = check_iters(X, y, 1, 1, test_iters, 1, test_size)
    features_range = trange(1,  4, desc='pca')
    for n_features_pca in (2**e for e in features_range): #start from 2, exponential step
        X_train, X_test, y_train, y_test = sel._transform(X, y, n_features_pca=n_features_pca, normalize=True, standardize=True, test_size=test_size)
        for random_state_test in trange(0, test_iters, desc='test', leave=False): #random state test
            selected_X_train, selected_X_test, selected_y_train, selected_y_test, di_test = sel.selection_mean(
                                                                                              X_train, X_test, y_train, y_test,
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


        print_to_file(dataset+"_TEST_3_PCA:"+str(n_features_pca), KNN_exps, aQKNN_exps)
        aQKNN_exps = []
        KNN_exps = []
        di_test = []

