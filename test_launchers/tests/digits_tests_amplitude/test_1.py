import utility.selections as sel
from tqdm import trange
from tests.test_body import execute_body, check_iters
from utility.save_results import print_to_file

def test_1_no_pca(X, y, knn_k, test_iters, train_iters, dataset, n_classes, n_features_real, training_type, test_size):
    #TEST 1
    # 1 test istance, 1 istance per class training
    # 100 Random Selection on Training, 100 Random Selection on Test
    pca = 'false'
    n_records_per_class = 1


    ##### TEST 1 No PCA #####

    di_test = []
    di_train = []

    aQKNN_exps = []
    KNN_exps = []

    test_iters, train_iters = check_iters(X, y, n_records_per_class, n_classes, test_iters, train_iters, test_size)

    for random_state_test in trange(0, test_iters, desc='test'): #random state test
        X_train, X_test, y_train, y_test = sel._transform(X, y, n_features_pca=8, normalize=True, standardize=True, test_size=test_size)
        if len(di_test):
            X_test.drop(di_test, inplace=True)
            y_test.drop(di_test, inplace=True)
        for random_state_training in trange(0,train_iters, desc='training', leave=False): #random state training
            selected_X_train, selected_X_test, selected_y_train, selected_y_test, di_train, di_test = sel.selection_random(
                                                                                                    X_train, X_test, y_train, y_test,
                                                                                                    n_records_per_class=n_records_per_class,
                                                                                                    random_state_training=random_state_training,
                                                                                                    random_state_test=random_state_test,
                                                                                                    discard_index_train=di_train,
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
                                                            random_state_training #exp_id
                                                            )

            
            aQKNN_exps.append(aQKNN)
            KNN_exps.append(KNN)


        print_to_file(dataset+"_TEST_1_No_PCA", KNN_exps, aQKNN_exps)
        aQKNN_exps = []
        KNN_exps = []
        di_train = []


def test_1_pca(X, y, knn_k, test_iters, train_iters, dataset, n_classes, n_features_real, training_type, test_size):

    pca = 'true'
    n_records_per_class=1

    aQKNN_exps = []
    KNN_exps = []

    di_train = []
    di_test = []

    test_iters, train_iters = check_iters(X, y, n_records_per_class, n_classes, test_iters, train_iters, test_size)

    features_range = trange(1,  4, desc='pca')
    for n_features_pca in (2**e for e in features_range): #start from 2, exponential step
        print(n_features_pca)
        di_test = []
        for random_state_test in trange(0,test_iters, desc='test', leave=False): #random state test
            X_train, X_test, y_train, y_test = sel._transform(X, y, n_features_pca=n_features_pca, normalize=True, standardize=True, test_size=test_size)
            if len(di_test):
                X_test.drop(di_test, inplace=True)
                y_test.drop(di_test, inplace=True)
            for random_state_training in trange(0,train_iters, desc='training', leave=False): #random state training
                selected_X_train, selected_X_test, selected_y_train, selected_y_test, di_train, di_test = sel.selection_random(
                                                                                                        X_train, X_test, y_train, y_test,
                                                                                                        n_records_per_class=n_records_per_class,
                                                                                                        random_state_training=random_state_training,
                                                                                                        random_state_test=random_state_test,
                                                                                                        discard_index_train=di_train,
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
                                                            random_state_training #exp_id
                                                            )


                aQKNN_exps.append(aQKNN)
                KNN_exps.append(KNN)


                

            print_to_file(dataset+"_TEST_1_PCA", KNN_exps, aQKNN_exps)
            aQKNN_exps = []
            KNN_exps = []
            di_train = []


