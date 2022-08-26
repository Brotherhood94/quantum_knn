import utility.selections as sel
import math
from tqdm import trange
from tests.test_body import execute_body_basis, check_iters
from utility.save_results import print_to_file



def test_1_pca(X, y, knn_k, test_iters, train_iters, dataset, n_classes, n_features_real, training_type, test_size, enc_type, n_bits):

    pca = 'true'
    n_records_per_class=1

    bQKNN_exps = []
    bKNN_exps = []

    di_train = []
    di_test = []

    test_iters, train_iters = check_iters(X, y, n_records_per_class, n_classes, test_iters, train_iters, test_size)

    features_range = trange(1,  int(math.ceil(math.log2(n_features_real))), desc='pca')
    for n_features_pca in (2**e for e in features_range): #start from 2, exponential step
        di_test = []
        for random_state_test in trange(0, test_iters, desc='test', leave=False): #random state test
            X_train, X_test, y_train, y_test = sel._transform(X, y, n_features_pca=n_features_pca, normalize=True, standardize=True, test_size=test_size, enc_type=enc_type, n_bits=n_bits)
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
                selected_y_test = [y.loc[di_test[0]]] 

                bKNN, bQKNN = execute_body_basis(knn_k, selected_X_train, selected_X_test, selected_y_train, selected_y_test,
                                                            dataset,
                                                            n_classes,
                                                            n_features_real,
                                                            training_type,
                                                            n_records_per_class*n_classes,# len(selected_X_train),#training_size
                                                            n_features_pca, #len(selected_X_train[0]),
                                                            pca,
                                                            di_test[0], #test_id
                                                            random_state_training, #exp_id
                                                            n_bits,
                                                            enc_type
                                                            )


                bKNN_exps.append(bKNN)
                bQKNN_exps.append(bQKNN)
                

            print_to_file(dataset+"_TEST_1_PCA_"+enc_type, bKNN_exps, bQKNN_exps)
            bQKNN_exps = []
            bKNN_exps = []
            di_train = []


