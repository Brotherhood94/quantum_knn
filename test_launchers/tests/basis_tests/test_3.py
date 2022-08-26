import utility.selections as sel
import math
from tqdm import trange
from utility.save_results import print_to_file
from tests.test_body import execute_body_basis, check_iters



### TEST_3_PCA

def test_3_pca(X, y, knn_k, test_iters, dataset, n_classes, n_features_real, training_type, test_size, enc_type, n_bits):

    bQKNN_exps = []
    bKNN_exps = []

    di_test = []

    pca = 'true'

    test_iters, _ = check_iters(X, y, 1, 1, test_iters, 1, test_size)
    features_range = trange(2,  int(math.ceil(math.log2(n_features_real))), desc='pca')
    for n_features_pca in (2**e for e in features_range): #start from 2, exponential step
        X_train, X_test, y_train, y_test = sel._transform(X, y, n_features_pca=n_features_pca, normalize=True, standardize=True, test_size=test_size, enc_type=enc_type, n_bits=n_bits)
        for random_state_test in trange(0, test_iters, desc='test', leave=False): #random state test
            selected_X_train, selected_X_test, selected_y_train, selected_y_test, di_test = sel.basis_selection_mean(
                                                                                              X_train, X_test, y_train, y_test,
                                                                                              random_state_test=random_state_test,
                                                                                              discard_index_test=di_test)

            selected_y_test = [y.loc[di_test[0]]] 


            bKNN, bQKNN = execute_body_basis(knn_k, selected_X_train, selected_X_test, selected_y_train, selected_y_test,
                                                    dataset,
                                                    n_classes,
                                                    n_features_real,
                                                    training_type,
                                                    n_classes,# len(selected_X_train),#training_size
                                                    n_features_pca, #len(selected_X_train[0]),
                                                    pca,
                                                    di_test[0], #test_id
                                                    0,#exp_id
                                                    n_bits,
                                                    enc_type
                                                    )

            bQKNN_exps.append(bQKNN)
            bKNN_exps.append(bKNN)


        print_to_file(dataset+"_TEST_3_PCA:"+str(n_features_pca)+"_"+enc_type, bKNN_exps, bQKNN_exps)
        bQKNN_exps = []
        bKNN_exps = []
        di_test = []

