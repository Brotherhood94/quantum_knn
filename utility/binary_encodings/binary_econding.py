from sklearn.preprocessing import LabelEncoder
from utility.binary_encodings.lsh import LSH
from utility.binary_encodings.discretizer import EntropyDiscretizer, QuartileDiscretizer, UniformDiscretizer
from sklearn.ensemble import RandomTreesEmbedding

import numpy as np
import pandas as pd


def int2binary(val, n_bits): 
    bin_str_pattern = '{:0%sb}' % n_bits
    binary = bin_str_pattern.format(val)
    return binary


def base_encode(X, y=None, n_features = None, n_bits=None, enc_type='entropy', random_seed=None, num_tables=100):

    if n_bits is None:
        n_bits = int(np.round(np.log2(X.shape[1]))) * n_features
    elif n_bits < n_features and enc_type in ['entropy', 'quartile']:
        raise Exception('Unsupported enc_type %s and n_bits < n_features' % enc_type)

    n_bits_per_feature = max(1, int(np.round(n_bits/n_features)))

    #fit
    discr = None
    lsh_encoder = None
    if enc_type == 'entropy':
        discr = EntropyDiscretizer(X, [], np.arange(X.shape[1]), y, n_bits_per_feature)
    elif enc_type == 'quartile':
        discr = QuartileDiscretizer(X, [], np.arange(X.shape[1]), y, n_bits_per_feature)
    elif enc_type == 'uniform':
        discr = UniformDiscretizer(X, [], np.arange(X.shape[1]), y, n_bits_per_feature)
    elif enc_type == 'hash':
        lsh_encoder = LSH(n_features, n_bits, num_tables=10, random_seed=random_seed)
    elif enc_type == 'forest':
        n_bits_per_feature = max(2, n_bits_per_feature)
        rte = RandomTreesEmbedding(n_estimators=n_features, max_leaf_nodes=n_bits_per_feature, random_state=random_seed)
        rte.fit(X)
    else:
        raise Exception('Unknown enc_type %s.' % enc_type)


    #transform
    if enc_type in ['entropy', 'quartile', 'uniform']:
        X = discr.discretize(X)
        X_ = [''.join([int2binary(int(v), n_bits_per_feature) for v in x]) for x in X]
    elif enc_type == 'hash':
        X_ = [lsh_encoder.get_hash_encoding(x) for x in X]
    elif enc_type == 'forest':
        X_ = rte.transform(X).toarray()
        X_ = [''.join([str(int(v)) for v in x]) for x in X_]
    else:
        raise Exception('Unknown enc_type %s.' % enc_type)
        
    y_ = y

    return X_, y_

