import sys
sys.path.append('..')
from sklearn.datasets import load_iris, load_breast_cancer, load_digits
import numpy as np
import pandas as pd   


def load_soybean_small(return_X_y=True, as_frame=True):
    data = pd.read_csv('./datasets/soybean-small.data', header=None)
    if return_X_y == False:
        return data
    else: #split X and y
        X = data[data.columns[:-1]]
        y = data.iloc[: , -1]
        #replacing target class 'D0, D1, D2, D3' with '0,1,2,3'
        replace_dict = dict(zip(np.unique(y), range(len(y))))
        y = y.replace(to_replace=replace_dict)
        return X, y


def load_real_estate(return_X_y=True, as_frame=True):
    data = pd.read_csv('./datasets/real-estate.data')
    if return_X_y == False:
        return data
    else: #split X and y
        X = data[data.columns[:-1]]
        #print(np.unique(X['price']))
        y = data.iloc[: , -1]
        return X, y


def load_sonar(return_X_y=True, as_frame=True):
    data = pd.read_csv('./datasets/sonar.all-data', header=None)
    if return_X_y == False:
        return data
    else: #split X and y
        X = data[data.columns[:-1]]
        y = data.iloc[: , -1]
        #replacing target class 'R,M' with '1,0'
        replace_dict = dict(zip(np.unique(y), range(len(y))))
        y = y.replace(to_replace=replace_dict)
        return X, y

# X, y = load_real_estate(return_X_y=True, as_frame=True)
