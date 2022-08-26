import sys
sys.path.append('..')

from tkinter import W
from sklearn.datasets import load_iris, load_breast_cancer, load_digits
from load_external_dataset import load_sonar, load_soybean_small
import numpy as np

datasets = {load_iris:'iris', load_breast_cancer:'breast_cancer', load_digits:'digits', load_sonar:'sonar', load_soybean_small:'soybean_small'}

for d_call, d_name in datasets.items():
    X, y = d_call(return_X_y=True, as_frame=True)
    knn_k = 1
    print(d_call)
    print(d_name)
    print(X)
    print(y)

