from sklearn.datasets import load_iris, load_breast_cancer, load_digits
import utility.selections as sel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time


#------------ Datasets ----------
X, y = load_iris(return_X_y=True, as_frame=True)
#X, y = load_breast_cancer(return_X_y=True, as_frame=True)
#X, y = load_digits(return_X_y=True, as_frame=True)
#---------------------------------

#------- If too expensive the simulation of the QKNN on all the classes => filter classes. 
# comb = [0, 8] #Keeping only classes "0" and "8"
# X = X.drop(y[(y!=comb[0]) & (y != comb[1])].index)
# y = y[(y == comb[0]) | (y == comb[1])]
#---------------------------------

#------ Set up -------------------
knn_k = 1 
test_size = 0.3
X_train, X_test, y_train, y_test = sel._transform(X, y, n_features_pca=None, normalize=True, standardize=True, test_size=test_size)
#---------------------------------


# KNN Instance
classical_knn = KNeighborsClassifier(n_neighbors=knn_k)

#KNN fit
classical_knn.fit(X_train, y_train)

# Arrays storing results
ground_truth = []
c_predictions = []
pred_times = []

for y_i in y_test:
    ground_truth.append(y_i)

for index, x_row in X_test.iterrows():
    start_fit = time.time()
    classic_pred = classical_knn.predict([x_row]) #KNN Predict
    end_fit = time.time()
    pred_times.append(end_fit - start_fit)
    c_predictions.append(classic_pred)


#---------------------------------
classic_accuracy_train = accuracy_score(ground_truth, c_predictions)
print("Classical KNN Accuracy: {}".format(classic_accuracy_train))
print("Preditions Time Avg: {}".format(sum(pred_times)/len(pred_times)))





