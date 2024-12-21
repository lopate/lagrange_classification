from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.gaussian_process.kernels import RBF
from tqdm import tqdm
from sklearn.decomposition import PCA
from pathlib import Path
import pickle
import torch
import numpy as np

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import cross_validate

CV = 5
def get_data(first, end, targets, experiment_name):
    id_list = range(first, end)
    x_arr = []
    for i in id_list:
        with open(Path("code/logs", f"lnn_model_{experiment_name}_{i + 1}.pickle"), 'rb') as f:
            x_arr.append(np.array(pickle.load(f)))
                
    X = np.array(x_arr)
    y = np.array(targets[first: end])
    return X, y



def get_data(first, end, targets, experiment_name):
    id_list = range(first, end)
    x_arr = []
    for i in id_list:
        with open(Path("code/logs", f"lnn_model_{experiment_name}_{i + 1}.pickle"), 'rb') as f:
            x_arr.append(np.array(pickle.load(f)))
                
    X = np.array(x_arr)
    y = np.array(targets[first: end])
    return X, y

def transform_data(X_train, X_test, componentsNumber = 17):
    pca_instance = PCA(componentsNumber)
    X_train_transformed = pca_instance.fit_transform(X_train)
    X_test_transformed = pca_instance.transform(X_test)
    return  X_train_transformed,  X_test_transformed, pca_instance

@ignore_warnings(category=ConvergenceWarning)
def compareClassifiresperPCA(starPCA, endPCA, X_train, X_test, y_train, y_test , random_state, cv = CV):
    x = []
    y_1 = []
    y_2 = []
    y_3 = []
    y_4 = []
    y_5 = []

    y_1_std = []
    y_2_std = []
    y_3_std = []
    y_4_std = []
    y_5_std = []
    #y_train_1 = []
    #y_train_2 = []
    #y_train_3 = []
    #y_train_4 = []
    for i in tqdm(range(starPCA, endPCA)):
        X_train_transformed,  X_test_transformed, pca = transform_data(X_train, X_test, i)
        
        regression = LogisticRegression(C=1.0)
        cv_results_regression = cross_validate(regression,  X_train_transformed, y_train, cv = cv)
        #regression.fit(X_train_transformed, y_train)
        y_1.append(cv_results_regression['test_score'].mean())
        y_1_std.append(cv_results_regression['test_score'].std())
        
        gpc = GaussianProcessClassifier(kernel= RBF(1000), random_state=random_state)
        cv_results_gpc = cross_validate(gpc, X_train_transformed, y_train, cv = cv)
        #gpc.fit(X_train_transformed, y_train)
        y_2.append(cv_results_gpc['test_score'].mean())
        y_2_std.append(cv_results_gpc['test_score'].std())
        
        forest = RandomForestClassifier(n_estimators=100)
        cv_results_forest = cross_validate(forest, X_train_transformed, y_train, cv = cv)
        #forest.fit(X_train_transformed, y_train)
        y_3.append(cv_results_forest['test_score'].mean())
        y_3_std.append(cv_results_forest['test_score'].std())

        knn = KNeighborsClassifier()
        cv_results_knn = cross_validate(knn, X_train_transformed, y_train, cv = cv)
        #knn.fit(X_train_transformed, y_train)
        y_4.append(cv_results_knn['test_score'].mean())
        y_4_std.append(cv_results_knn['test_score'].std())

        #svc = svm.SVC(kernel='linear')
        svc = svm.SVC(kernel=RBF(1000))
        #svc.fit(X_train_transformed, y_train)
        #y_pred = svc.predict(X_test_transformed)
        #acc_score = accuracy_score(y_pred, y_test)
        cv_results_svc  = cross_validate(svc, X_train_transformed, y_train, cv = 5)
        y_5.append(cv_results_svc['test_score'].mean())
        y_5_std.append(cv_results_svc['test_score'].std())
        #y_train_pred_1 = regression.predict(X_train_transformed)
        #acc_train_score_1 = accuracy_score(y_train, y_train_pred_1)
        #y_train_1.append(acc_train_score_1)
        #y_train_pred_2 = gpc.predict(X_train_transformed)
        #acc_train_score_2 = accuracy_score(y_train, y_train_pred_2)
        #y_train_2.append(acc_train_score_2)
        
        #y_train_pred_3 = forest.predict(X_train_transformed)
        #acc_train_score_3 = accuracy_score(y_train, y_train_pred_3)
        #y_train_3.append(acc_train_score_3)

        #y_train_pred_4 = knn.predict(X_train_transformed)
        #acc_train_score_4 = accuracy_score(y_train, y_train_pred_4)
        #y_train_4.append(acc_train_score_4)

        #y_pred_1 = regression.predict(X_test_transformed)
        #acc_score_1 = accuracy_score(y_test, y_pred_1)
        #y_1.append(acc_score_1)
        
        #y_pred_2 = gpc.predict(X_test_transformed)
        ##acc_score_2 = accuracy_score(y_test, y_pred_2)
        #y_2.append(acc_score_2)
        
        #y_pred_3 = forest.predict(X_test_transformed)
        #acc_score_3 = accuracy_score(y_test, y_pred_3)
        #y_3.append(acc_score_3)

        #y_pred_4 = knn.predict(X_test_transformed)
        #acc_score_4 = accuracy_score(y_test, y_pred_4)
        #y_4.append(acc_score_4)
        
        x.append(i)
    #return x, y_1, y_2, y_3, y_4, y_train_1, y_train_2, y_train_3, y_train_4
    x, y_1, y_2, y_3, y_4, y_5, y_1_std, y_2_std, y_3_std, y_4_std, y_5_std = np.array(x),  np.array(y_1),  np.array(y_2),  np.array(y_3),  np.array(y_4),  np.array(y_5),  np.array(y_1_std),  np.array(y_2_std),  np.array(y_3_std),  np.array(y_4_std),  np.array(y_5_std)
    return x, y_1, y_2, y_3, y_4, y_5, y_1_std, y_2_std, y_3_std, y_4_std, y_5_std