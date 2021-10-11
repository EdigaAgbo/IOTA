from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
# from missingpy import MissForest
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge
from functools import reduce
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# The code below will import the dataset from the directory containing
# the csv and .py... NOTE dataset would be loaded with pre-simulated missing data
# represented distinct values which would usually be 0 and sometimes (-1)
# on one column (1)

def KNN_imputation(X):
    imputer = KNNImputer(n_neighbors=2, weights="uniform",missing_values=np.nan)
    imputer.fit(X)
    KNN_imputed = imputer.fit_transform(X)
    return KNN_imputed

def mean_imputation(X):
    mean = SimpleImputer(missing_values =np.nan, strategy='mean')
    mean.fit(X)
    mean_imputed = mean.transform(X)
    return mean_imputed

def mode_imputation(X):
    mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    mode.fit(X)
    mode_imputed = mode.transform(X)
    return mode_imputed

def rf_imputation(X):
    imputer = MissForest(random_state=1337, missing_values=np.nan)
    rf_imputed = imputer.fit_transform(X)
    return rf_imputed

def reg_imputation(X):
    imputer = IterativeImputer(BayesianRidge())
    reg_imputed = imputer.fit_transform(X)
    return reg_imputed

def impute_em(X, max_iter=3000, eps=1e-08):
    nr, nc = X.shape
    C = np.isnan(X) == False

    # Collect M_i and O_i's
    one_to_nc = np.arange(1, nc + 1, step=1)
    M = one_to_nc * (C == False) - 1
    O = one_to_nc * C - 1

    # Generate Mu_0 and Sigma_0
    Mu = np.nanmean(X, axis=0)
    observed_rows = np.where(np.isnan(sum(X.T)) == False)[0]
    S = np.cov(X[observed_rows,].T)
    if np.isnan(S).any():
        S = np.diag(np.nanvar(X, axis=0))
    # Start updating
    Mu_tilde, S_tilde = {}, {}
    X_tilde = X.copy()
    no_conv = True
    iteration = 0
    while no_conv and iteration < max_iter:
        for i in range(nr):
            S_tilde[i] = np.zeros(nc ** 2).reshape(nc, nc)
            if set(O[i,]) != set(one_to_nc - 1):  # missing component exists
                M_i, O_i = M[i,][M[i,] != -1], O[i,][O[i,] != -1]
                S_MM = S[np.ix_(M_i, M_i)]
                S_MO = S[np.ix_(M_i, O_i)]
                S_OM = S_MO.T
                S_OO = S[np.ix_(O_i, O_i)]
                Mu_tilde[i] = Mu[np.ix_(M_i)] +\
                    S_MO @ np.linalg.inv(S_OO) @\
                    (X_tilde[i, O_i] - Mu[np.ix_(O_i)])
                X_tilde[i, M_i] = Mu_tilde[i]
                S_MM_O = S_MM - S_MO @ np.linalg.inv(S_OO) @ S_OM
                S_tilde[i][np.ix_(M_i, M_i)] = S_MM_O
        Mu_new = np.mean(X_tilde, axis=0)
        S_new = np.cov(X_tilde.T, bias=1) +\
                reduce(np.add, S_tilde.values()) / nr
        no_conv = \
            np.linalg.norm(Mu - Mu_new) >= eps or \
            np.linalg.norm(S - S_new, ord=2) >= eps
        Mu = Mu_new
        S = S_new
        iteration += 1

    result = {
        'mu': Mu,
        'Sigma': S,
        'X_imputed': X_tilde,
        'C': C,
        'iteration': iteration
    }
    return result

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def nrmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())/(sum(targets)/len(targets))

def r_squared(y_pred, y_true):
    R_square = r2_score(y_pred, y_true)
    return R_square

def MAE (y_pred, y_true):
    error = mean_absolute_error(y_pred, y_true)
    return error