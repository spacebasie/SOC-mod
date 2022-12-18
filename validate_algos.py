import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR, NuSVR, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
# from skopt import 

# VALIDATION FUNCTIONS
# RF REGRESSOR
def RFreg_validate(X_train, y_train):
    algo = "RF Regressor"
    # Choose variables to optimize: n_estimators, max_depth, max_features
    n_est = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    max_feat = [1.1, 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_est,
               'max_features': max_feat,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
    pipeline = RandomForestRegressor()
    rf_random = RandomizedSearchCV(estimator = pipeline, param_distributions = random_grid, \
        n_iter = 100, cv = 3, verbose=2, random_state=0, n_jobs = -1)
    rf_random.fit(X_train, y_train)
    print(rf_random.best_params_)
    n_est = rf_random.best_params_['n_estimators']
    min_samp_spl = rf_random.best_params_['min_samples_split']
    max_feats = rf_random.best_params_['max_features']
    max_dep = rf_random.best_params_['max_depth']
    boot = rf_random.best_params_['bootstrap']
    # mse_best = 100
    # r2_best = 0
    # n_best= 0
    # for n in n_est:
    #     print('Validating RFreg for n=%02d'%(n))
    #     pipeline = RandomForestRegressor(n_estimators=n, random_state=0)
    #     pipeline.fit(X_train, y_train)
    #     y_pred = pipeline.predict(X_val)
    #     mse_curr = mean_squared_error(y_val, y_pred)
    #     r2_curr = r2_score(y_val, y_pred)
    #     if mse_curr < mse_best and r2_curr > r2_best:
    #         mse_best = mse_curr
    #         r2_best = r2_curr
    #         n_best = n
    return n_est, min_samp_spl, max_feats, max_dep, boot

# LINEAR SUPPORT VECTOR REGRESSOR
def linSVR_validate(X_train, X_val, y_train, y_val):
    algo = "Linear SVR"
    # Parameters to optimize:
    # param_grid = {'C': [0.1, 1, 10, 100, 1000],
    #             'loss': ['hinge', 'squared_hinge'],
    #             'tol': np.logspace(-7, -2, 10)
    #             }
    toll = np.logspace(-7, -2, 10)
    mse_best = 100
    r2_best = 0
    toll_best = 0
    for te in toll:
        print('Validating SVR for tol=%03f'%(te))
        ring = make_pipeline(StandardScaler(), LinearSVR(random_state=0, tol=te, max_iter=1e5))
        ring.fit(X_train, y_train)
        y_pred = ring.predict(X_val)
        mse_curr = mean_squared_error(y_val, y_pred)
        r2_curr = r2_score(y_val, y_pred)
        if mse_curr < mse_best and r2_curr > r2_best:
            mse_best = mse_curr
            r2_best = r2_curr
            toll_best = te
    return toll_best

# Stochastic Gradient Descent Regressor
def SGDR_validate(X_train, X_val, y_train, y_val):
    algo = "Stochastic Gradient Descent"
    tolle = np.logspace(-7, -2, 10)
    mse_best = 100
    r2_best = 0
    tole_best = 0
    for tole in tolle:
        print('Validating SGDR for tol=%03f'%(tole))
        reggy = make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, tol= tole, penalty='l2'))
        reggy.fit(X_train, y_train)
        y_pred = reggy.predict(X_val)
        mse_curr = mean_squared_error(y_val, y_pred)
        r2_curr = r2_score(y_val, y_pred)
        if mse_curr < mse_best and r2_curr > r2_best:
            mse_best = mse_curr
            r2_best = r2_curr
            tole_best = tole
    return tole_best

# Gradient Boosting Regressor
def GBR_validate(X_train, y_train):
    algo = "Gradient Boosting Regressor"
    loss = ['squared_error', 'absolute_error', 'huber', 'quantile']
    n_est = [int(x) for x in np.linspace(start = 50, stop = 2000, num = 20)]
    max_feats = [1.0, 'sqrt', 'log2']
    max_depth = [1, 2, 3, 4, 5, 8, 12]
    min_samp_spl = [int(x) for x in np.linspace(start = 2, stop = 100, num = 10)]
    min_samp_leaf = [int(x) for x in np.linspace(start = 1, stop = 70, num = 10)]

    params = {'loss': loss,
            'n_estimators': n_est,
            'max_features': max_feats,
            'max_depth': max_depth,
            'min_samples_split': min_samp_spl,
            'min_samples_leaf': min_samp_leaf
    }
    pipe = GradientBoostingRegressor()
    xgb = RandomizedSearchCV(estimator = pipe, param_distributions = params, \
        n_iter = 100, cv = 3, verbose=2, random_state=0, n_jobs = -1)
    xgb.fit(X_train, y_train)
    n_best = xgb.best_params_['n_estimators']
    lossest = xgb.best_params_['loss']
    maxf = xgb.best_params_['max_features']
    maxd = xgb.best_params_['max_depth']
    min_ssplit = xgb.best_params_['min_samples_split']
    min_sleaf = xgb.best_params_['min_samples_leaf']
    return n_best, lossest, maxf, maxd, min_ssplit, min_sleaf



    # n_est = [10, 50, 100, 1000, 2000, 5000, 10000]
    # mse_best = 100
    # r2_best = 0
    # n_best = 0
    # for n in n_est:
    #     print('Validating GBR for n=%02d'%(n))
    #     XGB = GradientBoostingRegressor(loss="squared_error", n_estimators=n, random_state=0)
    #     XGB.fit(X_train, y_train)
    #     y_pred = XGB.predict(X_val)
    #     mse_curr = mean_squared_error(y_val, y_pred)
    #     r2_curr = r2_score(y_val, y_pred)
    #     if mse_curr < mse_best and r2_curr > r2_best:
    #         mse_best = mse_curr
    #         r2_best = r2_curr
    #         n_best = n
    # return n_best

# Ada Boost Regressor
def adaB_validate(X_train, y_train):
    algo = "Ada Boost Regressor"
    loss = ['linear', 'square', 'exponential']
    n_est = [int(x) for x in np.linspace(start = 50, stop = 2000, num = 20)]
    # max_feats = [1.0, 'sqrt', 'log2']
    # max_depth = [1, 2, 3, 4, 5, 8, 12]
    # min_samp_spl = [int(x) for x in np.linspace(start = 2, stop = 100, num = 10)]
    # min_samp_leaf = [int(x) for x in np.linspace(start = 1, stop = 70, num = 10)]

    params = {'loss': loss,
            'n_estimators': n_est,
            # 'max_features': max_feats,
            # 'max_depth': max_depth,
            # 'min_samples_split': min_samp_spl,
            # 'min_samples_leaf': min_samp_leaf
    }
    # pipe = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth))
    ada = RandomizedSearchCV(estimator = AdaBoostRegressor(),\
        param_distributions = params, n_iter = 100, cv = 3, verbose=2, random_state=0, n_jobs = -1)
    ada.fit(X_train, y_train)
    n_best = ada.best_params_['n_estimators']
    lossestada = ada.best_params_['loss']
    # maxf = ada.best_params_['max_features']
    # maxda = ada.best_params_['max_depth']
    # min_ssplit = ada.best_params_['min_samples_split']
    # min_sleaf = ada.best_params_['min_samples_leaf']
    return n_best, lossestada#, maxd


    # mse_best = 100
    # r2_best = 0
    # n_best = 0
    # n_est = [10, 50, 100, 1000, 5000, 10000]
    # for n in n_est:
    #     print('Validating ADA for n=%02d'%(n))
    #     ada = AdaBoostRegressor(loss="linear", n_estimators=n, random_state=0)
    #     ada.fit(X_train, y_train)
    #     y_pred = ada.predict(X_val)
    #     mse_curr = mean_squared_error(y_val, y_pred)
    #     r2_curr = r2_score(y_val, y_pred)
    #     if mse_curr < mse_best and r2_curr > r2_best:
    #         mse_best = mse_curr
    #         r2_best = r2_curr
    #         n_best = n
    # return n_best

# Extra Trees Regressor
def extra_validate(X_train, y_train):
    algo = "Extra Tree Regressor"
    algo = "RF Regressor"
    # Choose variables to optimize: n_estimators, max_depth, max_features
    n_est = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    max_feat = [1.0, 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_est,
               'max_features': max_feat,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
    pipeline = ExtraTreesRegressor()
    extrarf_random = RandomizedSearchCV(estimator = pipeline, param_distributions = random_grid, \
        n_iter = 100, cv = 3, verbose=2, random_state=0, n_jobs = -1)
    extrarf_random.fit(X_train, y_train)
    print(extrarf_random.best_params_)
    n_est = extrarf_random.best_params_['n_estimators']
    min_samp_spl = extrarf_random.best_params_['min_samples_split']
    max_feats = extrarf_random.best_params_['max_features']
    max_dep = extrarf_random.best_params_['max_depth']
    boot = extrarf_random.best_params_['bootstrap']
    # mse_best = 100
    # r2_best = 0
    # n_best = 0
    # n_est = [10, 50, 75, 90, 100, 150, 200, 500, 675, 750, 900, 1000, 2500]
    # mse_best = 100
    # r2_best = 0
    # n_best= 0
    # for n in n_est:
    #     print('Validating RFreg for n=%02d'%(n))
    #     pipeline = ExtraTreesRegressor(n_estimators=n, random_state=0)
    #     pipeline.fit(X_train, y_train)
    #     y_pred = pipeline.predict(X_val)
    #     mse_curr = mean_squared_error(y_val, y_pred)
    #     r2_curr = r2_score(y_val, y_pred)
    #     if mse_curr < mse_best and r2_curr > r2_best:
    #         mse_best = mse_curr
    #         r2_best = r2_curr
    #         n_best = n
    return n_est, min_samp_spl, max_feats, max_dep, boot