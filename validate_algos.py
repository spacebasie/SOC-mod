import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import LinearSVR, NuSVR, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, r2_score
import matplotlib.pyplot as plt

# VALIDATION FUNCTIONS
# RF REGRESSOR
def RFreg_validate(X_train, X_val, y_train, y_val):
    algo = "RF Regressor"
    # Choose variables to optimize: n_estimators, max_depth, n_informative
    n_est = [5, 10, 15, 20, 50, 60, 75, 85, 90, 95, 100, 110]
    mse_best = 100
    r2_best = 0
    n_best= 0
    for n in n_est:
        pipeline = RandomForestRegressor(n_estimators=n, random_state=0)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
        mse_curr = mean_squared_error(y_val, y_pred)
        r2_curr = r2_score(y_val, y_pred)
        if mse_curr < mse_best and r2_curr > r2_best:
            mse_best = mse_curr
            r2_best = r2_curr
            n_best = n
    return n

# LINEAR SUPPORT VECTOR REGRESSOR
def linSVR_validate(X_train, X_val, y_train, y_val):
    algo = "Linear SVR"
    toll = np.logspace(-7, -2, 10)
    mse_best = 100
    r2_best = 0
    toll_best = 0
    for te in toll:
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
def GBR_validate(X_train, X_val, y_train, y_val):
    algo = "Gradient Boosting Regressor"
    n_est = [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000, 10000000000]
    mse_best = 100
    r2_best = 0
    n_best = 0
    for n in n_est:
        XGB = GradientBoostingRegressor(loss="squared_error", n_estimators=n, random_state=0)
        XGB.fit(X_train, y_train)
        y_pred = XGB.predict(X_val)
        mse_curr = mean_squared_error(y_val, y_pred)
        r2_curr = r2_score(y_val, y_pred)
        if mse_curr < mse_best and r2_curr > r2_best:
            mse_best = mse_curr
            r2_best = r2_curr
            n_best = n
    return n_best
