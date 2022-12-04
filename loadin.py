import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import LinearSVR, NuSVR, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, r2_score


# LOADING IN DATA
def load_split(filepath):
    soc = pd.read_csv(filepath, header=0)
    X_soc = soc[["Catch", "Conv", "Elev", "LS", "Gcurv", "Hillshade", "Slope", "SPI", "SVF", "SWI", "TCurv", \
        "VDepth", "NDVI_max", "NDVI_median", "NDVI_sd"]] # MAKE THIS AS AN INPUT TO THE FUNCTION TO CHOOSE WHICH PARAMS WE TRAIN WITH
    y_soc = soc["SOC (%)"]
    X_train, X_test, y_train, y_test = train_test_split(X_soc, y_soc, test_size=0.2, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state=0)
    return X_train, X_val, X_test, y_train, y_val, y_test

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

"""

    Testing Algorithms
    
                        """

# Testing RF Regressor
def RFreg_test(X_train, X_test, y_train, y_test, n_est):
    algo = "RF Regressor"
    # Maybe also try decision tree or extreme random forest
    pipeline = RandomForestRegressor(n_estimators=n_est, random_state=0)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print(algo)
    print("Mean Squared Error is: %.2f"%(mean_squared_error(y_test, y_pred)))
    print("Mean Absolute Error is: %.2f"%(mean_absolute_error(y_test, y_pred)))
    print("Determination Coefficient R2 is: %.2f"%(r2_score(y_test, y_pred)))
    print('\n')
    return 0

# Linear Support Vector Regressor
def linSVR_test(X_train, X_test, y_train, y_test, toll):
    algo = "Linear Support Vector Regressor"
    ring = make_pipeline(StandardScaler(), LinearSVR(random_state=0, tol=toll, max_iter=1e5))
    ring.fit(X_train, y_train)
    y_pred = ring.predict(X_test)
    print(algo)
    print("Mean Squared Error is: %.2f"%(mean_squared_error(y_test, y_pred)))
    print("Mean Absolute Error is: %.2f"%(mean_absolute_error(y_test, y_pred)))
    print("Determination Coefficient R2 is: %.2f"%(r2_score(y_test, y_pred)))
    print('\n')
    return 0

# Stochastic Gradient Descent Regressor
def SGDR_test(X_train, X_test, y_train, y_test, tole):
    algo = "Stochastic Gradient Descent"
    reggy = make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, tol= tole, penalty='l2'))
    reggy.fit(X_train, y_train)
    y_pred = reggy.predict(X_test)
    print(algo)
    print("Mean Squared Error is: %.2f"%(mean_squared_error(y_test, y_pred)))
    print("Mean Absolute Error is: %.2f"%(mean_absolute_error(y_test, y_pred)))
    print("Determination Coefficient R2 is: %.2f"%(r2_score(y_test, y_pred)))
    print('\n')
    return 0

# Gradient Boosting Regressor
def GBR_test(X_train, X_test, y_train, y_test, n_ideal):
    algo = "Gradient Boosting Regressor"
    XGB = GradientBoostingRegressor(loss="squared_error", n_estimators=n_ideal, random_state=0)
    XGB.fit(X_train, y_train)
    y_pred = XGB.predict(X_test)
    print(algo)
    print("Mean Squared Error is: %.2f"%(mean_squared_error(y_test, y_pred)))
    print("Mean Absolute Error is: %.2f"%(mean_absolute_error(y_test, y_pred)))
    print("Determination Coefficient R2 is: %.2f"%(r2_score(y_test, y_pred)))
    print('\n')
    return 0

def main():
    # Initial test
    """PART 1 - LOADING IN THE DATA"""
    # Load in the training, validation and test from dataset
    X_tr, X_val, X_te, y_tr, y_val, y_te = load_split('Project/SOC_Database.csv')

    """PART 2 - CHOOSING FEATURES TO TRAIN ON"""
    # Make a section where we will be testing with various different parameters in the training set (variables)
    # To obtain the optimal variables to train with
    # IDEAS: I think we should use the following 8-10 features and optimize:
    # Catch - Conv - Elev - LS - GCurv - Hillshade - Slope - SPI - SVF - SWI - TCurv - VDepth - NDVI_Max - NDVI_median - NDVI_sd

    """PART 3 - VALIDATION TESTING TO FIND THE OPTIMAL PARAMETERS FOR EACH ALGORITHM"""
    # Make a section where we will be testing with the validation set and choosing the best parameters
    n_est = RFreg_validate(X_tr, X_val, y_tr, y_val)
    toll = linSVR_validate(X_tr, X_val, y_tr, y_val)
    tole = SGDR_validate(X_tr, X_val, y_tr, y_val)
    # n_ideal = GBR_validate(X_tr, X_val, y_tr, y_val)

    # STEPS FOR VALIDATION TESTING TO FIND OPTIMAL PARAMETERS:
    # i) Choose 5 algorithms that we want to test (we can also do more!)
    # ii) Identify the parameters for each algorithm that we need to optimize
    # iii) Choose an evaluation method (rmse, mae, determination coefficient, F1-score, ratio of performance to deviation)
    # iv) Estimate a logarithmic range of 10 values for each parameter and run loop testing combinations
    # of those parameters with eachother and downselect the combo of parameters with smallest error
    # v) Select for each algorithm the optimal parameters for the next section

    """PART 4 - RUN TEST SET FOR EACH ALGORITHM WITH THE OPTIMAL PARAMETERS"""
    # Test algorithms again with the best parameters chosen from previous section to do a direct comparison
    # Test each algorithm
    RFreg_test(X_tr, X_te, y_tr, y_te, n_est)
    linSVR_test(X_tr, X_te, y_tr, y_te, toll)
    SGDR_test(X_tr, X_val, y_tr, y_val, tole)
    GBR_test(X_tr, X_val, y_tr, y_val, n_ideal=100)

    """PART 5 - DISPLAY THE PERFORMANCE AS A GRAPH"""

    """PART 6 - TBD: IF ENOUGH TIME AT HAND: REPEAT TESTING IN A DIFFERENT DATASET AND COMPARE PERFORMANCE"""

if __name__ == "__main__":
    main()