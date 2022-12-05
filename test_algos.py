import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import LinearSVR, NuSVR, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, r2_score
import matplotlib.pyplot as plt

"""

    Testing Algorithms
    
                        """

# Do the error boogie
def twist_errors(algo, y_test, y_pred):
    print(algo)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Mean Squared Error is: %.2f"%(mse))
    print("Mean Absolute Error is: %.2f"%(mae))
    print("Determination Coefficient R2 is: %.2f"%(r2))
    print('\n')
    return [mse, mae, r2]

# Testing RF Regressor
def RFreg_test(X_train, X_test, y_train, y_test, n_est):
    algo = "RF Regressor"
    # Maybe also try decision tree or extreme random forest
    pipeline = RandomForestRegressor(n_estimators=n_est, random_state=0)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    res = twist_errors(algo, y_test, y_pred)
    return res


# Linear Support Vector Regressor
def linSVR_test(X_train, X_test, y_train, y_test, toll):
    algo = "Linear Support Vector Regressor"
    ring = make_pipeline(StandardScaler(), LinearSVR(random_state=0, tol=toll, max_iter=1e5))
    ring.fit(X_train, y_train)
    y_pred = ring.predict(X_test)
    res = twist_errors(algo, y_test, y_pred)
    return res


# Stochastic Gradient Descent Regressor
def SGDR_test(X_train, X_test, y_train, y_test, tole):
    algo = "Stochastic Gradient Descent"
    reggy = make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, tol= tole, penalty='l2'))
    reggy.fit(X_train, y_train)
    y_pred = reggy.predict(X_test)
    res = twist_errors(algo, y_test, y_pred)
    return res

# Gradient Boosting Regressor
def GBR_test(X_train, X_test, y_train, y_test, n_ideal):
    algo = "Gradient Boosting Regressor"
    XGB = GradientBoostingRegressor(loss="squared_error", n_estimators=n_ideal, random_state=0)
    XGB.fit(X_train, y_train)
    y_pred = XGB.predict(X_test)
    res = twist_errors(algo, y_test, y_pred)
    return res

# Plot errors
def show_errors(RFreg, linSVR, SGDR, GBR, ADA):
    # Set bar width and sizes
    barW = 0.15
    fig = plt.subplots(figsize=(12,8))

    # Setting positions of bars on x-axis
    b1 = np.arange(len(RFreg))
    b2 = [x + barW for x in b1]
    b3 = [x + barW for x in b2]
    b4 = [x + barW for x in b3]
    b5 = [x + barW for x in b4]

    # Plot the bars
    plt.bar(b1, RFreg, color='lightgreen', width=barW, edgecolor = 'grey', label='RFreg')
    plt.bar(b2, linSVR, color='sandybrown', width=barW, edgecolor = 'grey', label='linSVR')
    plt.bar(b3, SGDR, color='royalblue', width=barW, edgecolor = 'grey', label='SGDR')
    plt.bar(b4, GBR, color='darkslategrey', width=barW, edgecolor = 'grey', label='GBR')
    plt.bar(b5, ADA, color='salmon', width=barW, edgecolor = 'grey', label='ADA')

    # Those x-ticks go here
    plt.xlabel('Performance Evaluators', fontweight='bold', fontsize=15)
    plt.ylabel('Algorithm Performance', fontweight='bold', fontsize=15)
    plt.xticks([r + barW for r in range(len(RFreg))], ['MSE', 'MAE', 'R2'])
    # plt.grid(axis='y')
    plt.legend()
    plt.show()

# Ada Boost Regressor
def adaB_test(X_train, X_test, y_train, y_test, n_ideal):
    algo = "Ada Boost Regressor"
    ada = AdaBoostRegressor(loss="linear", n_estimators=n_ideal, random_state=0)
    ada.fit(X_train, y_train)
    y_pred = ada.predict(X_test)
    res = twist_errors(algo, y_test, y_pred)
    return res