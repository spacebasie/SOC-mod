import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.svm import LinearSVR, NuSVR, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt
import time

"""

    Testing Algorithms
    
                        """

# Do the error boogie
def twist_errors(algo, y_test, y_pred, start):
    print(algo)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Mean Squared Error is: %.2f"%(mse))
    print("Mean Absolute Error is: %.2f"%(mae))
    print("Mean Absolute Percentage Error is: %.2f"%(mape))
    print("Determination Coefficient R2 is: %.2f"%(r2))
    print('Training Time %s = %.1f sec'%(algo, time.time() - start))
    print('\n')
    return [mse, mae, mape, r2]

# Testing RF Regressor
def RFreg_test(X_train, X_test, y_train, y_test, n_est, min_samp_spl, max_feats, max_dep, boot):
    start = time.time()
    algo = "RF Regressor"
    # Maybe also try decision tree or extreme random forest
    pipeline = RandomForestRegressor(n_estimators=n_est, random_state=0, min_samples_split=min_samp_spl, \
        max_features=max_feats, max_depth=max_dep, bootstrap=boot)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    res = twist_errors(algo, y_test, y_pred, start)
    print('Best Parameters for %s: '%(algo))
    print('n_estimators = %.2f'%(n_est))
    print('Min Samples Split = %.2f'%(min_samp_spl))
    print('Max Features = %.2s'%(max_feats))
    print('Max Depth = ', max_dep)
    print('Bootstrap = ', boot)
    print('\n\n')
    return res


# Linear Support Vector Regressor
def linSVR_test(X_train, X_test, y_train, y_test, C_b, ker_b):
    start = time.time()
    algo = "Linear Support Vector Regressor"
    ring = make_pipeline(StandardScaler(), SVR(kernel=ker_b, C=C_b, max_iter=1e5))
    ring.fit(X_train, y_train)
    y_pred = ring.predict(X_test)
    res = twist_errors(algo, y_test, y_pred, start)
    print('Best Parameters for %s: '%(algo))
    print('C = %.2f'%(C_b))
    print('Kernel = %s'%(ker_b))
    print('\n\n')

    return res


# Stochastic Gradient Descent Regressor
def SGDR_test(X_train, X_test, y_train, y_test, bloss, bpen):
    start = time.time()    
    algo = "Stochastic Gradient Descent"
    reggy = make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, loss=bloss, penalty=bpen))
    reggy.fit(X_train, y_train)
    y_pred = reggy.predict(X_test)
    res = twist_errors(algo, y_test, y_pred, start)
    print('Best Parameters for %s: '%(algo))
    print('Loss: %s'%(bloss))
    print('Penalty: %s'%(bpen))
    print('\n\n')
    return res

# Gradient Boosting Regressor
def GBR_test(X_train, X_test, y_train, y_test, xgn_best, lossest, maxf, maxd, min_ssplit, min_sleaf):
    start = time.time()
    algo = "Gradient Boosting Regressor"
    XGB = GradientBoostingRegressor(loss=lossest, n_estimators=xgn_best, random_state=0, max_features=maxf, \
        max_depth=maxd, min_samples_leaf=min_sleaf, min_samples_split=min_ssplit)
    XGB.fit(X_train, y_train)
    y_pred = XGB.predict(X_test)
    res = twist_errors(algo, y_test, y_pred, start)
    print('Best Parameters for %s: '%(algo))
    print('n_estimators = %.2f'%(xgn_best))
    print('loss = %s'%(lossest))
    print('Max Features = ', maxf)
    print('Max Depth = ', maxd)
    print('Min Samples Split = ', min_ssplit)
    print('Min Samples Leaf = ', min_sleaf)
    print('\n\n')
    return res

# Plot errors
def show_errors(RFreg, linSVR, SGDR, GBR, ADA, Extra):
    # Set bar width and sizes
    barW = 0.15
    fig = plt.subplots(figsize=(16,12))

    # Setting positions of bars on x-axis
    b1 = np.arange(len(RFreg))
    b2 = [x + barW for x in b1]
    b3 = [x + barW for x in b2]
    b4 = [x + barW for x in b3]
    b5 = [x + barW for x in b4]
    b6 = [x + barW for x in b5]

    # Plot the bars
    plt.bar(b1, RFreg, color='lightgreen', width=barW, edgecolor = 'grey', label='RFreg')
    plt.bar(b2, linSVR, color='sandybrown', width=barW, edgecolor = 'grey', label='linSVR')
    plt.bar(b3, SGDR, color='royalblue', width=barW, edgecolor = 'grey', label='SGDR')
    plt.bar(b4, GBR, color='darkslategrey', width=barW, edgecolor = 'grey', label='GBR')
    plt.bar(b5, ADA, color='salmon', width=barW, edgecolor = 'grey', label='ADA')
    plt.bar(b6, Extra, color='yellow', width=barW, edgecolor='grey', label='ExtraTree')

    # Those x-ticks go here
    plt.xlabel('Performance Evaluators', fontweight='bold', fontsize=15)
    plt.ylabel('Algorithm Performance', fontweight='bold', fontsize=15)
    plt.xticks([r + barW for r in range(len(RFreg))], ['MSE', 'MAE', 'MAPE', 'R2'])
    # plt.grid(axis='y')
    plt.legend()
    plt.show()

# Ada Boost Regressor
def adaB_test(X_train, X_test, y_train, y_test, n_ideal, los):
    start = time.time()
    algo = "Ada Boost Regressor"
    ada = AdaBoostRegressor(loss=los, n_estimators=n_ideal, random_state=0)
    ada.fit(X_train, y_train)
    y_pred = ada.predict(X_test)
    res = twist_errors(algo, y_test, y_pred, start)
    print('Best Parameters for %s: '%(algo))
    print('Loss = %s'%(los))
    print('N_estimators = %.2f'%(n_ideal))
    print('\n\n')
    return res

# Extra Trees Regressor
def extra_test(X_train, X_test, y_train, y_test, exn_est, exmin_samp_spl, exmax_feats, exmax_dep, exboot):
    start = time.time()
    algo = "Extra Trees Regressor"
    extra =  ExtraTreesRegressor(n_estimators=exn_est, random_state=0, min_samples_split=exmin_samp_spl, \
        max_features=exmax_feats, max_depth=exmax_dep, bootstrap=exboot)
    extra.fit(X_train, y_train)
    y_pred = extra.predict(X_test)
    res = twist_errors(algo, y_test, y_pred, start)
    print('Best Parameters for %s: '%(algo))
    print('n_estimators = ', exn_est)
    print('Min Samples Split = ', exmin_samp_spl)
    print('Max Features = ', exmax_feats)
    print('Max Depth = ', exmax_dep)
    print('Bootstrap = ', exboot)
    print('\n\n')
    return res