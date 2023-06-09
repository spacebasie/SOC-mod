from loadin import load_split, creem_features, load_split2
from validate_algos import RFreg_validate, linSVR_validate, SGDR_validate, GBR_validate, adaB_validate, extra_validate
from test_algos import RFreg_test, linSVR_test, SGDR_test, GBR_test, show_errors, adaB_test, extra_test

def main():

    """PART 1 - LOADING IN THE DATA"""
    # Load in the training, validation and test from dataset
    filepath = 'Project/SOC_Database.csv'
    significant_feats = creem_features(filepath)
    X_tr, X_val, X_te, y_tr, y_val, y_te = load_split2(filepath, significant_feats)
    X_train, X_test, y_train, y_test = load_split(filepath, significant_feats)
    """PART 2 - CHOOSING FEATURES TO TRAIN ON"""
    # Make a section where we will be testing with various different parameters in the training set (variables)
    # To obtain the optimal variables to train with
    # IDEAS: I think we should use the following 8-10 features and optimize:
    # Catch - Conv - Elev - LS - GCurv - Hillshade - Slope - SPI - SVF - SWI - TCurv - VDepth - NDVI_Max - NDVI_median - NDVI_sd

    """PART 3 - VALIDATION TESTING TO FIND THE OPTIMAL PARAMETERS FOR EACH ALGORITHM"""
    # Fine-tuning with different parameters using the validation set to optimize each algo
    n_est, min_samp_spl, max_feats, max_dep, boot = RFreg_validate(X_train, y_train)
    C_b, ker_b = linSVR_validate(X_tr, X_val, y_tr, y_val)
    bloss, bpen = SGDR_validate(X_tr, X_val, y_tr, y_val)
    xgn_best, lossest, maxf, maxd, min_ssplit, min_sleaf = GBR_validate(X_train, y_train)
    n_best, lossestada = adaB_validate(X_train, y_train)
    exn_est, exmin_samp_spl, exmax_feats, exmax_dep, exboot = extra_validate(X_train, y_train)

    # STEPS FOR VALIDATION TESTING TO FIND OPTIMAL PARAMETERS:
    # i) Choose algorithms that we want to test
    # ii) Identify the parameters for each algorithm that we need to optimize
    # iii) Choose an evaluation method (rmse, mae, determination coefficient, ratio of performance to deviation)
    # for iii we are now using mse and r2
    # iv) Estimate a logarithmic range of 10 values for each parameter and run loop testing combinations
    # of those parameters with eachother and downselect the combo of parameters with smallest error
    # v) Select for each algorithm the optimal parameters for the test section in part 4

    """PART 4 - RUN TEST SET FOR EACH ALGORITHM WITH THE OPTIMAL PARAMETERS"""
    # Test algorithms again with the best parameters chosen from previous section to do a direct comparison
    # Test each algorithm and get its performance/errors
    rf_err = RFreg_test(X_train, X_test, y_train, y_test, n_est, min_samp_spl, max_feats, max_dep, boot)
    linSVR_err = linSVR_test(X_tr, X_te, y_tr, y_te, C_b, ker_b)
    sgdr_err = SGDR_test(X_tr, X_te, y_tr, y_te, bloss, bpen)
    gbr_err = GBR_test(X_train, X_test, y_train, y_test, xgn_best, lossest, maxf, maxd, min_ssplit, min_sleaf)
    ada_err = adaB_test(X_train, X_test, y_train, y_test, n_best, lossestada)
    extra_err = extra_test(X_train, X_test, y_train, y_test, exn_est, exmin_samp_spl, exmax_feats, exmax_dep, exboot)

    """PART 5 - DISPLAY THE PERFORMANCE AS A GRAPH"""
    # Plot the types of errors for each algorithm to show the optimal one
    # Work on some plots for fintuning parameters, that would be nice, like https://scikit-optimize.github.io/stable/
    # Also we have not done cross-validation yet MAYBE LOOK INTO DOING 5-FOLD CROSS VALIDATION to get an average of the 
    # test scores and errors, surely that would help, though for the the validation fine-tuning just use skopt examples
    # such as from BayesSearchCV, BayesianOptimization, Tuning...
    show_errors(rf_err, linSVR_err, sgdr_err, gbr_err, ada_err, extra_err)

if __name__ == "__main__":
    main()