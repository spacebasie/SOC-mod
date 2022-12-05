from loadin import load_split
from validate_algos import RFreg_validate, linSVR_validate, SGDR_validate, GBR_validate
from test_algos import RFreg_test, linSVR_test, SGDR_test, GBR_test, show_errors

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
    rf_err = RFreg_test(X_tr, X_te, y_tr, y_te, n_est)
    linSVR_err = linSVR_test(X_tr, X_te, y_tr, y_te, toll)
    sgdr_err = SGDR_test(X_tr, X_val, y_tr, y_val, tole)
    gbr_err = GBR_test(X_tr, X_val, y_tr, y_val, n_ideal=100)

    """PART 5 - DISPLAY THE PERFORMANCE AS A GRAPH"""
    # Start working on this to show a few plots in our presentation
    # As well as plots for fintuning parameters would be nice
    show_errors(rf_err, linSVR_err, sgdr_err, gbr_err)

    """PART 6 - TBD: IF ENOUGH TIME AT HAND: REPEAT TESTING IN A DIFFERENT DATASET AND COMPARE PERFORMANCE"""

if __name__ == "__main__":
    main()