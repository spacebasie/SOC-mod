import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, r2_score



def load_split(filepath):
    soc = pd.read_csv(filepath, header=0)
    X_soc = soc[["Catch", "Conv", "Elev", "NDVI_median"]] # MAKE THIS AS AN INPUT TO THE FUNCTION TO CHOOSE WHICH PARAMS WE TRAIN WITH
    y_soc = soc["SOC (%)"]
    X_train, X_test, y_train, y_test = train_test_split(X_soc, y_soc, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.25, random_state=1)
    return X_train, X_val, X_test, y_train, y_val, y_test

def RFreg_test(X_train, X_test, y_train, y_test):
    pipeline = RandomForestRegressor(n_estimators=100, random_state=0)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print("Mean Squared Error is: %.2f"%(mean_squared_error(y_test, y_pred)))
    print("Mean Absolute Error is: %.2f"%(mean_absolute_error(y_test, y_pred)))
    print("Determination Coefficient R2 is: %.2f"%(r2_score(y_test, y_pred)))
    return 0



def main():
    # Initial test
    """PART 1 - LOADING IN THE DATA"""
    # Load in the training, validation and test from dataset
    X_tr, X_va, X_te, y_tr, y_va, y_te = load_split('Project/SOC_Database.csv')

    """PART 2 - CHOOSING FEATURES TO TRAIN ON"""
    # Make a section where we will be testing with various different parameters in the training set (variables)
    # To obtain the optimal variables to train with

    """PART 3 - VALIDATION TESTING TO FIND THE OPTIMAL PARAMETERS FOR EACH ALGORITHM"""
    # Make a section where we will be testing with the validation set and choosing the best parameters
    # WRITE CODE

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
    RFreg_test(X_tr, X_te, y_tr, y_te)

    """PART 5 - TBD: IF ENOUGH TIME AT HAND: REPEAT TESTING IN A DIFFERENT DATASET AND COMPARE PERFORMANCE"""

if __name__ == "__main__":
    main()