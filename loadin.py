import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# LOADING IN DATA
def load_split(filepath, feats):
    soc = pd.read_csv(filepath, header=0)
    # X_soc = soc[["Catch", "Conv", "Elev", "LS", "Gcurv", "Hillshade", "Slope", "SPI", "SVF", "SWI", "TCurv", \
    #     "VDepth", "NDVI_max", "NDVI_median", "NDVI_sd"]] # MAKE THIS AS AN INPUT TO THE FUNCTION TO CHOOSE WHICH PARAMS WE TRAIN WITH
    # Test with all variables of positive correlation:
    X_soc = soc[feats]
    y_soc = soc["SOC (%)"]
    X_train, X_test, y_train, y_test = train_test_split(X_soc, y_soc, test_size=0.2, random_state=0)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state=0)
    # return X_train, X_val, X_test, y_train, y_val, y_test
    return X_train, X_test, y_train, y_test

def load_split2(filepath, feats):
    soc = pd.read_csv(filepath, header=0)
    # X_soc = soc[["Catch", "Conv", "Elev", "LS", "Gcurv", "Hillshade", "Slope", "SPI", "SVF", "SWI", "TCurv", \
    #     "VDepth", "NDVI_max", "NDVI_median", "NDVI_sd"]] # MAKE THIS AS AN INPUT TO THE FUNCTION TO CHOOSE WHICH PARAMS WE TRAIN WITH
    # Test with all variables of positive correlation:
    X_soc = soc[feats]
    y_soc = soc["SOC (%)"]
    X_train, X_test, y_train, y_test = train_test_split(X_soc, y_soc, test_size=0.2, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state=0)
    return X_train, X_val, X_test, y_train, y_val, y_test

def creem_features(filepath):
    soc = pd.read_csv(filepath, header=0)
    y_soc = soc["SOC (%)"]
    X_soc = soc.drop(columns=["SampleID","SOC (%)", "Y (DD)", "X (DD)"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X_soc, y_soc, test_size=0.2, random_state=0)
    from sklearn.ensemble import RandomForestRegressor
    rf_creem = RandomForestRegressor(n_estimators=2500, random_state=0)
    rf_creem.fit(X_train, y_train)
    significance = list(rf_creem.feature_importances_)
    feature_importances = [(feature, round(importance,2)) for feature,
    importance in zip(X_soc, significance)]

    feature_importances = sorted(feature_importances, key=lambda x:
    x[1], reverse=True)
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

    # Plotting the Variable Importances to Select features:
    x_values = list(range(len(significance)))

    # Make bar chart
    import matplotlib.pyplot as plt
    plt.bar(x_values, significance, orientation='vertical', color='blue', \
        edgecolor='grey', linewidth=1.2)
    plt.xticks(x_values, X_soc, rotation='vertical')
    plt.ylabel('Importance'); plt.xlabel('Features'); plt.title('Feature Selection by Importance')
    plt.show()

    # Get features with significance >=0.2
    features = []
    for significance in feature_importances:
        if significance[1] >= 0.02:
            features.append(significance[0])
    
    return features