import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# LOADING IN DATA
def load_split(filepath):
    soc = pd.read_csv(filepath, header=0)
    X_soc = soc[["Catch", "Conv", "Elev", "LS", "Gcurv", "Hillshade", "Slope", "SPI", "SVF", "SWI", "TCurv", \
        "VDepth", "NDVI_max", "NDVI_median", "NDVI_sd"]] # MAKE THIS AS AN INPUT TO THE FUNCTION TO CHOOSE WHICH PARAMS WE TRAIN WITH
    y_soc = soc["SOC (%)"]
    X_train, X_test, y_train, y_test = train_test_split(X_soc, y_soc, test_size=0.2, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state=0)
    return X_train, X_val, X_test, y_train, y_val, y_test







