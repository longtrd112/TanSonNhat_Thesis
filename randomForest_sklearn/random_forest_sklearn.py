import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from randomForest_sklearn.utils.data_preprocessing import Data1, Data3


def get_data(file_name):
    # ETA with 1 entry waypoint and with/without 2 previous data points
    if file_name == 'final_data.csv':
        data = Data1(dataFile=pd.read_csv(file_name).dropna())
    elif file_name == 'final_data_3points.csv':
        data = Data3(dataFile=pd.read_csv(file_name).dropna())
    else:
        raise Exception("Invalid data file.")

    return data


class CreateRandomForestModel:
    def __init__(self, data):
        self.data = data

        # Data after preprocessing: dropping outliers, splitting training, scaling features
        X_train, y_train, X_val, y_val, X_test, y_test = \
            data.X_train, data.y_train, data.X_val, data.y_val, data.X_test, data.y_test
        number_of_features = data.number_of_features

        # Random forest model

        # Hyperparameter tuning with HalvingGridSearchCV
        ''' 
        hyper_parameters = {
            'max_depth': [15, 16, 17, 18, 19],
            'max_features': [4, 5],
            'min_samples_leaf': [3]
        }

        grid_search = HalvingGridSearchCV(estimator=RandomForestRegressor(random_state=42, n_estimators=1000),
                                          param_grid=hyper_parameters, n_jobs=7, random_state=42, verbose=2)
        grid_search.fit(X_train, y_train.values.ravel())

        print(grid_search.best_params_)
        print(grid_search.best_score_)
          
        After running, hyper-parameters with the best score are used below.
        '''

        rfr = RandomForestRegressor(max_depth=15, max_features=4, min_samples_leaf=3, n_estimators=5000, n_jobs=6)
        rfr.fit(X_train, y_train.values.ravel())

        # Prediction
        y_predict = rfr.predict(X_test)
        y_cal = y_test.to_numpy()

        # Evaluate the metrics
        self.mae = mean_absolute_error(y_test.to_numpy(), y_predict.reshape(-1, 1))
        self.rmse = np.sqrt(mean_squared_error(y_test.to_numpy(), y_predict.reshape(-1, 1)))
        self.mape = 100 * np.mean(np.abs((y_test.to_numpy() - y_predict.reshape(-1, 1)) / np.abs(y_test.to_numpy())))
