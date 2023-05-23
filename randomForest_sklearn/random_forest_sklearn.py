import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from randomForest_sklearn.utils.data_preprocessing_RF import Data1, Data3
from randomForest_sklearn.utils.plot_feature_importance import plot_feature_importance


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
        X_train, y_train, X_test, y_test = \
            data.X_train, data.y_train, data.X_test, data.y_test
        number_of_features = data.number_of_features
        print(X_train.columns)
        # Random forest model
        # Hyperparameter tuning with HalvingGridSearchCV (different for each input data set)
        if 'first_latitude' in X_train.columns:     # 3-points data
            # hyper_parameters = {
            #     'max_depth': [16, 17, 18, 19, 20, 21],
            #     'max_features': [7, 8, 9, 10, 11, 12],
            #     'min_samples_leaf': [1, 2, 3]
            # }
            hyper_parameters = {
                'max_depth': [18],
                'max_features': [10],
                'min_samples_leaf': [2]
            }

        else:                                       # 1 entry-TMA point data
            hyper_parameters = {
                'max_depth': [12, 13, 14, 15, 16],
                'max_features': [4, 5, 6, 7],
                'min_samples_leaf': [1, 2, 3]
            }

        grid_search = HalvingGridSearchCV(estimator=RandomForestRegressor(random_state=42, n_estimators=1000),
                                          param_grid=hyper_parameters, n_jobs=-1, random_state=42, verbose=1,
                                          scoring='neg_mean_absolute_error')
        grid_search.fit(X_train, y_train.values.ravel())

        print("Optimal hyperparameter value: ", grid_search.best_params_)
        print("Best grid search score: ", grid_search.best_score_)

        self.model = grid_search

        # Prediction
        y_predict = grid_search.predict(X_test)

        # Evaluate the metrics
        self.mae = mean_absolute_error(y_test.to_numpy(), y_predict.reshape(-1, 1))
        self.rmse = np.sqrt(mean_squared_error(y_test.to_numpy(), y_predict.reshape(-1, 1)))
        self.mape = 100 * np.mean(np.abs((y_test.to_numpy() - y_predict.reshape(-1, 1)) / np.abs(y_test.to_numpy())))

        # Saving figures
        # figure_numbering = int(time.time())
        # figure_name = f"figures/figure_{figure_numbering}.png"
        # plot_feature_importance(grid_search, data)
        # plt.savefig(figure_name, bbox_inches='tight')
        # plt.clf()

        print(grid_search.best_estimator_.feature_importances_)


# Test
# model_1 = CreateRandomForestModel(data=get_data(file_name='final_data.csv'), loop=False)
model_3 = CreateRandomForestModel(data=get_data(file_name='final_data_3points.csv'))
