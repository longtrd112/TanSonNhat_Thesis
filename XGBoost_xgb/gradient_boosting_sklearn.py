import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
# from XGBoost_xgb.utils.data_preprocessing_xgb_val import Data1, Data3
from XGBoost_xgb.utils.data_preprocessing_xgb import Data1, Data3
from sklearn.metrics import mean_absolute_error
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV


def get_data(file_name):
    # ETA with 1 entry waypoint and with/without 2 previous data points
    if file_name == 'final_data.csv':
        data = Data1(dataFile=pd.read_csv(file_name).dropna())

    elif file_name == 'final_data_3points.csv':
        data = Data3(dataFile=pd.read_csv(file_name).dropna())

    else:
        raise Exception("Invalid data file.")

    return data


data = get_data('final_data.csv')
# X_train, y_train, X_test, y_test, X_val, y_val, X, y = \
#     data.X_train, data.y_train, data.X_test, data.y_test, data.X_val, data.y_val, data.X, data.y
X_train, y_train, X_test, y_test, X, y = data.X_train, data.y_train, data.X_test, data.y_test, data.X, data.y

hyper_parameters = {
    'max_depth': [12, 13, 14],
    'max_features': ['log2', None],
    'min_samples_leaf': [2, 3, 4]
}
grb = GradientBoostingRegressor(n_estimators=300,
                                learning_rate=0.02,
                                loss='absolute_error',
                                validation_fraction=0.2,
                                n_iter_no_change=10, tol=5)

grid_search = HalvingGridSearchCV(estimator=grb,
                                  param_grid=hyper_parameters, n_jobs=-1, random_state=42, verbose=1,
                                  scoring='neg_mean_absolute_error')
grid_search.fit(X_train, y_train.values.ravel())

print("Optimal hyperparameter value: ", grid_search.best_params_)
print("Best grid search score: ", grid_search.best_score_)

# model = GradientBoostingRegressor(n_estimators=300,
#                                   learning_rate=0.01,
#                                   loss='absolute_error',
#                                   validation_fraction=0.2,
#                                   n_iter_no_change=10, tol=5)
# model.fit(X_train, y_train.values.ravel())
# Make predictions on the testing set
y_predict = grid_search.predict(X_test)

# Calculate the mean squared error of the model
mae = mean_absolute_error(y_test, y_predict)
print(mae)
