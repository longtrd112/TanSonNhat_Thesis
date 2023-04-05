import numpy as np
import pandas as pd
import xgboost as xgb
from XGBoost_xgb.utils.data_preprocessing_xgb_val import Data1, Data3
# from XGBoost_xgb.utils.data_preprocessing_xgb import Data1, Data3
from sklearn.metrics import mean_absolute_error
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
import matplotlib.pyplot as plt


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
X_train, y_train, X_test, y_test, X_val, y_val, X, y = \
    data.X_train, data.y_train, data.X_test, data.y_test, data.X_val, data.y_val, data.X, data.y

dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_val, label=y_val)

# Define hyperparameters to search over
params = {
    'max_depth': [10, 12, 14],
    'learning_rate': [0.005, 0.01],
    'subsample': [0.3, 0.5, 0.7],
    'colsample_bytree': [0.3, 0.5, 0.7],
    'reg_alpha': [0.005, 0.01],
    'reg_lambda': [0.005, 0.01]
}
eval_set = [(X_train, y_train), (X_val, y_val)]

# Initialize XGBRegressor
xgb_model = xgb.XGBRegressor(objective='reg:absoluteerror', eval_metric='mae', n_estimators=200,
                             early_stopping_rounds=10)

# Initialize GridSearchCV
grid_search = HalvingGridSearchCV(xgb_model, param_grid=params, cv=3, scoring='neg_mean_absolute_error',
                                  n_jobs=4, verbose=2)

# Fit the GridSearchCV object
grid_search.fit(X_train, y_train, eval_set=eval_set)
print(grid_search.best_params_)

# Get the best hyperparameters
best_params = grid_search.best_params_

# Initialize XGBRegressor with best hyperparameters
xgb_model = xgb.XGBRegressor(objective='reg:absoluteerror', eval_metric='mae', **best_params)

# Fit the XGBRegressor with early stopping based on validation set performance

xgb_model.fit(X_train, y_train, eval_set=eval_set, verbose=True)

# Make predictions on the test set
y_predict = xgb_model.predict(X_test)

# Calculate MAE
mae = mean_absolute_error(y_test, y_predict)
print("Mean Absolute Error: {:.4f}".format(mae))

# Plot feature importances
fig, ax = plt.subplots(figsize=(10, 6))
xgb.plot_importance(xgb_model, ax=ax)
plt.tight_layout()
plt.show()
