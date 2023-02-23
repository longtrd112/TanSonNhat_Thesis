import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, HalvingGridSearchCV
from randomForest_sklearn.outliers_and_preprocessing import drop_outliers_IQR, normalizeData

df = pd.read_csv('final_features.csv')

features = ['entry_latitude', 'entry_longitude', 'entry_altitude', 'entry_ground_speed', 'entry_heading_angle',
            'distance_to_airport', 'wind_speed', 'landing_runway', 'model_type']

features_with_outliers = ['entry_altitude', 'entry_ground_speed', 'distance_to_airport', 'wind_speed']

y = pd.DataFrame(df, columns=['time_in_TMA'], index=df.index)
X = df[features]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

print(f'Size of training-validation-test set: {y_train.shape}, {y_val.shape}, {y_test.shape}')

# Preprocessing
# Drop outliers in features
for feature in features_with_outliers:
    X_train, train_drop_index_1 = drop_outliers_IQR(X_train, feature)
    y_train = y_train.drop(train_drop_index_1)

    X_val, val_drop_index_1 = drop_outliers_IQR(X_val, feature)
    y_val = y_val.drop(val_drop_index_1)

    X_test, test_drop_index_1 = drop_outliers_IQR(X_test, feature)
    y_test = y_test.drop(test_drop_index_1)

# Drop outliers in target
y_train, train_drop_index_2 = drop_outliers_IQR(y_train, 'time_in_TMA')
X_train = X_train.drop(train_drop_index_2)

y_val, val_drop_index_2 = drop_outliers_IQR(y_val, 'time_in_TMA')
X_val = X_val.drop(val_drop_index_2)

y_test, test_drop_index_2 = drop_outliers_IQR(y_test, 'time_in_TMA')
X_test = X_test.drop(test_drop_index_2)

print(f'Size of training-validation-test set after dropping outliers: {y_train.shape}, {y_val.shape}, {y_test.shape}')

column_to_scale = ['entry_latitude', 'entry_longitude', 'entry_altitude', 'entry_ground_speed', 'entry_heading_angle',
                   'distance_to_airport', 'wind_speed']
column_to_onehot = ['landing_runway']
column_to_ordinal = ['model_type']

Preprocessing_Progress = {
    'min_max_scale': [MinMaxScaler(), column_to_scale],
    'one_hot_encoding': [OneHotEncoder(sparse_output=False), column_to_onehot],
    'ordinal_encoding': [OrdinalEncoder(), column_to_ordinal]
}

for step in Preprocessing_Progress:
    func = Preprocessing_Progress[step][0]
    affected_column = Preprocessing_Progress[step][1]

    X_train = normalizeData(X_train, func, affected_column)
    X_val = normalizeData(X_val, func, affected_column)
    X_test = normalizeData(X_test, func, affected_column)

feature_columns = X_train.columns

# Build random forest model
'''
# Hyper-parameter tuning with HalvingGridSearchCV
hyper_parameters = {
    'n_estimators': [500, 1000, 2000, 5000],
    'max_depth': [3, 4, 5, 9, 10, 11],
    'max_features': [4, 5, 6, 9],
    'min_samples_leaf': [3, 4, 5, 6]
}

grid_search = HalvingGridSearchCV(estimator=RandomForestRegressor(random_state=42), 
                                  param_grid=hyper_parameters, n_jobs=7, random_state=42, verbose=2)    
grid_search.fit(X_train, y_train.values.ravel())

print(grid_search.best_params_)

After running, hyper-parameters with the best score are used below.
'''

rfr = RandomForestRegressor(max_depth=10, max_features=5, min_samples_leaf=5, n_estimators=5000,
                            random_state=42, n_jobs=4)

y_predict = rfr.predict(X_test)

# Evaluate the metrics
print('Mean Absolute Error: ', mean_absolute_error(y_test, y_predict.reshape(-1, 1)), 'seconds')
print('Root Mean Squared Error: ', np.sqrt(mean_squared_error(y_test, y_predict.reshape(-1, 1))), 'seconds')
print('Mean Absolute Percentage Error: ',
      100 * np.mean(np.abs((y_test - y_predict.reshape(-1, 1)) / np.abs(y_test))), '%')
