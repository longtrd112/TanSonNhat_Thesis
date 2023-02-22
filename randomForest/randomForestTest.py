import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn import metrics

df = pd.read_csv('final_features.csv')
df = df.dropna(how='any', axis=0)
df = df.loc[df['time_in_TMA'] > 0]

y = df['time_in_TMA']

one_hot = pd.get_dummies(data=df, columns=['landing_runway'])
# df = df.drop('entry_waypoint', axis=1)
df = df.drop('landing_runway', axis=1)

X = one_hot.drop(['flight', 'date', 'entry_time', 'arrival_time',
                  'entry_time_HCM', 'arrival_time_HCM', 'time_in_TMA', 'entry_waypoint'
                  ], axis=1)

features = list(X.columns)
print(one_hot)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

parameters = {
    'n_estimators': [800, 900, 1000, 2000, 3000, 5000, 10000],
    'max_depth': [8, 9, 10, 11, 12, 13, 14],
    'max_features': ['sqrt', 'log2', 'None', 3, 4],
    'min_samples_leaf': [4, 5, 6]
}

rfr = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rfr, param_grid=parameters, cv=KFold(n_splits=5, shuffle=True, random_state=1))
grid_search.fit(X_train, y_train)

y_predict = grid_search.predict(X_test)

print('Mean Absolute Error:', mean_absolute_error(y_test, y_predict), 'seconds')
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_predict)), 'seconds')
mape = np.mean(np.abs((y_test - y_predict) / np.abs(y_test)))
print('Mean Absolute Percentage Error:', mape, '%')

#
# fig, (ax1, ax2) = plt.subplots(1, 2)
# cv = KFold(n_splits=3)

# plt.barh(features, rfr.feature_importances_)
# plt.show()
#
# learning_curves(rfr, X, y, ax=ax1, cv=cv, train_sizes=np.linspace(0.1, 1, 5))
# learning_curves(rfr, X, y, ax=ax2, cv=cv, train_sizes=np.linspace(0.5, 1, 5))

# plt.show()
