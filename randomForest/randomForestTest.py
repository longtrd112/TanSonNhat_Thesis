import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


df = pd.read_csv('final_features.csv')

y = df['time_in_TMA']

one_hot = pd.get_dummies(data=df, columns=['entry_waypoint', 'landing_runway'])
df = df.drop('entry_waypoint', axis=1)
df = df.drop('landing_runway', axis=1)

X = one_hot.drop(['flight', 'date', 'entry_time', 'arrival_time',
                  'entry_time_HCM', 'arrival_time_HCM', 'time_in_TMA'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rfr = RandomForestRegressor(n_estimators=2000, max_depth=10, random_state=42)

rfr.fit(X_train, y_train)

y_predict = rfr.predict(X_test)

ep = 0
epsilon = 1e-3
for i in range(len(y_predict)):

    error = np.abs(y_predict[i] - y_test.iloc[i])

    if y_test.iloc[i] < epsilon:
        ep += error

    else:
        ep += error / y_test.iloc[i]

MAPE = ep/len(y_predict)*100

print('Mean Absolute Error:', mean_absolute_error(y_test, y_predict), 'seconds')
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_predict)), 'seconds')
print(f'Mean Absolute Percentage Error: {MAPE} %')
