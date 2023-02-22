import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.neural_network import MLPRegressor

df = pd.read_csv('final_features.csv')

df = df.drop('wind_direction', axis=1)
df = df.loc[df['time_in_TMA'] > 0]
df = df.loc[df['model'] != "0"]

y = df['time_in_TMA']

standard_scaler = StandardScaler()
le = LabelEncoder()
ohe = OneHotEncoder(sparse_output=False)

column_to_scale = ['entry_latitude', 'entry_longitude', 'entry_altitude', 'entry_ground_speed',
                   'entry_heading_angle', 'distance_to_airport', 'wind_speed']
column_to_onehot = ['landing_runway', 'model']

scaled_column = standard_scaler.fit_transform(df[column_to_scale])
encoded_column = ohe.fit_transform(df[column_to_onehot])

X = np.concatenate([scaled_column, encoded_column], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlp = MLPRegressor(random_state=42, hidden_layer_sizes=(8,), activation='relu', solver='adam', alpha=0.3,
                   max_iter=1000, n_iter_no_change=10)

mlp.fit(X_train, y_train)

y_predict = mlp.predict(X_test)

print('Mean Absolute Error:', mean_absolute_error(y_test, y_predict), 'seconds')
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_predict)), 'seconds')
mape = np.mean(np.abs((y_test - y_predict) / np.abs(y_test)))
print('Mean Absolute Percentage Error:', mape, '%')

print(mlp.best_loss_)
print(mlp.score(X_test, y_test))
# fig, (ax1, ax2) = plt.subplots(1, 2)
# cv = KFold(n_splits=3)
pd.DataFrame(mlp.loss_curve_).plot()
plt.show()

