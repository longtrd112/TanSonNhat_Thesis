import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from randomForest.outliers_and_preprocessing import drop_outliers_IQR, normalizeData

df = pd.read_csv('final_features.csv')

features = ['entry_latitude', 'entry_longitude', 'entry_altitude', 'entry_ground_speed', 'entry_heading_angle',
            'distance_to_airport', 'wind_speed', 'landing_runway', 'model_type']

features_with_outliers = ['entry_altitude', 'entry_ground_speed', 'distance_to_airport', 'wind_speed']

y = pd.DataFrame(df, columns=['time_in_TMA'], index=df.index)
X = df[features]


# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

# Preprocessing
for feature in features_with_outliers:
    X_train, train_drop_index_1 = drop_outliers_IQR(X_train, feature)
    y_train = y_train.drop(train_drop_index_1)

    X_val, val_drop_index_1 = drop_outliers_IQR(X_val, feature)
    y_val = y_val.drop(val_drop_index_1)

    X_test, test_drop_index_1 = drop_outliers_IQR(X_test, feature)
    y_test = y_test.drop(test_drop_index_1)

y_train, train_drop_index_2 = drop_outliers_IQR(y_train, 'time_in_TMA')
X_train = X_train.drop(train_drop_index_2)
y_val, val_drop_index_2 = drop_outliers_IQR(y_val, 'time_in_TMA')
X_val = X_val.drop(val_drop_index_2)
y_test, test_drop_index_2 = drop_outliers_IQR(y_test, 'time_in_TMA')
X_test = X_test.drop(test_drop_index_2)

print(y_train.shape, y_val.shape, y_test.shape)

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

# Build Keras modelS
tf.random.set_seed(42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation='relu', kernel_regularizer="l2"),
    tf.keras.layers.Dense(50, activation='relu', kernel_regularizer="l2"),
    tf.keras.layers.Dense(25, activation='relu', kernel_regularizer="l1")
])

model.compile(loss=tf.keras.losses.mean_squared_error,
              optimizer=tf.keras.optimizers.Adam(clipnorm=1),
              metrics=["mae", tf.keras.metrics.RootMeanSquaredError()])

result = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), verbose=2, batch_size=128)

y_predict = model.predict(X_test)

model.summary()

try:
    print(model.evaluate(X_test, y_test))
except:
    model.evaluate(X_test, y_test)

plt.plot(result.history['loss'], label='loss')
plt.plot(result.history['val_loss'], label='val loss')
plt.title("Loss vs Val_Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
