import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from neuralNetwork_keras.outliers_and_preprocessing import drop_outliers_IQR, normalizeData, target_scale

df = pd.read_csv('final_features_3points.csv')

# Fill NaN visibility value
df = df.fillna(method='ffill')

# 19 features
features = ['first_latitude', 'first_longitude', 'first_altitude', 'first_ground_speed', 'first_heading_angle',
            'second_latitude', 'second_longitude', 'second_altitude', 'second_ground_speed', 'second_heading_angle',
            'entry_latitude', 'entry_longitude', 'entry_altitude', 'entry_ground_speed', 'entry_heading_angle',
            'wind_speed', 'visibility', 'landing_runway', 'model_type']

y = pd.DataFrame(df, columns=['time_in_TMA'], index=df.index)
X = df[features]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

print(f'Size of training-validation-test set: {y_train.shape}, {y_val.shape}, {y_test.shape}')

# Preprocessing
# Drop outliers in features
features_with_outliers = ['first_altitude', 'first_ground_speed',
                          'second_altitude', 'second_ground_speed',
                          'entry_altitude', 'entry_ground_speed']

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


column_to_scale = [feature for feature in features if feature not in ('landing_runway', 'model_type')]
# column_to_scale = ['first_latitude', 'first_longitude', 'first_altitude',
#                    'first_ground_speed', 'first_heading_angle',
#
#                    'second_latitude', 'second_longitude', 'second_altitude',
#                    'second_ground_speed', 'second_heading_angle',
#
#                    'entry_latitude', 'entry_longitude', 'entry_altitude',
#                    'entry_ground_speed', 'entry_heading_angle',
#
#                    'wind_speed', 'visibility']
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


# Build Keras model
tf.random.set_seed(42)
batch_size = 128
initial_learning_rate = 0.01
epochs = 1000

# Exponential decay LR
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.9)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(50, activation='relu', kernel_regularizer="l2"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(20, activation='relu', kernel_regularizer="l1"),
    tf.keras.layers.Dense(10, activation='relu', kernel_regularizer="l1"),
    tf.keras.layers.Dense(1, activation='linear')
])

model.compile(loss=tf.keras.losses.mean_squared_error,
              optimizer=optimizer,
              metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.RootMeanSquaredError()])

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, start_from_epoch=50, verbose=1)

result = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), verbose=2,
                   batch_size=batch_size, workers=6, callbacks=callback)

y_predict = model.predict(X_test)

model.summary()

print(model.evaluate(X_test, y_test))

print('Mean Absolute Percentage Error:',
      100 * np.mean(np.abs((y_test - y_predict) / np.abs(y_test))), '%')

first_plotted_epoch = 10

plt.plot(range(first_plotted_epoch, len(result.history['loss'])),
         result.history['loss'][first_plotted_epoch - 1:-1], label='loss')
plt.plot(range(first_plotted_epoch, len(result.history['loss'])),
         result.history['val_loss'][first_plotted_epoch - 1:-1], label='val loss')
plt.title("Loss vs Validation Loss (MSE)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
