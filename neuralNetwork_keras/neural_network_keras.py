import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from preprocessing_data import Data
import seaborn as sns
from sklearn.metrics import mean_absolute_error

file_name = 'final_features.csv'

# ETA with 1 entry waypoint and with/without 2 previous data points
if file_name == 'final_features.csv':
    data = Data(dataFile=pd.read_csv(file_name).dropna(), dataPoint=1)
else:
    data = Data(dataFile=pd.read_csv(file_name).dropna(), dataPoint=3)

# Data after preprocessing: dropping outliers, splitting training, scaling features
X_train, y_train, X_val, y_val, X_test, y_test = data.X_train, data.y_train, \
                                                 data.X_val, data.y_val, \
                                                 data.X_test, data.y_test
# for feature in data.X.columns:
#     plt.hist(data.X[feature])
#     plt.title(feature)
#     plt.show()
# plt.hist(data.y)
# plt.show()

# Model
batch_size = 128
initial_learning_rate = 0.005
max_epochs = 1000

# Exponential decay LR
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.9)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=10e-6, clipnorm=1)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(50, activation='relu', kernel_regularizer=tf.keras.regularizers.L1(0.001)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(20, activation='relu', kernel_regularizer=tf.keras.regularizers.L1(0.001)),
    tf.keras.layers.Dense(10, activation='relu', kernel_regularizer=tf.keras.regularizers.L1(0.001)),
    tf.keras.layers.Dense(1, activation='softplus')
])

model.compile(loss=[tf.keras.losses.mean_squared_error],
              optimizer=optimizer,
              metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.RootMeanSquaredError()])

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, start_from_epoch=5, verbose=2)

result = model.fit(X_train, y_train, epochs=max_epochs, validation_data=(X_val, y_val), verbose=2,
                   batch_size=batch_size, callbacks=callback)

y_predict = model.predict(X_test)

model.summary()

model.evaluate(X_test, y_test)

print(mean_absolute_error(y_test, y_predict))

print('Mean Absolute Percentage Error:',
      100 * np.mean(np.abs((y_test - y_predict) / np.abs(y_test))), '%')

first_plotted_epoch = 5

plt.plot(range(first_plotted_epoch, len(result.history['loss'])),
         result.history['loss'][first_plotted_epoch - 1:-1], label='loss')
plt.plot(range(first_plotted_epoch, len(result.history['loss'])),
         result.history['val_loss'][first_plotted_epoch - 1:-1], label='val loss')

plt.title("Loss vs Validation Loss (MSE)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
