import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from preprocessing_data import Data1, Data3
from plot_postprocessing import PlotLoss
import warnings
warnings.filterwarnings("ignore")

file_name = 'FF3.csv'

# ETA with 1 entry waypoint and with/without 2 previous data points
if file_name == 'FF.csv':
    data = Data1(dataFile=pd.read_csv(file_name).dropna())
else:
    data = Data3(dataFile=pd.read_csv(file_name).dropna())

# Data after preprocessing: dropping outliers, splitting training, scaling features
X_train, y_train, X_val, y_val, X_test, y_test = data.X_train, data.y_train, \
                                                 data.X_val, data.y_val, \
                                                 data.X_test, data.y_test
number_of_features = data.number_of_features

for feature in data.X.columns:
    plt.hist(data.X[feature])
    plt.title(feature)
    plt.show()

# Model
max_epochs = 200
batch_size = 256
initial_learning_rate = 0.01
lr_decay_steps = min(len(X_train) // batch_size, 1000)

# Exponential decay learning rate
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=500,
    decay_rate=0.8)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=10e-4, clipnorm=1)
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, start_from_epoch=2,
                                            min_delta=400, verbose=2)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=number_of_features, activation='relu', kernel_regularizer='l1_l2'),
    tf.keras.layers.Dense(units=number_of_features//3, activation='relu', kernel_regularizer='l1_l2'),
    tf.keras.layers.Dense(units=1, activation=tf.keras.layers.LeakyReLU(alpha=0.005))
])

model.compile(loss=tf.keras.losses.mean_squared_error,
              optimizer=optimizer,
              metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.RootMeanSquaredError()])

result = model.fit(X_train, y_train, epochs=max_epochs, validation_data=(X_val, y_val),
                   batch_size=batch_size, callbacks=callback, verbose=2)
model.summary()

# Prediction
y_predict = model.predict(X_test)
model.evaluate(X_test, y_test)

print('Mean Absolute Percentage Error:', 100 * np.mean(np.abs((y_test - y_predict) / np.abs(y_test))), '%')

PlotLoss(result, y_test, y_predict)
plt.show()
