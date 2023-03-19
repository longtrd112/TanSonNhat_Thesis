import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from lstm_keras.utils.plot_loss import PlotLoss
from lstm_keras.utils.data_preprocessing_LSTM import Data


# Define input and output data
class CreateLSTMModel:
    def __init__(self, data_x, data_y):
        data = Data(data_x=data_x, data_y=data_y)
        seq_len = 3

        # Split data into train and test sets
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = \
            data.X_train, data.y_train, data.X_val, data.y_val, data.X_test, data.y_test

        # Model
        max_epochs = 30
        batch_size = 2048

        # Hyperparameter tuning
        optimizer = tf.keras.optimizers.Adam(epsilon=10e-4, clipnorm=1)
        plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.000001)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, min_delta=5,
                                                          start_from_epoch=3, verbose=0)

        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(units=50, activation='relu',
                                 input_shape=(seq_len, data_x.shape[-1]), kernel_regularizer='l1_l2'),
            tf.keras.layers.Dense(units=1, activation=tf.keras.layers.LeakyReLU(alpha=0.005))
        ])

        model.compile(loss=tf.keras.losses.mean_absolute_error, optimizer=optimizer,
                      metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.RootMeanSquaredError()])

        # Training
        result = model.fit(self.X_train, self.y_train, epochs=max_epochs, validation_data=(self.X_val, self.y_val),
                           batch_size=batch_size, callbacks=[early_stopping, plateau], verbose=2)
        model.summary()
        self.model = model

        # Evaluate the metrics
        model.evaluate(self.X_test, self.y_test)

        # Prediction
        y_predict = model.predict(self.X_test, verbose=0)

        # Evaluate the metrics
        self.mae = mean_absolute_error(self.y_test.reshape(-1, 1), y_predict)
        self.rmse = np.sqrt(mean_squared_error(self.y_test.reshape(-1, 1), y_predict))
        self.mape = 100 * np.mean(np.abs((self.y_test.reshape(-1, 1) - y_predict) / np.abs(self.y_test.reshape(-1, 1))))

        # Saving figures
        figure_numbering = int(time.time())
        figure_name = f"figures/figure_{figure_numbering}.png"
        PlotLoss(result=result, data=data, model=model)
        plt.savefig(figure_name, bbox_inches='tight')
        plt.clf()
