import numpy as np
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error
from lstm_keras.utils.plot_loss import PlotLoss
from lstm_keras.utils.data_preprocessing_LSTM import Data
from lstm_keras.utils.learning_rate_schedule import scheduler


# Define input and output data
class CreateLSTMModel:
    def __init__(self, data_x, data_y):
        data = Data(data_x=data_x, data_y=data_y)
        seq_len = 3

        # Split data into train and test sets
        X_train, y_train, X_val, y_val, X_test, y_test = \
            data.X_train, data.y_train, data.X_val, data.y_val, data.X_test, data.y_test

        # Model
        max_epochs = 25
        batch_size = 8192

        # Hyperparameter tuning
        optimizer = tf.keras.optimizers.Adam(epsilon=10e-4, clipnorm=1, learning_rate=0.0005)
        lr = tf.keras.callbacks.LearningRateScheduler(scheduler)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, min_delta=10,
                                                          start_from_epoch=2, verbose=0)

        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(units=100, activation='relu',
                                 input_shape=(seq_len, data_x.shape[-1]),
                                 kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.05, l2=0.05)),
            tf.keras.layers.Dense(units=50, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2()),
            tf.keras.layers.Dense(units=25, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2()),
            tf.keras.layers.Dense(units=1, activation=tf.keras.layers.LeakyReLU(alpha=0.005))
        ])

        model.compile(loss=tf.keras.losses.mean_absolute_error, optimizer=optimizer,
                      metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.RootMeanSquaredError()])

        # Training
        training = model.fit(X_train, y_train, epochs=max_epochs, validation_data=(X_val, y_val),
                             batch_size=batch_size, callbacks=[early_stopping, lr], verbose=2)
        model.summary()
        self.model = model

        # Evaluate the metrics
        model.evaluate(X_test, y_test, verbose=2)

        # Prediction
        y_predict = model.predict(X_test, verbose=0)

        # Evaluate the metrics
        self.mae = mean_absolute_error(y_test.reshape(-1, 1), y_predict)
        self.rmse = np.sqrt(mean_squared_error(y_test.reshape(-1, 1), y_predict))
        self.mape = 100 * np.mean(np.abs((y_test.reshape(-1, 1) - y_predict) / np.abs(y_test.reshape(-1, 1))))

        # n_examples = 10
        # for i in range(n_examples):
        #     # Choose a random example from the test set
        #     example_idx = np.random.randint(0, len(X_test))
        #
        #     # Get the predicted and actual values for this example
        #     pred_value = y_predict[example_idx]
        #     actual_value = y_test[example_idx]
        #
        #     # Print out the predicted and actual values side-by-side
        #     print(f"Example {i + 1}: Predicted={pred_value}, Actual={actual_value}")

        # Saving figures
        figure_numbering = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        figure_name = f"figures/figure_{figure_numbering}.png"
        PlotLoss(training=training, test=y_test, prediction=y_predict)
        plt.savefig(figure_name, bbox_inches='tight')
        plt.clf()


# Test
# with open("sequentialData/data_x_new.pkl", "rb") as f:
#     data_x = pickle.load(f)
# with open("sequentialData/data_y_new.pkl", "rb") as f:
#     data_y = pickle.load(f)
# model = CreateLSTMModel(data_x, data_y)
