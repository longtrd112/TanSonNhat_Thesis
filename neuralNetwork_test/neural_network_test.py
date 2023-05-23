import tensorflow as tf
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from neuralNetwork_test.utils.data_preprocessing_NN import Data1, Data3
from neuralNetwork_test.utils.plot_loss import PlotLoss
from neuralNetwork_keras.utils.plot_feature_importance import plot_feature_importance


def get_data(file_name):
    # ETA with 1 entry waypoint and with/without 2 previous data points
    if file_name == 'final_data.csv':
        data = Data1(dataFile=pd.read_csv(file_name).dropna())
    elif file_name == 'final_data_3points.csv':
        data = Data3(dataFile=pd.read_csv(file_name).dropna())
    else:
        raise Exception("Invalid data file.")

    return data


class CreateNeuralNetworkModel:
    def __init__(self, data):
        self.data = data

        # Data after preprocessing: dropping outliers, splitting training, scaling features
        X_train, y_train, X_val, y_val, X_test, y_test, X, y = \
            data.X_train, data.y_train, data.X_val, data.y_val, data.X_test, data.y_test, data.X, data.y

        X_flight_train, X_flight_val, X_flight_test, X_weather_train, X_weather_val, X_weather_test = \
            data.X_flight_train, data.X_flight_val, data.X_flight_test, \
            data.X_weather_train, data.X_weather_val, data.X_weather_test

        number_of_flight_features = len(X_flight_train.columns)
        number_of_weather_features = len(X_weather_train.columns)

        # # Features histogram
        # for feature in data.X.columns:
        #     plt.hist(data.X[feature])
        #     plt.title(feature)
        #     plt.show()

        # Model
        max_epochs = 100
        batch_size = 64

        # Hyperparameter tuning
        optimizer = tf.keras.optimizers.Adam(epsilon=10e-4, clipnorm=1)
        plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.000001)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=20, verbose=0)

        flight_input = tf.keras.layers.Input(shape=(number_of_flight_features,))
        weather_input = tf.keras.layers.Input(shape=(number_of_weather_features,))

        # Define hidden layers for flight features
        x = tf.keras.layers.Dense(100, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2())(flight_input)
        x = tf.keras.layers.Dense(50, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2())(x)
        flight_output = tf.keras.layers.Dense(number_of_flight_features, activation='relu')(x)

        # Define hidden layers for weather features
        y = tf.keras.layers.Dense(3, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2())(weather_input)
        weather_output = tf.keras.layers.Dense(1, activation='relu')(y)

        combined_input = tf.keras.layers.concatenate([flight_output, weather_output])

        # Define hidden layers for combined input
        z = tf.keras.layers.Dense(32, activation='relu')(combined_input)
        z = tf.keras.layers.Dense(16, activation='relu')(z)

        # Define output layer for ETA prediction
        eta_output = tf.keras.layers.Dense(1, activation=tf.keras.layers.LeakyReLU(alpha=0.005))(z)

        # model = tf.keras.Sequential([
        #     tf.keras.layers.Dense(units=100, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(),
        #                           input_shape=(number_of_features,)),
        #     tf.keras.layers.Dense(units=50, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2()),
        #     tf.keras.layers.Dense(units=1, activation=tf.keras.layers.LeakyReLU(alpha=0.005))
        # ])

        # Define the model with multiple inputs and outputs
        model = tf.keras.models.Model(inputs=[flight_input, weather_input], outputs=eta_output)

        model.compile(loss=tf.keras.losses.mean_absolute_error, optimizer=optimizer,
                      metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.RootMeanSquaredError()])

        # Training
        training = model.fit([X_flight_train, X_weather_train], y_train,
                             validation_data=([X_flight_val, X_weather_val], y_val),
                             epochs=max_epochs, batch_size=batch_size, callbacks=[plateau, early_stopping], verbose=2)
        model.summary()
        self.model = model

        # Prediction
        y_predict = model.predict([X_flight_test, X_weather_test], verbose=0)
        model.evaluate([X_flight_test, X_weather_test], y_test)

        # Evaluate the metrics
        self.mae = mean_absolute_error(y_test, y_predict)
        self.rmse = np.sqrt(mean_squared_error(y_test, y_predict))
        self.mape = 100 * np.mean(np.abs((y_test.to_numpy() - y_predict) / np.abs(y_test.to_numpy())))

        # Saving figures
        figure_numbering = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        figure_name = f"figures/figure_{figure_numbering}.png"
        PlotLoss(training=training, test=y_test, prediction=y_predict)
        plt.savefig(figure_name, bbox_inches='tight')
        plt.clf()

        # # Plot feature importance
        # plot_feature_importance(model, X, y)
        # plt.show()


# Test
model_1 = CreateNeuralNetworkModel(data=get_data(file_name="final_data.csv"))
# model_3 = CreateNeuralNetworkModel(data=get_data(file_name="final_data_3points.csv"))
