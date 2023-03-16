import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from neuralNetwork_keras.utils.data_preprocessing_NN import Data1, Data3
from neuralNetwork_keras.utils.plot_loss import PlotLoss
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
    def __init__(self, data, loop):
        self.data = data

        # Data after preprocessing: dropping outliers, splitting training, scaling features
        X_train, y_train, X_val, y_val, X_test, y_test, X, y = \
            data.X_train, data.y_train, data.X_val, data.y_val, data.X_test, data.y_test, data.X, data.y
        number_of_features = data.number_of_features

        # X_train = X_train.drop(['skyc1'], axis=1)
        # X_val = X_val.drop(['skyc1'], axis=1)
        # X_test = X_test.drop(['skyc1'], axis=1)

        # # Features histogram
        # for feature in data.X.columns:
        #     plt.hist(data.X[feature])
        #     plt.title(feature)
        #     plt.show()

        # Model
        max_epochs = 100
        batch_size = 64

        optimizer = tf.keras.optimizers.Adam(epsilon=10e-4, clipnorm=1)

        plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.000001)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=400, verbose=0)

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=100, activation='relu', kernel_regularizer='l1_l2',
                                  input_shape=(number_of_features,)),
            tf.keras.layers.Dense(units=50, activation='relu', kernel_regularizer='l1_l2'),
            tf.keras.layers.Dense(units=1, activation=tf.keras.layers.LeakyReLU(alpha=0.005))
        ])

        model.compile(loss=tf.keras.losses.mean_absolute_error, optimizer=optimizer,
                      metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.RootMeanSquaredError()])

        result = model.fit(X_train, y_train, epochs=max_epochs, validation_data=(X_val, y_val),
                           batch_size=batch_size, callbacks=[plateau, early_stopping], verbose=2)
        model.summary()

        # Prediction
        y_predict = model.predict(X_test, verbose=0)
        model.evaluate(X_test, y_test)

        # Evaluate the metrics
        self.mae = mean_absolute_error(y_test, y_predict)
        self.rmse = np.sqrt(mean_squared_error(y_test, y_predict))
        self.mape = 100 * np.mean(np.abs((y_test.to_numpy() - y_predict) / np.abs(y_test.to_numpy())))

        if not loop:
            # Plot loss history
            PlotLoss(result=result, data=data, model=model)
            plt.show()

            # # Plot feature importance
            # plot_feature_importance(model, X, y)
            # plt.show()


# Test
# model_1 = CreateNeuralNetworkModel(data=get_data(file_name="final_data.csv"), loop=False)
model_3 = CreateNeuralNetworkModel(data=get_data(file_name="final_data_3points.csv"), loop=False)
