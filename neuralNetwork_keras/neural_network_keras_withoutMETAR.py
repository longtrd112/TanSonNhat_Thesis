import tensorflow as tf
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from neuralNetwork_keras.utils.data_preprocessing_NN_withoutMETAR import Data1, Data3
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
    def __init__(self, data):
        self.data = data

        # Data after preprocessing: dropping outliers, splitting training, scaling features
        X_train, y_train, X_val, y_val, X_test, y_test, X, y = \
            data.X_train, data.y_train, data.X_val, data.y_val, data.X_test, data.y_test, data.X, data.y
        number_of_features = data.number_of_features

        # Model
        max_epochs = 100
        batch_size = 64

        # Hyperparameter tuning
        optimizer = tf.keras.optimizers.Adam(epsilon=10e-4, clipnorm=1)
        plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.000001)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=20, verbose=0)

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=100, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(),
                                  input_shape=(number_of_features,)),
            tf.keras.layers.Dense(units=50, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2()),
            tf.keras.layers.Dense(units=1, activation=tf.keras.layers.LeakyReLU(alpha=0.005))
        ])

        model.compile(loss=tf.keras.losses.mean_absolute_error, optimizer=optimizer,
                      metrics=[tf.keras.metrics.MeanAbsoluteError(),
                               tf.keras.metrics.RootMeanSquaredError(),
                               tf.keras.metrics.MeanAbsolutePercentageError()])

        # Training
        training = model.fit(X_train, y_train, epochs=max_epochs, validation_data=(X_val, y_val),
                             batch_size=batch_size, callbacks=[plateau, early_stopping], verbose=0)
        model.summary()
        self.model = model
        self.params = model.count_params()

        # Prediction
        y_predict = model.predict(X_test, verbose=0)
        evaluate = model.evaluate(X_test, y_test)

        # Evaluate the metrics
        self.mae = evaluate[1]
        self.rmse = evaluate[2]
        self.mape = evaluate[3]

        # Saving figures
        figure_numbering = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        figure_name = f"figures_withoutMETAR/figure_{figure_numbering}.png"
        PlotLoss(training=training, test=y_test, prediction=y_predict)
        plt.savefig(figure_name, bbox_inches='tight')
        plt.clf()

        # # Plot feature importance
        # plot_feature_importance(model, X_test, y_test)
        # plt.show()


# Test
# model_1 = CreateNeuralNetworkModel(data=get_data(file_name="final_data.csv"))
# model_3 = CreateNeuralNetworkModel(data=get_data(file_name="final_data_3points.csv"))