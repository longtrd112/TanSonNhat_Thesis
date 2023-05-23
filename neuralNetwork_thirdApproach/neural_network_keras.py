import tensorflow as tf
import pandas as pd
import numpy as np
import datetime
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from neuralNetwork_thirdApproach.utils.data_preprocessing_NN3 import Data
from neuralNetwork_keras.utils.plot_feature_importance import plot_feature_importance
import geopy.distance
import matplotlib.font_manager as font_manager

font = font_manager.FontProperties(family='Times New Roman', size=13)


# def get_data(file_name):
#     # ETA with 1 entry waypoint and with/without 2 previous data points
#     if file_name == 'final_data.csv':
#         data = Data1(dataFile=pd.read_csv(file_name).dropna())
#     elif file_name == 'final_data_3points.csv':
#         data = Data3(dataFile=pd.read_csv(file_name).dropna())
#     else:
#         raise Exception("Invalid data file.")
#
#     return data


class CreateNeuralNetworkModel:
    def __init__(self, data_x, data_y):
        data = Data(data_x=data_x, data_y=data_y)

        # Data after preprocessing: dropping outliers, splitting training, scaling features
        X_train, y_train, X_val, y_val, X_test, y_test = \
            data.X_train, data.y_train, data.X_val, data.y_val, data.X_test, data.y_test

        # # Features histogram
        # for feature in data.X.columns:
        #     plt.hist(data.X[feature])
        #     plt.title(feature)
        #     plt.show()

        # Model
        max_epochs = 20
        batch_size = 8192

        # Hyperparameter tuning
        optimizer = tf.keras.optimizers.Adam(epsilon=10e-4, clipnorm=1)
        plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.000001)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=20, verbose=0)

        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=data_x[0].shape),
            tf.keras.layers.Dense(units=100, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2()),
            tf.keras.layers.Dense(units=50, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2()),
            tf.keras.layers.Dense(units=1, activation=tf.keras.layers.LeakyReLU(alpha=0.005))
        ])

        model.compile(loss=tf.keras.losses.mean_absolute_error, optimizer=optimizer,
                      metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.RootMeanSquaredError()])

        # Training
        training = model.fit(X_train, y_train, epochs=max_epochs, validation_data=(X_val, y_val),
                             batch_size=batch_size, callbacks=[plateau, early_stopping], verbose=2)
        model.summary()
        self.model = model

        # Prediction
        y_predict = model.predict(X_test, verbose=0)
        model.evaluate(X_test, y_test)

        # Evaluate the metrics
        self.mae = mean_absolute_error(y_test.reshape(-1, 1), y_predict.reshape(-1, 1))
        self.rmse = np.sqrt(mean_squared_error(y_test.reshape(-1, 1), y_predict.reshape(-1, 1)))
        self.mape = 100 * np.mean(np.abs(y_test.reshape(-1, 1) - y_predict.reshape(-1, 1)) / np.abs(y_test.reshape(-1, 1)))

        # create some sample data
        dis = []
        pred = y_predict.tolist()
        ap = [10.8188, 106.652]
        for i in range(len(X_test)):
            latitude = X_test[i][2][0] + 10.8188
            longitude = X_test[i][2][1] + 106.652
            coord = [latitude, longitude]
            dis.append(geopy.distance.distance(coord, ap).km)
        dis_sorted, pred_sorted = zip(*sorted(zip(dis, pred)))

        segmentsSize = len(dis_sorted) // 20
        it = 0
        dis = []
        avgPred = []
        maxPred = []
        minPred = []
        p25 = []
        p75 = []

        while it < len(dis_sorted):
            dis.append(np.average(dis_sorted[it:it + segmentsSize]))
            avgPred.append(np.average(pred_sorted[it:it + segmentsSize]))
            maxPred.append(np.max(pred_sorted[it:it + segmentsSize]))
            minPred.append(np.min(pred_sorted[it:it + segmentsSize]))

            p25.append(np.percentile(pred_sorted[it:it + segmentsSize], 25))
            p75.append(np.percentile(pred_sorted[it:it + segmentsSize], 75))

            it = it + segmentsSize

        plt.plot(dis, avgPred, label='Average transit time')
        plt.plot(dis, p25, label='1st quartile transit time')
        plt.plot(dis, p75, label='3rd quartile transit time')

        plt.xlabel('Distance to airport (km)', fontproperties=font)
        plt.ylabel('Transit time (seconds)', fontproperties=font)

        plt.xticks(fontproperties=font)
        plt.xticks(fontproperties=font)

        plt.legend()
        plt.gca().invert_xaxis()

        # Saving figures
        # figure_numbering = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        # figure_name = f"figures/figure_{figure_numbering}.png"
        # PlotLoss(training=training, test=y_test, prediction=y_predict)
        # plt.savefig(figure_name, bbox_inches='tight')
        # plt.show()
        # plt.clf()

        # # Plot feature importance
        # plot_feature_importance(model, X, y)
        plt.show()


with open("sequentialData/data_x_new.pkl", "rb") as f:
    data_x = pickle.load(f)
with open("sequentialData/data_y_new.pkl", "rb") as f:
    data_y = pickle.load(f)

# Test
model_1 = CreateNeuralNetworkModel(data_x, data_y)
# model_3 = CreateNeuralNetworkModel(data=get_data(file_name="final_data_3points.csv"))
