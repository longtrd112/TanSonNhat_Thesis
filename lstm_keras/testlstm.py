import numpy as np
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error
from lstm_keras.utils.plot_loss import PlotLoss
from lstm_keras.utils.data_preprocessing_LSTM import Data
from lstm_keras.utils.learning_rate_schedule import scheduler
import geopy.distance
import matplotlib.font_manager as font_manager

font = font_manager.FontProperties(family='Times New Roman', size=13)


# Define input and output data
class CreateLSTMModel:
    def __init__(self, data_x, data_y):
        data = Data(data_x=data_x, data_y=data_y)
        seq_len = 3

        # Split data into train and test sets
        X_train, y_train, X_val, y_val, X_test, y_test = \
            data.X_train, data.y_train, data.X_val, data.y_val, data.X_test, data.y_test

        # Model
        max_epochs = 30
        batch_size = 8192

        # Hyperparameter tuning
        optimizer = tf.keras.optimizers.Adam(epsilon=10e-4, clipnorm=1, learning_rate=0.001)
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

        # latitude = np.array([X_test[i][2][0] for i in range(len(X_test))])
        # longitude = np.array([X_test[i][2][1] for i in range(len(X_test))])
        # magnitude = y_predict
        #
        # # plot the data
        # fig, ax = plt.subplots()
        # scatter = ax.scatter(longitude, latitude, c=magnitude)
        #
        # # add a colorbar
        # plt.colorbar(scatter)
        #
        # # set the x and y axis labels
        # ax.set_xlabel('Longitude')
        # ax.set_ylabel('Latitude')
        #
        # # set the title
        # ax.set_title('Magnitude of data points')

        # show the plot
        plt.show()


# Test
with open("sequentialData/data_x_new.pkl", "rb") as f:
    data_x = pickle.load(f)
with open("sequentialData/data_y_new.pkl", "rb") as f:
    data_y = pickle.load(f)
model = CreateLSTMModel(data_x, data_y)
