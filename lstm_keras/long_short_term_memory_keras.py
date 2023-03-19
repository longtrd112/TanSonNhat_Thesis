import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from lstm_keras.utils.plot_loss import PlotLoss
from lstm_keras.utils.data_preprocessing_LSTM import Data

# Define input and output data
seq_len = 3
with open("data_x_1.pkl", "rb") as f:
    data_x = pickle.load(f)
with open("data_y_1.pkl", "rb") as f:
    data_y = pickle.load(f)

for i in range(30, 40):
    data = Data(data_x=data_x, data_y=data_y)

    # Split data into train and test sets
    X_train, y_train, X_val, y_val, X_test, y_test = \
        data.X_train, data.y_train, data.X_val, data.y_val, data.X_test, data.y_test

    # Model
    max_epochs = 30
    batch_size = 4096

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
    result = model.fit(X_train, y_train, epochs=max_epochs, validation_data=(X_val, y_val),
                       batch_size=batch_size, callbacks=[early_stopping, plateau], verbose=2)
    model.summary()

    # Evaluate the metrics
    evaluate = model.evaluate(X_test, y_test)

    # Prediction
    y_predict = model.predict(X_test, verbose=0)

    # Plot figures
    figure_name = f"figures/figure_{i}.png"
    PlotLoss(result=result, data=data, model=model)
    plt.savefig(figure_name, bbox_inches='tight')
    plt.clf()

    with open('results/evaluate.txt', 'a') as f:
        f.write("Model: " + str(i) + "\n")
        f.write("Loss: " + str(evaluate[0]) + "\n")
        f.write("MAE: " + str(evaluate[1]) + "\n")
        mape = 100 * np.mean(np.abs((y_test.reshape(-1, 1) - y_predict.reshape(-1, 1)) / np.abs(y_test.reshape(-1, 1))))
        f.write("MAPE: " + str(mape) + "\n")
        f.write("---\n")
