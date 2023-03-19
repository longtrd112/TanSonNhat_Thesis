import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split


# Define input and output data
seq_len = 3
with open("data_x_1.pkl", "rb") as f:
    data_x = pickle.load(f)
with open("data_y_1.pkl", "rb") as f:
    data_y = pickle.load(f)

for i in range(15, 20):
    # Split data into train and test sets
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2)
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.25)

    # Define LSTM model
    model = Sequential()
    model.add(LSTM(units=50, activation='relu', input_shape=(seq_len, data_x.shape[-1]), kernel_regularizer='l1_l2'))
    model.add(Dense(units=1, activation=tf.keras.layers.LeakyReLU(alpha=0.005)))

    plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.000001)

    optimizer = tf.keras.optimizers.Adam(epsilon=10e-4, clipnorm=1)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, min_delta=10, verbose=0)
    model.compile(loss=tf.keras.losses.mean_absolute_error, optimizer=optimizer,
                  metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.RootMeanSquaredError()])
    # Train the model
    result = model.fit(train_x, train_y, epochs=30, batch_size=8192, validation_data=(val_x, val_y), verbose=1,
                       callbacks=[early_stopping, plateau])

    # Evaluate the model on test data
    evaluate = model.evaluate(test_x, test_y)

    y_predict = model.predict(test_x, verbose=0)

    figure_name = f"figure_{i}.png"
    plot_range = range(2, len(result.history['loss']))
    plt.plot(result.history['loss'], label='Training Loss')
    plt.plot(result.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(range(1, len(result.history['loss']), 2))
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_name, bbox_inches='tight')
    plt.clf()

    figure_name = f"figure_{i}_zoom.png"
    plt.plot(plot_range, result.history['loss'][1:-1], label='Training Loss')
    plt.plot(plot_range, result.history['val_loss'][1:-1], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(range(1, len(result.history['loss']), 2))
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_name, bbox_inches='tight')
    plt.clf()

    with open('evaluate.txt', 'a') as f:
        f.write("Model: " + str(i) + "\n")
        f.write("Loss: " + str(evaluate[0]) + "\n")
        f.write("MAE: " + str(evaluate[1]) + "\n")
        f.write("---\n")

    with open('mape.txt', 'a') as f:
        f.write("Model: " + str(i) + "\n")
        f.write(str(100*np.mean(np.abs((test_y.reshape(-1, 1)-y_predict.reshape(-1, 1))/np.abs(test_y.reshape(-1, 1)))))
                + "\n")
        f.write("---\n")
