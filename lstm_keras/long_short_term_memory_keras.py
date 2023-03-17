import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os

# Define input and output data
seq_len = 1
data_x = []
data_y = []
for flight in os.listdir('allFlights'):
    data = pd.read_csv(os.path.join('allFlights', flight))
    X = data[['latitude', 'longitude', 'altitude', 'ground_speed', 'heading_angle']]
    y = data.transit_time.values
    y = y.reshape(-1, 1)
    X = X.values # convert to numpy array
    scaler = MinMaxScaler() # normalize data to [0, 1] range
    X = scaler.fit_transform(X)
    data = np.concatenate((X, y), axis=1)

    for j in range(len(data) - seq_len):
        seq_in = data[j:j+seq_len]
        seq_out = data[j+seq_len, -1]

        data_x.append(seq_in)
        data_y.append(seq_out)

data_x = np.array(data_x)
data_y = np.array(data_y)

# Split data into train and test sets
split_idx = int(0.8 * len(data_x))
train_x, train_y = data_x[:split_idx], data_y[:split_idx]
test_x, test_y = data_x[split_idx:], data_y[split_idx:]

# Define LSTM model
model = Sequential()
model.add(LSTM(100, input_shape=(seq_len, data_x.shape[-1])))
model.add(Dense(1, activation='linear'))
model.compile(loss='mae', optimizer='adam')

# Train the model
model.fit(train_x, train_y, epochs=100, batch_size=64, validation_data=(test_x, test_y), verbose=2)

# Evaluate the model on test data
test_loss = model.evaluate(test_x, test_y, batch_size=32)
print(f'Test loss: {test_loss}')
