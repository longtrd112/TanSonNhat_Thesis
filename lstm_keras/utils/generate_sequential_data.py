import pandas as pd
import os
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, RobustScaler


def dataFitTransform(df, function, columns):
    try:
        x = df[columns].values
        x_scaled = function.fit_transform(x)
        df_temp = pd.DataFrame(x_scaled, columns=columns, index=df.index)
        df[columns] = df_temp

        # Return a dataframe instead of numpy array
        return df

    # Deal with OneHotEncoding()
    except (Exception,):
        return pd.get_dummies(data=df, columns=columns)


# Define input and output data
seq_len = 3
data_x = []
data_y = []

for flight in os.listdir('../allFlightsData'):
    try:
        data = pd.read_csv(os.path.join('../allFlightsData', flight))

        # Scaling features
        data['altitude'] = data['altitude'] / 35000
        data['ground_speed'] = data['ground_speed'] / 400

        X = data[['latitude', 'longitude', 'altitude', 'ground_speed', 'heading_angle_sine', 'heading_angle_cosine']]
        y = data.transit_time.values
        y = y.reshape(-1, 1)
        data = np.concatenate((X, y), axis=1)

        for j in range(len(data) - seq_len):
            seq_in = data[j:j + seq_len]
            seq_out = data[j + seq_len - 1, -1]

            data_x.append(seq_in)
            data_y.append(seq_out)

    except (Exception,):
        continue

with open("../sequentialData/data_x.pkl", "wb") as f:
    pickle.dump(np.array(data_x), f)

with open("../sequentialData/data_y.pkl", "wb") as f:
    pickle.dump(np.array(data_y), f)
