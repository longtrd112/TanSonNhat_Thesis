import pandas as pd
import os
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, RobustScaler

sorted_data_directory = "../data/sortedData"


# all_flights_directory = "allFlights"
# os.mkdir(all_flights_directory)
# predata = pd.read_csv('final_data.csv')
#
# for i in range(len(predata)):
#     date = predata.date.iloc[i]

# for date in os.listdir(sorted_data_directory):
#     dateDirectory = os.path.join(sorted_data_directory, date)
#     for flight in os.listdir(dateDirectory):
#         flightDirectory = os.path.join(dateDirectory, flight)
#         df = pd.read_csv(flightDirectory)
#         index = -1
#         transit_time = []
#         for i in reversed(range(len(df))):
#             if df.altitude.iloc[i] < 500:
#                 continue
#             else:
#                 landing_timestamp = df.timestamp.iloc[i]
#                 index = i
#                 break
#
#         for i in range(len(df)):
#             transit_time.append(-df.timestamp.iloc[i] + landing_timestamp)
#             df.latitude.iloc[i] = df.latitude.iloc[i] - 10.8188
#             df.longitude.iloc[i] = df.longitude.iloc[i] - 106.652
#
#         df['transit_time'] = transit_time
#         df = df[df['transit_time'] >= 0]
#         flightName = date + flight
#         df.to_csv(os.path.join('allFlights', flightName), index=False)

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
count = 0

for flight in os.listdir('allFlights'):
    try:
        data = pd.read_csv(os.path.join('allFlights', flight))
        data = data[data['transit_time'] > 0]
        data = dataFitTransform(data, RobustScaler(), ['latitude', 'longitude'])
        data = dataFitTransform(data, MinMaxScaler(), ['altitude', 'heading_angle', 'ground_speed'])

        X = data[['latitude', 'longitude', 'altitude', 'ground_speed', 'heading_angle']]
        y = data.transit_time.values
        y = y.reshape(-1, 1)
        data = np.concatenate((X, y), axis=1)
        for j in range(len(data) - seq_len):
            seq_in = data[j:j + seq_len]
            seq_out = data[j + seq_len, -1]

            data_x.append(seq_in)
            data_y.append(seq_out)
    except (Exception,):
        continue

with open("data_x_1.pkl", "wb") as f:
    pickle.dump(np.array(data_x), f)

with open("data_y_1.pkl", "wb") as f:
    pickle.dump(np.array(data_y), f)
