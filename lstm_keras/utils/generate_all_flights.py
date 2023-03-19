import pandas as pd
import numpy as np
import os
import math
import warnings

warnings.filterwarnings("ignore")

sorted_data_directory = "../../data/sortedData"
count = 1
for date in os.listdir(sorted_data_directory):
    print(count)
    try:
        dateDirectory = os.path.join(sorted_data_directory, date)

        for flight in os.listdir(dateDirectory):
            flightDirectory = os.path.join(dateDirectory, flight)
            df = pd.read_csv(flightDirectory,
                             usecols=['timestamp', 'latitude', 'longitude', 'altitude', 'heading_angle', 'ground_speed'])

            heading_angle_sine = []
            heading_angle_cosine = []

            df = df[np.abs(df['latitude'] - 10.8188) <= 2]
            df = df[np.abs(df['longitude'] - 106.652) <= 2]

            for i in range(len(df)):
                heading_angle_sine.append(math.sin(df.heading_angle.iloc[i]))
                heading_angle_cosine.append(math.cos(df.heading_angle.iloc[i]))

                df.latitude.iloc[i] = df.latitude.iloc[i] - 10.8188
                df.longitude.iloc[i] = df.longitude.iloc[i] - 106.652

            df['heading_angle_sine'] = heading_angle_sine
            df['heading_angle_cosine'] = heading_angle_cosine
            df = df.drop(columns=['heading_angle'])

            # Interpolation
            for i in range(len(df) - 1):
                time_difference = df.timestamp.iloc[i + 1] - df.timestamp.iloc[i]

                if int(time_difference / 60) == 1:
                    continue

                else:
                    row1 = df.iloc[i]
                    row2 = df.iloc[i + 1]

                    # Number of new rows
                    num_rows = int(time_difference / 60) - 1

                    # Loop over the columns and compute the equally spaced values
                    new_rows = {}

                    for feature in df.columns:
                        new_vals = np.linspace(row1[feature], row2[feature], num_rows + 2)[1:-1]
                        new_rows[feature] = new_vals

                    new_df = pd.DataFrame(new_rows)
                    insert_index = i + 1
                    df = pd.concat([df.iloc[:insert_index], new_df, df.iloc[insert_index:]]).reset_index(drop=True)

            # Find landing point and transit time
            landing_timestamp = 0
            transit_time = []

            for i in reversed(range(len(df))):
                if df.altitude.iloc[i] < 500:
                    continue

                else:
                    ratio = (df.altitude.iloc[i] - 500) / (df.altitude.iloc[i] - df.altitude.iloc[i - 1])
                    landing_timestamp = round(
                        df.timestamp.iloc[i] - (df.timestamp.iloc[i] - df.timestamp.iloc[i - 1]) * ratio)
                    break

            for i in range(len(df)):
                transit_time.append(landing_timestamp - df.timestamp.iloc[i])

            df['transit_time'] = transit_time
            df = df[df['transit_time'] >= 0]

            flightName = date + "_" + flight
            df.to_csv(os.path.join('../allFlightsData', flightName), index=False)
            count += 1

    except (Exception,):
        continue
