import pandas as pd
import os

sorted_data_directory = "../../data/sortedData"
for date in os.listdir(sorted_data_directory):
    dateDirectory = os.path.join(sorted_data_directory, date)

    for flight in os.listdir(dateDirectory):
        flightDirectory = os.path.join(dateDirectory, flight)
        df = pd.read_csv(flightDirectory)

        transit_time = []
        index = -1
        landing_timestamp = 0

        for i in reversed(range(len(df))):
            if df.altitude.iloc[i] < 500:
                continue

            else:
                landing_timestamp = df.timestamp.iloc[i]
                index = i
                break

        for i in range(len(df)):
            transit_time.append(landing_timestamp - df.timestamp.iloc[i])
            df.latitude.iloc[i] = df.latitude.iloc[i] - 10.8188
            df.longitude.iloc[i] = df.longitude.iloc[i] - 106.652

        df['transit_time'] = transit_time
        df = df[df['transit_time'] >= 0]

        flightName = date + flight
        df.to_csv(os.path.join('../allFlights', flightName), index=False)
