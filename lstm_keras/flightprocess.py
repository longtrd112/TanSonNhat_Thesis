import pandas as pd
import os
from featureExtraction.flightFeatures import intersectLandingPoint, find_landing_runway, \
    find_landing_location, find_aircraft_type

# sorted_data_directory = "../data/sortedData"
# all_flights_directory = "allFlights"
# os.mkdir(all_flights_directory)
# predata = pd.read_csv('final_data.csv')
#
# for i in range(len(predata)):
#     date = predata.date.iloc[i]

for flight in os.listdir('Test'):
    transit_time = []
    df = pd.read_csv(os.path.join('Test', flight))
    index = -1
    for i in reversed(range(len(df))):
        if df.altitude.iloc[i] < 500:
            continue
        else:
            landing_timestamp = df.timestamp.iloc[i]
            index = i
            break
    for i in range(len(df)):
        transit_time.append(-df.timestamp.iloc[i] + landing_timestamp)

    df['transit_time'] = transit_time
    df = df[df['transit_time'] >= 0]
    df.to_csv(os.path.join('allFlights', flight), index=False)
