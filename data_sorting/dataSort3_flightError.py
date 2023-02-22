import pandas as pd
import os
from data.utils.distance import distance


def find_final_location(df):
    lastLongitude = df['longitude'].iloc[-1]
    lastLatitude = df['latitude'].iloc[-1]
    return [lastLatitude, lastLongitude]


def append_error(d, f, e, day, flight, error):
    d.append(day)
    f.append(flight)
    e.append(error)


def detect_multiple_flight(df):
    for i in range(len(df) - 1):
        # Timestamp between 2 data points > 3600s --> Landing - take-off
        if df['timestamp'].iloc[i + 1] - df['timestamp'].iloc[i] > 3600:
            return True
    return False


d = []  # day
f = []  # flight
e = []  # error

airport = [10.8188, 106.652]

dataDirectory = "../data/Extract"
for day in os.listdir(dataDirectory):

    dayDirectory = os.path.join(dataDirectory, day)

    for flight in os.listdir(dayDirectory):
        flightDirectory = os.path.join(dayDirectory, flight)
        dataFile = pd.read_csv(flightDirectory)

        finalLocation = find_final_location(dataFile)
        distanceFromAirport = distance(finalLocation, airport)

        if dataFile['altitude'].iloc[-1] > 500:
            append_error(d, f, e, day, flight, "Did not land.")

        elif distanceFromAirport > 20:  # Distance(airport, iaf) ~ 18 km
            append_error(d, f, e, day, flight, "Did not land at TSN.")

        elif detect_multiple_flight(dataFile):
            append_error(d, f, e, day, flight, "Multiple flights detected.")

flightError = {'day': d, 'flight': f, 'error': e}
flightErrorCSV = pd.DataFrame(data=flightError)
flightErrorCSV.to_csv("../data/feature_Extraction/flightError.csv")

print(f'Number of errors: {len(flightErrorCSV)}')
