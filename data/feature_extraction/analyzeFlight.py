import pandas as pd
import os

dataDirectory = "../sortedData"
d = []
number = []

for day in os.listdir(dataDirectory):

    dayDirectory = os.path.join(dataDirectory, day)
    for flight in os.listdir(dayDirectory):
        flightDirectory = os.path.join(dayDirectory, flight)
        df = pd.read_csv(flightDirectory)

        model = df['aircraft_model'].iloc[0]

        if model not in d:
            d.append(model)

print(d)
