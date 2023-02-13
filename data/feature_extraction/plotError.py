import pandas as pd
import matplotlib.pyplot as plt
import os
import json

f = open('tsn_arrival.json')
tsn = json.load(f)

errorData = pd.read_csv("flightError.csv")
dataDirectory = "../Extract"

for i in range(0, len(errorData)):
    day = errorData['day'].iloc[i]
    flight = errorData['flight'].iloc[i]
    error = errorData['error'].iloc[i]

    flightDirectory = os.path.join(dataDirectory, day, flight)
    df = pd.read_csv(flightDirectory)

    # TSN Airport
    plt.scatter(x=tsn['airport'][1], y=tsn['airport'][0])

    # Flight
    plt.plot(df['longitude'], df['latitude'])
    plt.title(f'{day} - {flight.split(".")[0]}: {error}')
    plt.show()

f.close()
