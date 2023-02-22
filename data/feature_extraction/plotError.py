import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from data.utils.plotMAP import plotMAP

f = open('tsn_arrival.json')
tsn = json.load(f)

errorData = pd.read_csv("flightFeatureError.csv")
dataDirectory = "../sortedData"

for i in range(0, len(errorData)):
    day = errorData['day'].iloc[i]
    flight = str(errorData['flight'].iloc[i]) + ".csv"
    error = errorData['error'].iloc[i]

    flightDirectory = os.path.join(dataDirectory, day, flight)
    df = pd.read_csv(flightDirectory)

    # TSN Airport
    plt.scatter(x=tsn['airport'][1], y=tsn['airport'][0])

    # Flight
    plotMAP(flightDirectory)
    plt.title(f'{day} - {flight.split(".")[0]}: {error}')
    plt.show()

f.close()
