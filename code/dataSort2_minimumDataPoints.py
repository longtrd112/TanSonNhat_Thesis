import pandas as pd
import os

dataDirectory = "../data/Extract"
for day in os.listdir(dataDirectory):

    dayDirectory = os.path.join(dataDirectory, day)

    for flight in os.listdir(dayDirectory):

        flightDirectory = os.path.join(dayDirectory, flight)
        dataFile = pd.read_csv(flightDirectory)

        if len(dataFile) < 10:
            os.remove(flightDirectory)
