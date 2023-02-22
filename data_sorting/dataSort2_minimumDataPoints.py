import pandas as pd
import os

dataDirectory = "../data/Extract"
for day in os.listdir(dataDirectory):

    dayDirectory = os.path.join(dataDirectory, day)

    for flight in os.listdir(dayDirectory):

        flightDirectory = os.path.join(dayDirectory, flight)
        dataFile = pd.read_csv(flightDirectory)

        # 50 data points ~ Data within 50 minutes
        if len(dataFile) < 50:
            os.remove(flightDirectory)
