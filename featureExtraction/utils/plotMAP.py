import pandas as pd
import matplotlib.pyplot as plt
import json
from featureExtraction.utils.trim import trim_outside_TMA
from featureExtraction.flightFeatures import trim_near_airport


def plotMAP(flightDirectory):
    with open('tsn_arrival.json') as f:
        tsn = json.load(f)

        df = pd.read_csv(flightDirectory)

        traj = df[['latitude', 'longitude']].to_numpy()
        trimOutsideTraj = trim_outside_TMA(traj)
        # trimNearAirport = trim_near_airport(trimOutsideTraj, tsn['waypoint'])

        # TSN Airport
        fig = plt.scatter(x=tsn['airport'][1], y=tsn['airport'][0])
        fig = plt.text(106.652, 10.8188, "TSN")

        # Flight
        trimmedLatitude = []
        trimmedLongitude = []

        for i in range(len(trimOutsideTraj)):
            trimmedLatitude.append(trimOutsideTraj[i][0])
            trimmedLongitude.append(trimOutsideTraj[i][1])

        fig = plt.plot(trimmedLongitude, trimmedLatitude)
        fig = plt.title(flightDirectory)

    return fig
