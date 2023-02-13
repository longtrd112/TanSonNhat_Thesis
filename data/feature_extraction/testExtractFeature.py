import pandas as pd
import os
import json
from data.utils.plotMAP import plotMAP, plt
from data.utils.plotSTAR import plotSTAR

f = open('tsn_arrival.json')
tsn = json.load(f)

result = pd.read_csv("test_extracted_data.csv")
resultNP = result.to_numpy()

for i in range(len(result)):
    flight = resultNP[i][0]
    landing_runway = resultNP[i][3]
    entry_waypoint = resultNP[i][2]

    fileDirectory = "../sortedData/2020-05-11"
    fileName = flight + ".csv"

    flightDirectory = os.path.join(fileDirectory, fileName)

    for iaf in tsn['iaf']:
        plt.scatter(x=tsn['waypoint'][iaf][1], y=tsn['waypoint'][iaf][0])
        plt.text(tsn['waypoint'][iaf][1], tsn['waypoint'][iaf][0], iaf)

    for wp in tsn['entry_waypoint']:
        plt.scatter(x=tsn['waypoint'][wp][1], y=tsn['waypoint'][wp][0])

    plotMAP(flightDirectory)
    plt.title(f'{flight} - {entry_waypoint} - {landing_runway}')
    plt.legend()

    plt.show()
