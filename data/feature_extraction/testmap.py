from data.utils.plotSTAR import plotSTAR, plt
from data.utils.plotMAP import plotMAP
import json

f = open('tsn_arrival.json')
tsn = json.load(f)

for entry_waypoint in tsn['entry_waypoint']:

    lat = tsn['waypoint'][entry_waypoint][0]
    long = tsn['waypoint'][entry_waypoint][1]

    plt.scatter(x=long, y=lat)
    plt.text(long, lat, entry_waypoint)


plt.scatter(x=tsn['airport'][1], y=tsn['airport'][0])
plt.text(tsn['airport'][1], tsn['airport'][0], "TSN")
plt.show()
