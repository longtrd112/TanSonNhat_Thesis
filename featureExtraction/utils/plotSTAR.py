import matplotlib.pyplot as plt
import pandas as pd
import json


def plotSTAR(route):
    # Airport TSN
    fig = plt.scatter(x=106.652, y=10.8188)
    fig = plt.text(106.652, 10.8188, "TSN")

    with open('tsn_arrival.json') as f:
        data = json.load(f)

        longitude = []
        latitude = []

        # Waypoints' coordinates in route
        for wp in data['arrival_dict'][route]:
            latitude.append(data['waypoint'][wp][0])
            longitude.append(data['waypoint'][wp][1])

        d = {'longitude': longitude, 'latitude': latitude}
        df = pd.DataFrame(data=d)

        fig = plt.plot(df['longitude'], df['latitude'], label=route)

        # Annotate entry waypoint
        entryWaypoint = data['arrival_dict'][route][0]
        plt.scatter(x=data['waypoint'][entryWaypoint][1], y=data['waypoint'][entryWaypoint][0])
        plt.text(data['waypoint'][entryWaypoint][1], data['waypoint'][entryWaypoint][0], entryWaypoint)

        # Annotate ending waypoint/iaf
        endingWaypoint = data['arrival_dict'][route][-1]
        plt.scatter(x=data['waypoint'][endingWaypoint][1], y=data['waypoint'][endingWaypoint][0])
        plt.text(data['waypoint'][endingWaypoint][1], data['waypoint'][endingWaypoint][0], endingWaypoint)

    return fig
