import numpy as np
import json
from data.utils.distance import distance

f = open('tsn_arrival.json')
data = json.load(f)


def trim_outside_TMA(traj):

    countOutsideTMA = 0

    for i in range(len(traj)):

        coord = traj[i]
        distanceFromAirport = distance(coord, data['airport'])

        if distanceFromAirport > 75:
            countOutsideTMA += 1

        else:
            break

    trimmed = np.delete(traj, range(countOutsideTMA), axis=0)
    return trimmed
