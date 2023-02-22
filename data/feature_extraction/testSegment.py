from data.utils.intersect import split, intersectTMA
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import json
from data.utils.distance import distance


f = open('tsn_arrival.json')
tsn = json.load(f)
airport = tsn['airport']

df = pd.read_csv("../sortedData/2020-05-09/VJ173.csv")
traj = df[['latitude', 'longitude', 'ground_speed', 'timestamp', 'altitude', 'heading_angle']].to_numpy()

if distance(traj[0], airport) < 75:
    print("A")
else:
    for i in range(len(traj) - 1):
        if distance(traj[i], airport) > 75 and distance(traj[i + 1], airport) > 75:
            continue

        elif distance(traj[i], airport) > 75 and distance(traj[i + 1], airport) < 75:

            # Split the line between 2 points in-out of TMA
            segments = split(traj[i], traj[i + 1], 200)
            min_distance = 9999
            data_point = 0

            # Find the inside-and-nearest-to-TMA point (the outside points will be trimmed later)
            for j in range(len(segments)):
                if distance(segments[j], airport) > 75:
                    continue

                elif min_distance > np.abs(distance(segments[j], airport) - 75):
                    min_distance = np.abs(distance(segments[j], airport) - 75)
                    data_point = j

                else:
                    break

            # Interpolating 'ground_speed', 'timestamp', 'altitude', 'heading_angle'

            print(segments[data_point])
            linear_interpolate_value_row = np.array(segments[data_point])

            # Add new point to trajectory
            new_traj = np.insert(traj, i + 1, linear_interpolate_value_row, axis=0)
            print(new_traj)
