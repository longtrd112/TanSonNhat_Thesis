import pandas as pd
import numpy as np


def wind_time(data, wind):
    min_time_diff = 2e9
    index = 0

    for i in range(len(wind)):
        time_diff = np.abs(data - wind['timestamp'].iloc[i])

        if min_time_diff > time_diff:
            min_time_diff = time_diff
            index = i

        else:
            break

    return index


feature = pd.read_csv('extracted_features.csv')

metar = pd.read_csv('../metar/VVTS_metar.csv')

time_in_TMA = []
wind_speed = []
wind_direction = []

for i in range(len(feature)):
    time = feature['arrival_time'].iloc[i] - feature['entry_time'].iloc[i]
    time_in_TMA.append(time)

    index_wind = wind_time(feature['entry_time'].iloc[i], metar)

    wind_speed.append(metar['wind_speed'].iloc[index_wind])
    wind_direction.append(metar['wind_direction'].iloc[index_wind])

    print(i)

feature['time_in_TMA'] = time_in_TMA
feature['wind_speed'] = wind_speed
feature['wind_direction'] = wind_direction

feature.to_csv('final_features.csv', index=False)
