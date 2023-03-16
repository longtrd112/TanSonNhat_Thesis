import pandas as pd
import numpy as np


def wind_time(data, wind):
    min_time_diff = 2e9
    index = 0

    for j in range(len(wind)):
        time_diff = np.abs(data - wind['timestamp'].iloc[j])

        if min_time_diff > time_diff:
            min_time_diff = time_diff
            index = j

        else:
            break

    return index


feature = pd.read_csv('../../featureExtraction/extracted_features.csv')
metar = pd.read_csv('../../data/VVTS_metar.csv')

transit_time = []
wind_speed = []
visibility = []
skyc1 = []

for i in range(len(feature)):
    transit_time.append(feature['arrival_time'].iloc[i] - feature['entry_time'].iloc[i])

    index_wind = wind_time(feature['entry_time'].iloc[i], metar)
    wind_speed.append(metar['wind_speed'].iloc[index_wind])
    visibility.append(metar['visibility'].iloc[index_wind])
    skyc1.append(metar['skyc1'].iloc[index_wind])
    print(i)

feature['transit_time'] = transit_time
feature['wind_speed'] = wind_speed
feature['visibility'] = visibility
feature['skyc1'] = skyc1

# Export features
feature.to_csv('../final_data.csv', index=False)
