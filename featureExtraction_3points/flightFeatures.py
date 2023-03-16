import os
import sys
import pytz
import numpy as np
import pandas as pd
import geopy.distance
from datetime import datetime

np.set_printoptions(suppress=True, formatter={'float_kind': '{:f}'.format})

if os.path.abspath(__file__ + "/../../") not in sys.path:
    sys.path.insert(1, os.path.abspath(__file__ + "/../../"))


def distance(a, b):
    # [] = [latitude, longitude,...]
    first_point_coord = [a[0], a[1]]
    second_point_coord = [b[0], b[1]]

    return geopy.distance.geodesic(first_point_coord, second_point_coord).km


def split(start, end, segments):
    delta = []
    points = []

    # Features' delta
    for i in range(len(start)):
        delta.append((end[i] - start[i]) / float(segments))

    # Generate data of each point in segments
    for i in range(1, segments):
        temp_data = []

        # Generate data of each feature of each point
        for j in range(len(delta)):
            temp_data.append(start[j] + i * delta[j])
        points.append(temp_data)

    start = start.tolist()
    end = end.tolist()

    return [start] + points + [end]


def intersectTMA(traj, airport):
    # Get index to retrieve 3 data points before entering TMA
    intersect_with_TMA_index = 0

    if distance(traj[0], airport) < 75:
        return traj, intersect_with_TMA_index

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

                linear_interpolate_value_row = np.array(segments[data_point])

                # Add new point to trajectory
                new_traj = np.insert(traj, i + 1, linear_interpolate_value_row, axis=0)

                intersect_with_TMA_index = i
                return new_traj, intersect_with_TMA_index


def intersectLandingPoint(traj):
    if traj[-1][4] >= 500:
        return traj

    else:
        for i in range(len(traj) - 1):
            if traj[i][4] == 500:
                return traj

            elif traj[i][4] > 500 and traj[i + 1][4] > 500:
                continue

            elif traj[i][4] > 500 and traj[i + 1][4] < 500:
                # Split the line between 2 points in-out of TMA
                segments = split(traj[i], traj[i + 1], 200)
                data_point = 0

                # Find the inside-and-nearest-to-TMA point (the outside points will be trimmed later)
                for j in range(len(segments)):
                    if segments[j][4] > 500:
                        continue

                    else:
                        data_point = j
                        break

                linear_interpolate_value_row = np.array(segments[data_point])

                # Add new point to trajectory
                new_traj = np.insert(traj, i + 1, linear_interpolate_value_row, axis=0)

                return new_traj


def trim_outside_TMA(traj, config):
    countOutsideTMA = 0

    for i in range(len(traj)):
        coord = traj[i]
        distanceFromAirport = distance(coord, config['airport'])

        if distanceFromAirport > 75:
            countOutsideTMA += 1

        else:
            break

    trimmed = np.delete(traj, range(countOutsideTMA), axis=0)

    return trimmed


def trim_near_airport(simplified, waypoint_dict):
    # Trim near the airport
    distance_dict = {}
    min_distance = 999

    for wp in ["BUNVI", "XIMLA", "SAMDU"]:
        waypoint = waypoint_dict[wp]
        min_d = 999

        # Check every point
        for point in simplified:
            d_point_point = distance(waypoint, point)
            if d_point_point < min_d:
                min_d = d_point_point
                pos_M = point
                pos_H = point

        # Check segments on which the projected point locates in between
        for i in range(len(simplified) - 1):
            start = simplified[i]
            end = simplified[i + 1]
            MN = end - start
            if (MN[0:2] == np.zeros(2)).all():
                continue
            MP = waypoint - start[0:2]

            proj = np.dot(MP[0:2], MN[0:2]) / np.dot(MN[0:2], MN[0:2])
            if 0 < proj < 1:
                H = start + proj * MN
                d_MH = distance(waypoint, H)
                if d_MH < min_d:
                    min_d = d_MH
                    pos_H = H
                    pos_M = start

        distance_dict[wp] = [min_d, pos_M, pos_H]

        if min_d < min_distance:
            min_distance = min_d
            min_waypoint = wp

    trim_from_index = simplified.tolist().index(distance_dict[min_waypoint][1].tolist())
    if not (distance_dict[min_waypoint][1] == distance_dict[min_waypoint][2]).all():
        simplified = np.append(simplified[:trim_from_index + 1], np.array([distance_dict[min_waypoint][2]]), axis=0)
    return simplified


def find_entry_waypoint(traj, config):
    entry_waypoint = ""
    min_distance = 999
    entry_coord = [traj[0][0], traj[0][1]]

    for wp in config['entry_waypoint']:
        if min_distance > distance(entry_coord, config['waypoint'][wp]):
            min_distance = distance(entry_coord, config['waypoint'][wp])
            entry_waypoint = wp

    return entry_waypoint


def find_landing_runway(simplified_near_airport, config):
    landing_waypoint = ""
    min_distance = 999

    landing_coord = [simplified_near_airport[-1][0], simplified_near_airport[-1][1]]

    for wp in config['iaf']:
        if min_distance > distance(landing_coord, config['waypoint'][wp]):
            min_distance = distance(landing_coord, config['waypoint'][wp])
            landing_waypoint = wp

    if landing_waypoint == "XIMLA":
        landing_runway = "25RL"
    else:
        landing_runway = "07RL"

    return landing_runway


def find_landing_location(traj):
    landing_data = 0
    # Set landing altitude = 500
    for i in range(len(traj)):

        if traj[i][4] > 500:
            continue
        else:
            landing_data = i
            break

    return landing_data


def get_time_HCM(time):
    timeZoneHCM = pytz.timezone("Asia/Ho_Chi_Minh")
    timeHCM = datetime.fromtimestamp(time, timeZoneHCM).time()
    return timeHCM


def find_aircraft_type(model, config):
    for aircraft_type in config['aircraft']:
        for aircraft_subtype in config['aircraft'][aircraft_type]:
            if model in aircraft_subtype:
                return aircraft_type
            else:
                continue


class Flight:
    def __init__(self, file_path, config):
        self.config = config

        try:
            df = pd.read_csv(file_path)
            df.fillna(method='ffill', inplace=True)

        except Exception:
            raise Exception(f"Can't open file {file_path}.")

        # Convert pandas data series to numpy array for calculation
        if pd.Series(['latitude', 'longitude', 'ground_speed', 'timestamp',
                      'altitude', 'heading_angle']).isin(df.columns).all():
            traj = df[['latitude', 'longitude', 'ground_speed', 'timestamp', 'altitude', 'heading_angle']].to_numpy()

        else:
            raise Exception("Unknown data format.")

        # Find the entry-TMA data point
        interpolated_traj_1, intersect_TMA_index = intersectTMA(traj, config['airport'])
        interpolated_traj_2 = intersectLandingPoint(interpolated_traj_1)

        # Retrieve 2 data points before entering TMA
        if intersect_TMA_index < 1:
            raise Exception("Cannot retrieve 2 data points before entering TMA.")
        else:
            self.first_point_data = interpolated_traj_2[intersect_TMA_index - 1]
            self.second_point_data = interpolated_traj_2[intersect_TMA_index]

        # Trim data points outside TMA
        self.traj = trim_outside_TMA(interpolated_traj_2, config)

        # Average time in TMA is approximate 10 minutes --> exclude flight having less than 10 data points (choose 5)
        if self.traj.shape[0] < 5:
            raise Exception(f"Data error, no data within TSN TMA {file_path}.")

        # Instead of detecting arrival route which is insufficient,
        # locating entry waypoint and landing runway is a better choice
        self.entry_waypoint = find_entry_waypoint(self.traj, self.config)
        self.landing_data = find_landing_location(self.traj)

        # Extracting entry time and arrival time in order to calculate time in TMA for prediction
        self.entry_time_HCM = get_time_HCM(self.traj[0][3])
        self.arrival_time_HCM = get_time_HCM(self.traj[self.landing_data][3])

        # Addition feature other than entry lat, long
        self.distance_to_airport = distance(self.traj[0], config['airport'])

        # Landing runway
        self.simplified_near_airport = trim_near_airport(self.traj, config["waypoint"])
        self.landing_runway = find_landing_runway(self.simplified_near_airport, self.config)

        # Get model type: light/medium/heavy/super or NaN
        model = str(df['aircraft_model'].iloc[0])
        if model == "nan":
            self.type = "Unknown"
        else:
            self.type = find_aircraft_type(model, config)

    def get_info(self, timestamp):
        self.traj['time_diff'] = np.abs(self.traj['timestamp'] - timestamp)
        index = self.traj['time_diff'].idxmin()
        return (
            self.traj['timestamp'].iloc[index], self.traj['latitude'].iloc[index],
            self.traj['longitude'].iloc[index], self.traj['altitude'].iloc[index],
            self.traj['heading_angle'].iloc[index], self.traj['ground_speed'].iloc[index])
