import os
import sys
import pytz
import numpy as np
import pandas as pd
from data.utils.trim import trim_outside_TMA
from data.utils.distance import distance
from datetime import datetime


np.set_printoptions(suppress=True,
                    formatter={'float_kind': '{:f}'.format})

if os.path.abspath(__file__ + "/../../") not in sys.path:
    sys.path.insert(1, os.path.abspath(__file__ + "/../../"))


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


class Flight:
    def __init__(self, file_path, config):

        self.config = config

        try:
            df = pd.read_csv(file_path)
            df.fillna(method='ffill', inplace=True)
        except Exception:
            raise Exception(f"Can't open file {file_path}.")

        # To numpy
        if pd.Series(['latitude', 'longitude', 'ground_speed', 'timestamp', 'altitude', 'heading_angle']).isin(
                df.columns).all():
            traj = df[['latitude', 'longitude', 'ground_speed', 'timestamp', 'altitude', 'heading_angle']].to_numpy()
        else:
            raise Exception("Unknown data format.")

        # Trim outside TMA
        self.traj = trim_outside_TMA(traj)
        if self.traj.shape[0] < 2:
            raise Exception(f"Data error, no data within TSN TMA {file_path}.")

        self.entry_waypoint = find_entry_waypoint(self.traj, self.config)

        self.landing_data = find_landing_location(self.traj)

        self.simplified_near_airport = trim_near_airport(self.traj, config["waypoint"])
        self.landing_runway = find_landing_runway(self.simplified_near_airport, self.config)

        self.entry_time_HCM = get_time_HCM(self.traj[0][3])
        self.arrival_time_HCM = get_time_HCM(self.traj[self.landing_data][3])

    def get_info(self, timestamp):
        self.traj['time_diff'] = np.abs(self.traj['timestamp'] - timestamp)
        index = self.traj['time_diff'].idxmin()
        return (
            self.traj['timestamp'].iloc[index], self.traj['latitude'].iloc[index],
            self.traj['longitude'].iloc[index], self.traj['altitude'].iloc[index],
            self.traj['heading_angle'].iloc[index], self.traj['ground_speed'].iloc[index])

    def get_type(self):
        try:
            return self.traj['aircraft_model'][0]
        except (Exception,):
            return "unknown"
