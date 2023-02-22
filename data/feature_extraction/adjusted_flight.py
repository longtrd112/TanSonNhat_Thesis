import os
import sys
import pytz
import numpy as np
import pandas as pd
from data.utils.trim import trim_outside_TMA
from data.utils.distance import distance
from data.utils.intersect import intersect, intersect_pt, intersectTMA
from data.utils.rdp import rdp
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


def check_go_around(holding_path, airport):
    # Check every point
    min_d = min([distance(point, airport) for point in holding_path])

    # Check segments on which the projected point locates in between
    for i in range(len(holding_path) - 1):
        start = holding_path[i, 0:2]
        end = holding_path[i + 1, 0:2]
        MN = end - start
        MP = airport - start
        proj = np.dot(MP, MN) / np.dot(MN, MN)
        if 0 < proj < 1:
            H = start + proj * MN
            d_MH = distance(H, airport)
            if d_MH < min_d:
                min_d = d_MH

    if min_d < 0.05:
        return True
    return False


def find_holding_waypoint(point, config):
    distances = [distance(point, config['waypoint'][waypoint]) for waypoint in config['holding_waypoints']]
    return config['holding_waypoints'][distances.index(min(distances))]


def detect_holding(simplified, config):
    # Detect holding
    result_holding = []
    result_go_around = 0

    i = 2
    cont = i < len(simplified) - 1

    while cont:
        for j in np.arange(i - 2):
            inter = intersect(simplified[j, 0:2], simplified[j + 1, 0:2], simplified[i, 0:2], simplified[i + 1, 0:2])
            if inter == 2:
                alpha_point = (simplified[j] + simplified[j + 1]) / 2
                beta_point = (simplified[i] + simplified[i + 1]) / 2
                new_simplified = np.append(simplified[:j + 1], np.array([alpha_point, beta_point]), axis=0)
                new_simplified = np.append(new_simplified, simplified[i + 1:], axis=0)
                simplified = new_simplified
                result_holding.append([holding_waypoint, beta_point[3] - alpha_point[3]])
                i = j
                break
            if inter == 1:
                alpha_point, beta_point = intersect_pt(simplified[j], simplified[j + 1], simplified[i],
                                                       simplified[i + 1])
                holding_waypoint = find_holding_waypoint(alpha_point, config)
                holding_time = beta_point[3] - alpha_point[3]
                if holding_waypoint in config['iaf']:
                    # check whether a go-around
                    if check_go_around(simplified[j:i + 2], config['airport']):
                        result_go_around += 1
                    else:
                        result_holding.append([holding_waypoint, holding_time])
                else:
                    result_holding.append([holding_waypoint, holding_time])

                new_simplified = np.append(simplified[:j + 1], np.array([alpha_point, beta_point]), axis=0)
                new_simplified = np.append(new_simplified, simplified[i + 1:], axis=0)

                simplified = new_simplified
                i = j
                break
        i += 1
        cont = i < len(simplified) - 1

    return simplified, result_holding, result_go_around


def compute_distance(simplified, waypoint_dict):
    # Compute distance between every waypoint and the path

    distance_dict = {}
    for wp in waypoint_dict:
        waypoint = waypoint_dict[wp]
        min_d = 999
        pos_M = []
        pos_H = []

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
            proj = np.dot(MP, MN[0:2]) / np.dot(MN[0:2], MN[0:2])
            if 0 < proj < 1:
                H = start + proj * MN
                d_MH = distance(waypoint, H)
                if d_MH < min_d:
                    min_d = d_MH
                    pos_H = H
                    pos_M = start

        distance_dict[wp] = [min_d, pos_M, pos_H]
    return distance_dict


def detect_arrival_route(simplified, distance_dict, config, landing_runway):
    arrival_dict = config['arrival_dict']
    waypoint_dict = config['waypoint']
    min_avg_distance = 999
    arrival = ""

    for route in arrival_dict:
        sequence = arrival_dict[route]
        total_diff = np.sum([distance_dict[waypoint][0] for waypoint in sequence])
        avg = total_diff / (len(sequence) + 1)
        if avg < min_avg_distance:
            min_avg_distance = avg
            arrival = route

    if landing_runway == "25RL":
        if arrival == "BAOMY2D":
            arrival = "BAOMY2F"
        elif arrival == "BITIS2D":
            arrival = "BITIS2F"
        elif arrival == "SAPEN2K":
            arrival = "SAPEN2M"

    return arrival


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

        # To numpy
        if pd.Series(['latitude', 'longitude', 'ground_speed', 'timestamp', 'altitude', 'heading_angle']).isin(
                df.columns).all():
            traj = df[['latitude', 'longitude', 'ground_speed', 'timestamp', 'altitude', 'heading_angle']].to_numpy()
        else:
            raise Exception("Unknown data format.")

        # Find the entry-TMA data point
        interpolated_traj = intersectTMA(traj, config['airport'])

        # Trim outside TMA
        self.traj = trim_outside_TMA(interpolated_traj)

        if self.traj.shape[0] < 5:
            raise Exception(f"Data error, no data within TSN TMA {file_path}.")

        self.entry_waypoint = find_entry_waypoint(self.traj, self.config)
        self.landing_data = find_landing_location(self.traj)

        self.entry_time_HCM = get_time_HCM(self.traj[0][3])
        self.arrival_time_HCM = get_time_HCM(self.traj[self.landing_data][3])

        self.distance_to_airport = distance(self.traj[0], config['airport'])

        self.simplified_near_airport = trim_near_airport(self.traj, config["waypoint"])

        self.landing_runway = find_landing_runway(self.simplified_near_airport, self.config)

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
