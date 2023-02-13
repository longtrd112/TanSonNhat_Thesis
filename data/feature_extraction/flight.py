import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from data.utils.distance import distance
from data.utils.intersect import intersect, intersect_pt
from data.utils.projection import distance_on_segment2, angle, distance_on_segment
from data.utils.rdp import rdp
from data.utils.trim import trim_outside_TMA

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


def find_holding_waypoint(point, config):
    distances = [distance(point, config['waypoint'][waypoint]) for waypoint in config['holding_waypoints']]
    return config['holding_waypoints'][distances.index(min(distances))]


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


def detect_arrival_route(simplified, distance_dict, config):
    arrival_dict = config['arrival_dict']
    confusing = config['confusing']
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

    # Recheck by using min_avg_distance = sum (distance(waypoint, path)) + distance(entry, first waypoint)
    confusion_checked = False
    if arrival in confusing:
        sequence = arrival_dict[arrival]
        total_diff = np.sum([distance_dict[waypoint][0] for waypoint in sequence])
        min_modified_avg_distance = (total_diff + distance_on_segment2(waypoint_dict[arrival_dict[arrival][0]],
                                                                       simplified[0], simplified[1])) / (
                                            len(sequence) + 1) \
                                    + 0.1 * angle(simplified[0], simplified[1], waypoint_dict[arrival_dict[arrival][0]],
                                                  waypoint_dict[arrival_dict[arrival][1]])
        # TODO: change simplified[0], simplified[1] to two more apposite points
        min_avg_distance = total_diff / (len(sequence) + 1)
        for route in confusing[arrival]:
            sequence = arrival_dict[route]
            total_diff = np.sum([distance_dict[waypoint][0] for waypoint in sequence])
            avg = (total_diff + distance_on_segment2(waypoint_dict[arrival_dict[route][0]], simplified[0],
                                                     simplified[1])) / (len(sequence) + 1) \
                  + 0.1 * angle(simplified[0], simplified[1], waypoint_dict[arrival_dict[route][0]],
                                waypoint_dict[arrival_dict[route][1]])
            if avg < min_modified_avg_distance:
                confusion_checked = True
                min_modified_avg_distance = avg
                min_avg_distance = total_diff / (len(sequence) + 1)
                arrival = route

    # One more time
    # if confusion_checked and "SIERA" in arrival:
    #     min_avg_distance = 999
    #     arrival = ""
    #     for route in ["SIERA6B", "SIERA6D", "SIERA7A", "SIERA7C"]:
    #         sequence = arrival_dict[route]
    #         total_diff = np.sum([distance_dict[waypoint][0] for waypoint in sequence])
    #         avg = total_diff / (len(sequence) + 1)
    #         if avg < min_avg_distance:
    #             min_avg_distance = avg
    #             arrival = route

    return arrival


def great_circle_distance(lat1, long1, lat2, long2):
    a = [lat1, long1]
    b = [lat2, long2]
    return distance(a, b)


class Flight:
    def __init__(self, file_path, config):

        self.config = config

        try:
            df = pd.read_csv(file_path)
            df.fillna(method='ffill', inplace=True)
        except:
            raise Exception(f"Can't open file {file_path}.")

        # to numpy
        if pd.Series(['latitude', 'longitude', 'ground_speed', 'timestamp', 'altitude']).isin(df.columns).all():
            # df['timestamp'] = datetime.fromisoformat(df['timestamp']).timestamp()
            traj = df[['latitude', 'longitude', 'ground_speed', 'timestamp', 'altitude']].to_numpy()
        else:
            raise Exception("Unknown data format.")

        # Trim outside TMA
        self.traj = traj
        trimmed = trim_outside_TMA(self.traj)
        if trimmed.shape[0] < 2:
            raise Exception(f"Data error, no data within TSN TMA {file_path}.")

        # simplify trajectory by RPD algo
        simplified = np.array(rdp(trimmed, 0.005))

        # replace string datetime to timestamps for easy computation
        # if ('latitude' and 'longitude' and 'ground_speed' and 'timestamp') in df:
        #     simplified[:,3] = np.apply_along_axis(lambda x : datetime.fromisoformat(x[3]).timestamp(), 1, simplified)

        # detect holding
        simplified, self.holding, self.go_around = detect_holding(simplified, self.config)

        # trim near the airport
        self.simplified = trim_near_airport(simplified, self.config['waypoint'])

        # compute distance dict
        self.distance_dict = compute_distance(self.simplified, self.config['waypoint'])

        # detect arrival route
        self.arrival = detect_arrival_route(simplified, self.distance_dict, self.config)

        self.segmented = False
        self.segments = {}

        self.checked_vectoring = False
        self.vectoring = []
        self.shortrack = []

    def track_change(self):
        # Track changed
        track_change = False
        # for entry in ["CANTO", "ABBEY", "BETTY"]:
        #     if self.distance_dict[entry][0] < 0.1:
        #         if entry not in self.config["arrival_dict"][self.arrival]:
        #             track_change = True
        #             break

        return track_change

    def desegment(self):

        if self.segmented:
            return self.segments

        # Desegment
        sequence = self.config["arrival_dict"][self.arrival]

        i = 0
        j = 0

        flag = False
        subpath = []

        while (i < len(self.simplified)) and (j < len(sequence) - 1):

            is_starting = False
            if (self.simplified[i] == self.distance_dict[sequence[j]][1]).all():
                flag = True
                is_starting = True
                subpath = [self.distance_dict[sequence[j]][2]]

            if flag and not is_starting:
                subpath.append(self.simplified[i])

            if (self.simplified[i] == self.distance_dict[sequence[j + 1]][1]).all():
                flag = False

                if (self.distance_dict[sequence[j + 1]][1] != self.distance_dict[sequence[j + 1]][2]).any():
                    subpath.append(self.distance_dict[sequence[j + 1]][2])

                self.segments[sequence[j]] = subpath
                j += 1
                continue

            i += 1

        self.segmented = True
        return self.segments

    def check_vectoring(self):
        if self.checked_vectoring:
            return self.vectoring, self.shortrack

        # Segmentize the trajectory first
        self.desegment()
        # Check vectoring / short track
        sequence = self.config["arrival_dict"][self.arrival]

        prev = np.array([999, 999, 999, 999, 999])
        flag = False
        flag_normal = False
        count_down = 2

        normal = 0
        actual = 0
        no_warning = 0

        for i in range(len(sequence) - 1):
            wp_start = sequence[i]
            wp_end = sequence[i + 1]
            waypoint_start = np.array(self.config['waypoint'][wp_start])
            waypoint_end = np.array(self.config['waypoint'][wp_end])

            # path is too short maybe
            if len(self.segments[wp_start]) < 2:
                print("Warning at ", wp_start)
                no_warning += 1
                continue

            for point in self.segments[wp_start]:
                distance_to_designated = distance_on_segment(point, waypoint_start, waypoint_end)
                if np.abs(distance_to_designated) > 0.01:
                    count_down = 2
                    if flag:
                        actual += distance(point, prev)
                        dev.append(distance_to_designated)
                    else:
                        flag = True
                        if (prev == np.array([999, 999, 999, 999, 999])).all():
                            start_point = point
                        else:
                            start_point = prev
                            actual += distance(point, prev)
                        # find equiv of start_point
                        flag_normal = True

                        dev = [distance_to_designated]
                else:
                    if flag:
                        count_down -= 1
                        if count_down == 0:
                            flag = False
                            flag_normal = False
                            end_point = prev
                            # find equiv of end_point
                            proj = np.dot(end_point[0:2] - waypoint_start, waypoint_end - waypoint_start) / np.dot(
                                waypoint_end - waypoint_start, waypoint_end - waypoint_start)
                            equiv_end = waypoint_start + proj * (waypoint_end - waypoint_start)
                            normal += distance(waypoint_start, equiv_end)
                            if normal * 0.92 > actual > 0.2:
                                self.shortrack.append(normal - actual)
                            elif actual > normal * 1.08 and actual > 0.2:
                                self.vectoring.append(actual - normal)
                            elif (-1 in np.sign(dev)) and (1 in np.sign(dev)):
                                self.vectoring.append(actual - normal)

                            # reset
                            flag = False
                            count_down = 2
                            normal = 0
                            actual = 0
                            flag_normal = False
                        else:
                            actual += distance(point, prev)

                prev = point

            if flag_normal:
                proj = np.dot(start_point[0:2] - waypoint_start, waypoint_end - waypoint_start) / np.dot(
                    waypoint_end - waypoint_start, waypoint_end - waypoint_start)
                equiv_start = waypoint_start + proj * (waypoint_end - waypoint_start)
                normal += distance(equiv_start, waypoint_end)
                flag_normal = False
            elif flag:
                normal += distance(waypoint_start, waypoint_end)

        if flag:

            if normal * 0.93 > actual > 0.1:
                self.shortrack.append(normal - actual)
            elif actual > normal * 1.07 and actual > 0.1:
                self.vectoring.append(actual - normal)
            elif (-1 in np.sign(dev)) and (1 in np.sign(dev)):
                self.vectoring.append(actual - normal)

        self.checked_vectoring = True
        return self.vectoring, self.shortrack

    # def coming_to_LIMES(self):
    #     return "LIMES" in self.config['arrival_dict'][self.arrival]

    def get_info(self, timestamp):
        self.traj['time_diff'] = np.abs(self.traj['timestamp'] - timestamp)
        index = self.traj['time_diff'].idxmin()
        return self.traj['timestamp'].iloc[index], self.traj['latitude'].iloc[index], self.traj['longitude'].iloc[
            index], \
               self.traj['altitude'].iloc[index], self.traj['heading_angle'].iloc[index], \
               self.traj['ground_speed'].iloc[index]

    # def get_info_in_advance(self, km, entry_time):
    #     entry_waypoint = self.config['arrival_dict'][self.arrival][0]
    #     lat = self.config['waypoint'][entry_waypoint][0]
    #     long = self.config['waypoint'][entry_waypoint][1]
    #     self.traj['distance_diff'] = 999
    #     self.traj.loc[self.traj['timestamp'] < entry_time, 'distance_diff'] = np.abs(
    #         great_circle_distance(self.traj['lat'], self.traj['long'], lat, long) - km)
    #     index = self.traj['distance_diff'].idxmin()
    #     return self.traj['timestamp'].iloc[index], self.traj['lat'].iloc[index], self.traj['long'].iloc[index], \
    #            self.traj['alt'].iloc[index], self.traj['hangle'].iloc[index], self.traj['gspeed'].iloc[index]

    def get_type(self):
        try:
            return self.traj['aircraft_model'][0]
        except (Exception,):
            return "unknown"
