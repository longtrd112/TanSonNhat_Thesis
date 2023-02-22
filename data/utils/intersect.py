import numpy as np
from data.utils.projection import distance_line, projection_in_segment
from data.utils.distance import distance
from scipy.interpolate import griddata


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) >= (B[1] - A[1]) * (C[0] - A[0])


def intersect(A, B, C, D):
    # Check if line segments AB and CD intersect.
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    D = np.array(D)
    same_slope = (A[1] - B[1]) * (C[0] - D[0]) == (C[1] - D[1]) * (A[0] - B[0])
    if same_slope:
        if distance_line(A, C, D) == 0:
            if projection_in_segment(A, C, D) or projection_in_segment(B, C, D):
                return 2
            return 0
        return 0

    if ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D):
        return 1
    return 0


def intersect_pt(A, B, C, D):
    # Cond: AB and CD must intersect
    alpha = ((B[0] - A[0]) * (D[1] - A[1]) - (B[1] - A[1]) * (D[0] - A[0])) / (
            (B[1] - A[1]) * (C[0] - D[0]) - (B[0] - A[0]) * (C[1] - D[1]))
    beta = ((D[0] - C[0]) * (B[1] - C[1]) - (D[1] - C[1]) * (B[0] - C[0])) / (
            (D[1] - C[1]) * (A[0] - B[0]) - (D[0] - C[0]) * (A[1] - B[1]))
    return beta * np.array(A) + (1 - beta) * np.array(B), alpha * np.array(C) + (1 - alpha) * np.array(D)


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
    if distance(traj[0], airport) < 75:
        return traj
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

                return new_traj
