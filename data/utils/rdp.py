import numpy as np


def distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def point_line_distance(point, start, end):

    if (start == end).all():
        return distance(point, start)

    else:
        n = abs(
            (end[0] - start[0]) * (start[1] - point[1]) -
            (start[0] - point[0]) * (end[1] - start[1])
        )
        d = np.sqrt(
            (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2
        )
        return n / d


def rdp(points, epsilon):

    """Reduces a series of points to a simplified version that loses detail,
    but maintains the general shape of the series."""

    max_distance = 0.0
    index = 0

    for i in range(1, len(points) - 1):
        d = point_line_distance(points[i], points[0], points[-1])
        if d > max_distance:
            index = i
            max_distance = d

    if max_distance >= epsilon:
        results = rdp(points[:index + 1], epsilon)[:-1] + rdp(points[index:], epsilon)
    else:
        results = [points[0], points[-1]]

    return results
