import geopy.distance
import numpy as np


def distance(a, b):

    # [] = [latitude, longitude,...]

    pointA = [a[0], a[1]]
    pointB = [b[0], b[1]]

    return geopy.distance.geodesic(pointA, pointB).km
