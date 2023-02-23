import geopy.distance


def distance(a, b):
    # [] = [latitude, longitude,...]
    first_point_coord = [a[0], a[1]]
    second_point_coord = [b[0], b[1]]

    return geopy.distance.geodesic(first_point_coord, second_point_coord).km
