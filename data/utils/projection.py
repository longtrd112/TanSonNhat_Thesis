import numpy as np


def projection_in_segment(point, start, end):
    proj = np.dot(point - start, end - start) / np.dot(end - start, end - start)
    return True if 0 <= proj <= 1 else False


def project_on_line(point, start, end):
    proj = np.dot(point - start, end - start) / np.dot(end - start, end - start)
    return start + proj * (end - start)


def distance_line(point, start, end):
    H = project_on_line(point, start, end)
    return np.sqrt(np.dot(point - H, point - H))


def distance_on_segment2(C, A, B):
    return abs((B[1] - A[1]) * (C[0] - A[0]) - (B[0] - A[0]) * (C[1] - A[1])) / np.sqrt(
        (A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)


def distance_on_segment(C, A, B):
    return ((B[1] - A[1]) * (C[0] - A[0]) - (B[0] - A[0]) * (C[1] - A[1])) / np.sqrt(
        (A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)


def angle(a, b, c, d):
    a = np.array(a[0:2])
    b = np.array(b[0:2])
    c = np.array(c[0:2])
    d = np.array(d[0:2])
    return np.pi - np.dot(a - b, c - d) / np.sqrt(np.dot(a - b, a - b) * np.dot(c - d, c - d))
