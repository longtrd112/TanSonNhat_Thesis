import numpy as np

from data.utils.projection import distance_line, projection_in_segment


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
