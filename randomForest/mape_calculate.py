import numpy as np
import pandas as pd


def mean_absolute_percentage_error(predict, test):
    percentage_error = 0
    epsilon = 1e-3

    for i in range(len(predict)):

        error = np.abs(predict[i] - test.iloc[i])

        if test.iloc[i] < epsilon:
            percentage_error += error

        else:
            percentage_error += error / test.iloc[i]

    return 100 * percentage_error / len(predict)
