import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)

    return X_train, y_train, X_val, y_val, X_test, y_test


class Data:
    def __init__(self, data_x, data_y):
        try:
            # Splitting data set
            self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = \
                split_data(data_x, data_y)

            print("Size of training - validation - test set: ", len(self.X_train), len(self.X_val), len(self.X_test))

        except (Exception,):
            print(Exception)
