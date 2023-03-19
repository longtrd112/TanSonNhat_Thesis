import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, RobustScaler, OrdinalEncoder


def drop_outliers_IQR(df, feature):
    q1 = df[feature].quantile(0.01)
    q3 = df[feature].quantile(0.99)
    IQR = q3 - q1

    # outliers_index = df[((df[feature] < (q1 - 1.5 * IQR)) | (df[feature] > (q3 + 1.5 * IQR)))].index
    drop_outliers = df.drop(df[(df[feature] < (q1 - 1.5 * IQR)) | (df[feature] > (q3 + 1.5 * IQR))].index)

    return drop_outliers


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)

    return X_train, y_train, X_val, y_val, X_test, y_test


def dataFitTransform(df, function, columns):
    try:
        x = df[columns].values
        x_scaled = function.fit_transform(x)
        df_temp = pd.DataFrame(x_scaled, columns=columns, index=df.index)
        df[columns] = df_temp

        # Return a dataframe instead of numpy array
        return df

    # Deal with OneHotEncoding()
    except (Exception,):
        return pd.get_dummies(data=df, columns=columns)


def featuresProcessing(Feature_Processing_Dict, train, val, test):
    for step in Feature_Processing_Dict:
        function = Feature_Processing_Dict[step][0]
        affected_columns = Feature_Processing_Dict[step][1]

        train = dataFitTransform(train, function, affected_columns)
        val = dataFitTransform(val, function, affected_columns)
        test = dataFitTransform(test, function, affected_columns)

    return train, val, test


class Data:
    def __init__(self, data_x, data_y):
        try:
            # Splitting data set
            self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = \
                split_data(data_x, data_y)

            print("Size of training - validation - test set: ", len(self.X_train), len(self.X_val), len(self.X_test))

        except (Exception,):
            print(Exception)
