import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler


def drop_outliers_IQR(df, feature):
    q1 = df[feature].quantile(0.01)
    q3 = df[feature].quantile(0.99)
    IQR = q3 - q1

    outliers_index = df[((df[feature] < (q1 - 1.5 * IQR)) | (df[feature] > (q3 + 1.5 * IQR)))].index
    drop_outliers = df.drop(df[(df[feature] < (q1 - 1.5 * IQR)) | (df[feature] > (q3 + 1.5 * IQR))].index)

    return drop_outliers, outliers_index


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)

    return X_train, y_train, X_val, y_val, X_test, y_test


def normalizeData(df, function, columns):
    try:
        x = df[columns].values
        x_scaled = function.fit_transform(x)
        df_temp = pd.DataFrame(x_scaled, columns=columns, index=df.index)
        df[columns] = df_temp
        return df

    # Deal with OneHotEncoding()
    except (Exception,):
        return pd.get_dummies(data=df, columns=columns)


def normalize_features(Scaling_Progress, X_train, X_val, X_test):
    for step in Scaling_Progress:
        function = Scaling_Progress[step][0]
        affected_columns = Scaling_Progress[step][1]

        X_train = normalizeData(X_train, function, affected_columns)
        X_val = normalizeData(X_val, function, affected_columns)
        X_test = normalizeData(X_test, function, affected_columns)

    return X_train, X_val, X_test


class Data1:
    def __init__(self, dataFile):
        self.features = ['entry_latitude', 'entry_longitude', 'entry_altitude',
                         'entry_ground_speed', 'entry_heading_angle',
                         'model_type', 'landing_runway',
                         'wind_speed', 'visibility']
        self.features_with_outliers = []

        # self.columns_to_scale = [f for f in self.features if f not in ('landing_runway', 'model_type')]
        self.columns_to_minmax = [f for f in self.features if f not in ('entry_altitude', 'entry_ground_speed',
                                                                        'landing_runway', 'model_type')]
        self.columns_to_standard = ['entry_altitude', 'entry_ground_speed']
        self.columns_to_onehot = ['model_type', 'landing_runway']

        dataFile = dataFile.drop(dataFile[dataFile.distance_to_airport < 50].index, inplace=False)
        dataFile = dataFile.drop(['distance_to_airport'], axis=1)
        dataFile, drop_time_outliers = drop_outliers_IQR(dataFile, 'time_in_TMA')

        X = dataFile[self.features]
        y = pd.DataFrame(dataFile, columns=['time_in_TMA'], index=dataFile.index)

        self.X = X
        self.y = y

        # Splitting data set
        X_train, self.y_train, X_val, self.y_val, X_test, self.y_test = split_data(X, y)
        print("Size of training - validation - test set: ", len(X_train), len(X_val), len(X_test))

        # Scaling features
        Scaling_Progress = {
            'min_max_scale': [MinMaxScaler(), self.columns_to_minmax],
            'standard_scale': [StandardScaler(), self.columns_to_standard],
            'one_hot_encoding': [OneHotEncoder(sparse_output=False), self.columns_to_onehot]
        }

        self.X_train, self.X_val, self.X_test = normalize_features(Scaling_Progress, X_train, X_val, X_test)
        self.number_of_features = len(self.X_train.columns)


class Data3:
    def __init__(self, dataFile):
        self.features = ['first_latitude', 'first_longitude', 'first_altitude',
                         'first_ground_speed', 'first_heading_angle',
                         'second_latitude', 'second_longitude', 'second_altitude',
                         'second_ground_speed', 'second_heading_angle',
                         'entry_latitude', 'entry_longitude', 'entry_altitude',
                         'entry_ground_speed', 'entry_heading_angle',
                         'model_type', 'landing_runway',
                         'wind_speed', 'visibility']
        self.features_with_outliers = []

        self.columns_to_minmax = [f for f in self.features if f not in ('entry_altitude', 'entry_ground_speed',
                                                                        'landing_runway', 'model_type')]
        self.columns_to_standard = ['entry_altitude', 'entry_ground_speed']
        self.columns_to_onehot = ['model_type', 'landing_runway']

        dataFile = dataFile.drop(dataFile[dataFile.distance_to_airport < 50].index, inplace=False)
        dataFile = dataFile.drop(['distance_to_airport'], axis=1)
        dataFile, drop_time_outliers = drop_outliers_IQR(dataFile, 'time_in_TMA')

        X = dataFile[self.features]
        y = pd.DataFrame(dataFile, columns=['time_in_TMA'], index=dataFile.index)

        # Dropping outliers
        # for feature in self.features_with_outliers:
        #     X, drop_feature_index = drop_outliers_IQR(X, feature)
        #     y = y.drop(drop_feature_index)

        self.X = X
        self.y = y

        # Splitting data set
        X_train, self.y_train, X_val, self.y_val, X_test, self.y_test = split_data(X, y)
        print("Size of training - validation - test set: ", len(X_train), len(X_val), len(X_test))

        # Scaling features
        Scaling_Progress = {
            'min_max_scale': [MinMaxScaler(), self.columns_to_minmax],
            'standard_scale': [StandardScaler(), self.columns_to_standard],
            'one_hot_encoding': [OneHotEncoder(sparse_output=False), self.columns_to_onehot]
        }

        self.X_train, self.X_val, self.X_test = normalize_features(Scaling_Progress, X_train, X_val, X_test)
        self.number_of_features = len(self.X_train.columns)
