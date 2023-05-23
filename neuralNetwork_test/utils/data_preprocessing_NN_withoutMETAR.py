import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, RobustScaler, OrdinalEncoder


def drop_outliers_IQR(df, feature):
    q1 = df[feature].quantile(0.01)
    q3 = df[feature].quantile(0.99)
    IQR = q3 - q1

    # outliers_index = df[((df[feature] < (q1 - 1.5 * IQR)) | (df[feature] > (q3 + 1.5 * IQR)))].index
    drop_outliers = df.drop(df[(df[feature] < (q1 - 1.5 * IQR)) | (df[feature] > (q3 + 1.5 * IQR))].index)

    return drop_outliers


def convert_heading_angle(data):
    for feature in data.columns:
        if 'heading_angle' in feature:
            s = []
            c = []

            for i in range(len(data)):
                s.append(math.sin(math.radians(data[feature].iloc[i])))
                c.append(math.cos(math.radians(data[feature].iloc[i])))

            sine_column_name = feature + '_sine'
            cosine_column_name = feature + '_cosine'

            data[sine_column_name] = s
            data[cosine_column_name] = c

    for feature in data.columns:
        if 'heading_angle' in feature and 'sine' not in feature:
            data = data.drop([feature], axis=1)

    return data


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


class Data1:
    def __init__(self, dataFile):
        try:
            self.features = ['entry_latitude', 'entry_longitude', 'entry_altitude',
                             'entry_ground_speed', 'entry_heading_angle',
                             'model_type', 'landing_runway']
            self.features_with_outliers = []

            dataFile = dataFile.drop(dataFile[dataFile.distance_to_airport < 50].index, inplace=False)
            dataFile = dataFile.drop(['distance_to_airport'], axis=1)

            X = dataFile[self.features]
            X = convert_heading_angle(X)
            y = pd.DataFrame(dataFile, columns=['transit_time'], index=dataFile.index)

            columns_to_standard = ['entry_altitude', 'entry_ground_speed']
            columns_to_onehot = ['model_type', 'landing_runway']
            columns_to_ordinal = []
            column_not_to_minmax = columns_to_standard + columns_to_onehot + columns_to_ordinal
            columns_to_minmax = [f for f in X.columns.to_list()
                                 if f not in column_not_to_minmax and 'heading_angle' not in f]

            # Splitting data set
            X_train, self.y_train, X_val, self.y_val, X_test, self.y_test = split_data(X, y)
            print("Size of training - validation - test set: ", len(X_train), len(X_val), len(X_test))

            # Scaling features
            Feature_Processing_Dict = {
                'min_max_scaling': [MinMaxScaler(), columns_to_minmax],
                'standard_scaling': [StandardScaler(), columns_to_standard],
                'ordinal_encoding': [OrdinalEncoder(), columns_to_ordinal],
                'one_hot_encoding': [OneHotEncoder(sparse_output=False), columns_to_onehot]
            }

            self.X_train, self.X_val, self.X_test = featuresProcessing(Feature_Processing_Dict, X_train, X_val, X_test)

            self.number_of_features = len(self.X_train.columns)

            self.X = pd.concat([self.X_train, self.X_val, self.X_test], axis=0)
            self.y = pd.concat([self.y_train, self.y_val, self.y_test], axis=0)

        except (Exception,):
            raise Exception("Add METAR data to features by using code in utils.")


class Data3:
    def __init__(self, dataFile):
        try:
            self.features = ['first_latitude', 'first_longitude', 'first_altitude',
                             'first_ground_speed', 'first_heading_angle',
                             'second_latitude', 'second_longitude', 'second_altitude',
                             'second_ground_speed', 'second_heading_angle',
                             'entry_latitude', 'entry_longitude', 'entry_altitude',
                             'entry_ground_speed', 'entry_heading_angle',
                             'model_type', 'landing_runway']
            self.features_with_outliers = []

            dataFile = dataFile[dataFile.distance_to_airport >= 50]
            dataFile = dataFile.drop(['distance_to_airport'], axis=1)

            X = dataFile[self.features]
            X = convert_heading_angle(X)
            y = pd.DataFrame(dataFile, columns=['transit_time'], index=dataFile.index)

            columns_to_robust = ['first_latitude', 'first_longitude', 'second_latitude', 'second_longitude']
            columns_to_standard = ['entry_altitude', 'entry_ground_speed',
                                   'first_ground_speed', 'second_ground_speed',
                                   'first_altitude', 'second_altitude']
            columns_to_onehot = ['model_type', 'landing_runway']
            columns_to_ordinal = []
            column_not_to_minmax = columns_to_robust + columns_to_standard + columns_to_onehot + columns_to_ordinal
            columns_to_minmax = [f for f in X.columns.to_list()
                                 if f not in column_not_to_minmax and 'heading_angle' not in f]

            # Dropping outliers
            # for feature in self.features_with_outliers:
            #     X, drop_feature_index = drop_outliers_IQR(X, feature)
            #     y = y.drop(drop_feature_index)

            # Splitting data set
            X_train, self.y_train, X_val, self.y_val, X_test, self.y_test = split_data(X, y)
            print("Size of training - validation - test set: ", len(X_train), len(X_val), len(X_test))

            # Scaling features
            Feature_Processing_Dict = {
                'robust_scale': [RobustScaler(), columns_to_robust],
                'min_max_scale': [MinMaxScaler(), columns_to_minmax],
                'standard_scale': [StandardScaler(), columns_to_standard],
                'ordinal_encoding': [OrdinalEncoder(), columns_to_ordinal],
                'one_hot_encoding': [OneHotEncoder(sparse_output=False), columns_to_onehot]
            }

            X_train, X_val, X_test = featuresProcessing(Feature_Processing_Dict, X_train, X_val, X_test)
            self.X_train = convert_heading_angle(X_train)
            self.X_val = convert_heading_angle(X_val)
            self.X_test = convert_heading_angle(X_test)

            self.number_of_features = len(self.X_train.columns)

            self.X = pd.concat([self.X_train, self.X_val, self.X_test], axis=0)
            self.y = pd.concat([self.y_train, self.y_val, self.y_test], axis=0)

        except (Exception,):
            raise Exception("Add METAR data to features by using code in utils.")
