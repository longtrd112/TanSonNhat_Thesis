import pandas as pd


def drop_outliers_IQR(df, feature):
    q1 = df[feature].quantile(0.25)
    q3 = df[feature].quantile(0.75)

    IQR = q3 - q1

    outliers_index = df[((df[feature] < (q1 - 1.5 * IQR)) | (df[feature] > (q3 + 1.5 * IQR)))].index
    drop_outliers = df.drop(df[(df[feature] < (q1 - 1.5 * IQR)) | (df[feature] > (q3 + 1.5 * IQR))].index)

    return drop_outliers, outliers_index


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
