import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split


def append_error(d, f, e, day, flight, error):
    d.append(day)
    f.append(flight)
    e.append(error)


df = pd.read_csv('final_features.csv')

df = df.loc[df['time_in_TMA'] > 0]
# df = df.loc[df['model'] != "0"]

# for feature in df.columns.to_list():
#     try:
#         print(df[feature].describe())
#         print(feature)
#         print('\n')
#     except:
#         print("-")
d = []
f = []
e = []

for i in range(len(df)):
    if df['time_in_TMA'].iloc[i] > 3600 or df['time_in_TMA'].iloc[i] < 120:
        append_error(d, f, e, df['date'].iloc[i], df['flight'].iloc[i], "time")

flightError = {'day': d, 'flight': f, 'error': e}
flightErrorCSV = pd.DataFrame(data=flightError)
flightErrorCSV.to_csv("../data/feature_Extraction/flightFeatureError.csv")

