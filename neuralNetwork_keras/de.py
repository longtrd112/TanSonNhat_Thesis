import pandas as pd

df = pd.read_csv("final_features_3points.csv")
lat = ['first_latitude', 'second_latitude', 'entry_latitude']
long = ['first_longitude', 'second_longitude', 'entry_longitude']

for i in range(len(df)):
    for ftlat in lat:
        df[ftlat].iloc[i] = df[ftlat].iloc[i] - 10.8188

    for ftlong in long:
        df[ftlong].iloc[i] = df[ftlong].iloc[i] - 106.652

df.to_csv("FF3.csv", index=False)
