import pandas as pd

df = pd.read_csv('extracted_features.csv')

time_in_TMA = []

for i in range(len(df)):
    time = df['arrival_time'].iloc[i] - df['entry_time'].iloc[i]
    time_in_TMA.append(time)

df['time_in_TMA'] = time_in_TMA

df.to_csv('final_features.csv', index=False)
