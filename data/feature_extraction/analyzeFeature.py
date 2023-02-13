import pandas as pd
import json

f = open('tsn_arrival.json')
tsn = json.load(f)

df = pd.read_csv("extracted_features.csv")

for wp in tsn['entry_waypoint']:
    value = df['entry_waypoint'].value_counts()[wp]
    print(f'{wp}: {value}')

print("\n")

count07 = df['landing_runway'].value_counts()["07RL"]
count25 = df['landing_runway'].value_counts()["25RL"]
print(f'07RL: {count07}')
print(f'25RL: {count25}')

f.close()
