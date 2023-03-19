import pandas as pd

with open('evaluate.txt') as f:
    mae = []
    mape = []
    count = 1
    for line in f:
        if "MAE" in line.rstrip():
            mae.append(float(line.rstrip().replace('MAE: ', '')))
        if "MAPE" in line.rstrip():
            mape.append(float(line.rstrip().replace('MAPE: ', '')))
df = pd.DataFrame()
df['mae'] = mae
df['mape'] = mape
df.to_csv('results.csv', index=False)
