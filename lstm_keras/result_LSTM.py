import csv
import os
import pickle
import warnings
from lstm_keras.long_short_term_memory_keras import CreateLSTMModel
warnings.filterwarnings("ignore")

with open("sequentialData/data_x_new.pkl", "rb") as f:
    data_x = pickle.load(f)
with open("sequentialData/data_y_new.pkl", "rb") as f:
    data_y = pickle.load(f)

MAE_history = []
RMSE_history = []
MAPE_history = []

for i in range(0, 30):
    print(i)
    try:
        model = CreateLSTMModel(data_x=data_x, data_y=data_y)

        MAE_history.append(model.mae)
        RMSE_history.append(model.rmse)
        MAPE_history.append(model.mape)

    except Exception as e:
        print(e)

history_dict = {
    'MAE': MAE_history,
    'RMSE': RMSE_history,
    'MAPE': MAPE_history
}

with open('results/result_output_LSTM.txt', 'w') as f:
    print('Average mean absolute error: ', sum(MAE_history) / len(MAE_history), ' seconds.', file=f)
    print('Average root mean squared error: ', sum(RMSE_history) / len(RMSE_history), ' seconds.', file=f)
    print('Average mean absolute percentage error: ', sum(MAPE_history) / len(MAPE_history), ' %', file=f)

if not os.path.isfile('results/error_history_LSTM.csv'):
    # Create CSV file and header row
    with open('results/error_history_LSTM.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(history_dict.keys())

with open('results/error_history_LSTM.csv', 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=history_dict.keys())
    for values in zip(*history_dict.values()):
        row = dict(zip(history_dict.keys(), values))
        writer.writerow(row)