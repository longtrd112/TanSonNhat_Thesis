import warnings
import csv
import os
import pickle
from neural_network_keras import CreateNeuralNetworkModel

warnings.filterwarnings("ignore")

with open("sequentialData/data_x_new.pkl", "rb") as f:
    data_x = pickle.load(f)
with open("sequentialData/data_y_new.pkl", "rb") as f:
    data_y = pickle.load(f)


MAE_history = []
RMSE_history = []
MAPE_history = []
error_count = 0

for i in range(0, 30):
    print(i)
    try:
        model = CreateNeuralNetworkModel(data_x, data_y)

        if model.mape > 20:
            error_count += 1
        else:
            MAE_history.append(model.mae)
            RMSE_history.append(model.rmse)
            MAPE_history.append(model.mape)

    except Exception as e:
        error_count += 1
        print(e)

history_dict = {
    'MAE': MAE_history,
    'RMSE': RMSE_history,
    'MAPE': MAPE_history
}

with open('results/result_output_neural_network.txt', 'w') as f:
    print('Average mean absolute error: ', sum(MAE_history) / len(MAE_history), file=f)
    print('Average root mean squared error: ', sum(RMSE_history) / len(RMSE_history), file=f)
    print('Average mean absolute percentage error: ', sum(MAPE_history) / len(MAPE_history), file=f)

if not os.path.isfile('results/error_history_neural_network.csv'):
    with open('results/error_history_neural_network.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(history_dict.keys())

with open('results/error_history_neural_network.csv', 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=history_dict.keys())
    for values in zip(*history_dict.values()):
        row = dict(zip(history_dict.keys(), values))
        writer.writerow(row)
