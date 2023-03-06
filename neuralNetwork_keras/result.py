import pandas as pd
import warnings
from neural_network_keras import get_data, CreateNeuralNetworkModel
warnings.filterwarnings("ignore")

file_name = 'final_data_3points.csv'

MAE_history = []
RMSE_history = []
MAPE_history = []
error_count = 0

for i in range(0, 100):
    print(i)
    try:

        data = get_data(file_name)
        model = CreateNeuralNetworkModel(data)

        if model.mape > 20:
            error_count += 1
        else:
            MAE_history.append(model.mae)
            RMSE_history.append(model.rmse)
            MAPE_history.append(model.mape)

    except Exception as e:
        error_count += 1
        print(e)

with open('result_output_3.txt', 'w') as f:
    print('Average mean absolute error: ', sum(MAE_history) / len(MAE_history), file=f)
    print('Average root mean squared error: ', sum(RMSE_history) / len(RMSE_history), file=f)
    print('Average mean absolute percentage error: ', sum(MAPE_history) / len(MAPE_history), file=f)

history = {
    'MAE': MAE_history,
    'RMSE': RMSE_history,
    'MAPE': MAPE_history
}
historyDF = pd.DataFrame(data=history)
historyDF.to_csv('loss_history_3.csv', index=False)
