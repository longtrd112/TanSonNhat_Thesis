import pandas as pd
import warnings
from random_forest_sklearn import get_data, CreateRandomForestModel
warnings.filterwarnings("ignore")

file_name = 'final_data.csv'

MAE_history = []
RMSE_history = []
MAPE_history = []
error_count = 0

for i in range(0, 10):
    try:
        model = CreateRandomForestModel(data=get_data(file_name), loop=True)

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

if file_name == 'final_data.csv':
    with open('results/result_output_random_forest.txt', 'w') as f:
        print('Average mean absolute error: ', sum(MAE_history) / len(MAE_history), file=f)
        print('Average root mean squared error: ', sum(RMSE_history) / len(RMSE_history), file=f)
        print('Average mean absolute percentage error: ', sum(MAPE_history) / len(MAPE_history), file=f)

    history = pd.DataFrame(data=history_dict)
    history.to_csv('results/loss_history_random_forest.csv', index=False)

else:
    with open('results/result_output_random_forest_3points.txt', 'w') as f:
        print('Average mean absolute error: ', sum(MAE_history) / len(MAE_history), file=f)
        print('Average root mean squared error: ', sum(RMSE_history) / len(RMSE_history), file=f)
        print('Average mean absolute percentage error: ', sum(MAPE_history) / len(MAPE_history), file=f)

    history = pd.DataFrame(data=history_dict)
    history.to_csv('results/loss_history_random_forest_3points.csv', index=False)
