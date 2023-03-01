from random_forest_sklearn import get_data
import seaborn as sns
import matplotlib.pyplot as plt

data = get_data('final_data_3points.csv')
for ft in data.X_train.columns:
    sns.boxplot(data.X_train[ft])
    plt.title(ft)
    plt.show()