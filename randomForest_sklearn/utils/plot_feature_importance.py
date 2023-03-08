import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance


def plot_feature_importance(model, data):
    feature_importance = model.best_estimator_.feature_importances_
    indices = feature_importance.argsort()[::-1]

    plt.title("Feature importance")
    plt.bar(range(data.X.shape[1]), feature_importance[indices])
    plt.xticks(range(data.X.shape[1]), data.X.columns[indices], rotation=90)
    plt.tight_layout()

    return plt.gca()
