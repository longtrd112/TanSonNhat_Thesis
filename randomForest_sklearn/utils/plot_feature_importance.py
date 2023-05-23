import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import permutation_importance


def plot_feature_importance(model, data):
    feature_importance = model.best_estimator_.feature_importances_

    merged_importance_model_type = np.sum(feature_importance[18:21])
    merged_importance_landing_runway = np.sum(feature_importance[21:23])

    feature_names = data.X.columns
    feature_names[18:21] = ['model_type']
    feature_names[21:23] = ['landing_runway']
    importances = np.concatenate((feature_importance[:18], [merged_importance_model_type], [merged_importance_landing_runway]))

    # indices = feature_importance.argsort()[::-1]
    #
    # plt.title("Feature importance")
    # plt.bar(range(data.X.shape[1]), feature_importance[indices])
    # plt.xticks(range(data.X.shape[1]), data.X.columns[indices], rotation=90)
    # plt.tight_layout()

    sorted_indices = np.argsort(importances)[::-1]
    sorted_importances = importances[sorted_indices]
    sorted_feature_names = np.array(feature_names)[sorted_indices]

    # Plot the sorted feature importances
    plt.bar(range(len(sorted_importances)), sorted_importances)
    plt.xticks(range(len(sorted_importances)), sorted_feature_names)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importances')
    plt.show()

    return plt.gca()
