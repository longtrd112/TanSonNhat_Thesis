import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance


def plot_feature_importance(model, X, y):
    result = permutation_importance(model, X, y, scoring='neg_mean_absolute_error', n_jobs=-1, n_repeats=5)

    for i in result.importances_mean.argsort()[::-1]:
        print(f"{X.columns[i]:<8}: "
              f"{result.importances_mean[i]:.3f}"
              f" +/- {result.importances_std[i]:.3f}")

    # fig, ax = plt.subplots()
    # ax.bar(X.columns[result.importances_mean.argsort()[::-1]],
    #        result.importances_mean[result.importances_mean.argsort()[::-1]],
    #        yerr=result.importances_std[result.importances_mean.argsort()[::-1]])
    #
    # ax.set_xticklabels(X.columns[result.importances_mean.argsort()[::-1]], rotation=90)
    # ax.set_ylabel('Importance')
    # ax.set_title('Permutation Feature Importance')
    # fig.tight_layout()
    #
    # return plt.gca()
