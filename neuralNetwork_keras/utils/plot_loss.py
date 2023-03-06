import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error


def plot_loss_history(result, y_test, y_predict, start_epoch):
    plot_range = range(start_epoch, len(result.history['loss']))

    plt.plot(plot_range, result.history['loss'][start_epoch - 1: -1], label='training loss')
    plt.plot(plot_range, result.history['val_loss'][start_epoch - 1: -1], label='validation loss')

    # Plot test MSE loss value
    # plt.scatter(x=(len(result.history['loss']) - 1), y=mean_squared_error(y_test, y_predict))
    # plt.text((len(result.history['loss']) - 1), mean_squared_error(y_test, y_predict), "Test Loss")

    # Plot test MAE loss value
    plt.scatter(x=(len(result.history['loss']) - 1), y=mean_absolute_error(y_test, y_predict))
    plt.text((len(result.history['loss']) - 1), mean_absolute_error(y_test, y_predict), "Test Loss")

    plt.xticks(range(start_epoch, len(result.history['loss']), 5))

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    return plt.gca()


class PlotLoss:
    def __init__(self, result, y_test, y_predict):
        self.result = result
        self.y_test = y_test
        self.y_predict = y_predict

        fig, (ax1, ax2) = plt.subplots(nrows=2)
        plt.suptitle("Loss vs Validation Loss (MSE)")

        # Plot from beginning
        plt.sca(ax1)
        plot_loss_history(result, y_test, y_predict, start_epoch=1)

        # Plot early stopping
        plt.sca(ax2)
        plot_loss_history(result, y_test, y_predict, start_epoch=2)

        plt.gca()
