import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error


def plot_loss_history(result, start_epoch):
    plot_range = range(start_epoch, len(result.history['loss']))

    plt.plot(plot_range, result.history['loss'][start_epoch - 1: -1], label='Training loss')
    plt.plot(plot_range, result.history['val_loss'][start_epoch - 1: -1], label='Validation loss')

    plt.xticks(range(start_epoch, len(result.history['loss']), 2))
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    return plt.gca()


class PlotLoss:
    def __init__(self, result, data, model):
        self.result = result

        fig, (ax1, ax2) = plt.subplots(nrows=2)
        plt.suptitle("Training Loss vs validation Loss (Mean Absolute Error)")

        # Plot from beginning
        plt.sca(ax1)
        plot_loss_history(result, start_epoch=1)

        # Plot early stopping
        plt.sca(ax2)
        plot_loss_history(result, start_epoch=2)

        # Plot total loss of data set
        plt.hlines(xmin=5, xmax=(len(result.history['loss']) - 1),
                   y=mean_absolute_error(data.y_test, model.predict(data.X_test, verbose=0)),
                   label='Test error', color='c')

        plt.legend()
        plt.gca()
