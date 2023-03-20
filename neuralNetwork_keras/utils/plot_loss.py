import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error


def plot_loss_history(training, start_epoch):
    plot_range = range(start_epoch, len(training.history['loss']))

    plt.plot(plot_range, training.history['loss'][start_epoch - 1: -1], label='Training loss')
    plt.plot(plot_range, training.history['val_loss'][start_epoch - 1: -1], label='Validation loss')

    plt.xticks(range(start_epoch, len(training.history['loss']), 2))
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()

    return plt.gca()


class PlotLoss:
    def __init__(self, training, test, prediction):
        fig, (ax1, ax2) = plt.subplots(nrows=2)
        plt.suptitle("Training Loss vs validation Loss (Mean Absolute Error)")

        # Plot from beginning
        plt.sca(ax1)
        plot_loss_history(training, start_epoch=1)

        # Plot early stopping
        plt.sca(ax2)
        plot_loss_history(training, start_epoch=2)

        # Plot total error of test set
        plt.scatter(x=(len(training.history['loss']) - 1), y=mean_absolute_error(test, prediction),
                    label='Test error', color='c')

        plt.legend()
        plt.tight_layout()
        plt.gca()
