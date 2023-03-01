import tensorflow as tf


class StopOnEqualLossAndEpoch(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        train_loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        if train_loss is not None and val_loss is not None:
            if round(train_loss, 5) == round(val_loss, 5) and epoch > 30:
                print("\nTraining stopped on epoch", epoch)
                self.model.stop_training = True
