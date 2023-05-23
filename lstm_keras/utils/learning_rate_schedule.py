import tensorflow as tf


# Exponential decay learning rate scheduler
def scheduler(epoch, lr):
    if epoch < 2:
        return lr
    else:
        return lr * tf.math.exp(-0.1)
