import math


# Define the scheduling function
def exponential_lr(epoch):
    start_lr = 0.1
    exp_decay = 0.5

    def lr(epoch, start_lr, exp_decay):
        return start_lr * math.exp(-exp_decay * epoch)

    return lr(epoch, start_lr, exp_decay)


def cyclic_learning_rate(epoch, lr):
    base_lr = 0.01
    max_lr = 0.05

    step_size = 1000

    cycle = math.floor(1 + epoch / (2 * step_size))
    x = abs(epoch / step_size - 2 * cycle + 1)
    lr = base_lr + (max_lr - base_lr) * max(0, (1 - x))

    return lr

