from .op import Sigmoid


def sigmoid(x):
    op = Sigmoid()
    return op([x])
