from .op import Softmax


def softmax(x, dim):
    op = Softmax()
    return op([x])
