from .op import Sum


def sum(x, axes):
    op = Sum(axes=axes)
    return op(x)
