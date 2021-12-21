from .op import Sum


def sum(x, dim):
    op = Sum(dim=dim)
    return op([x])
