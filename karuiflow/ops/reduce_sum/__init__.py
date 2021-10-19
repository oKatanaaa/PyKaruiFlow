from .op import ReduceSum


def reduce_sum(x, axes):
    op = ReduceSum(axes=axes)
    return op(x)
