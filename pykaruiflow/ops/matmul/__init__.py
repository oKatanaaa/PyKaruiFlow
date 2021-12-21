from .op import MatMul


def matmul(a, b):
    op = MatMul()
    return op([a, b])

