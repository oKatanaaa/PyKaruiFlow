from .tensor import Tensor, TensorSpecs


class TensorHolder(Tensor):
    def __init__(self, dtype, shape):
        super().__init__(TensorSpecs(dtype, shape))
