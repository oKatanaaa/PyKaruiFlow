import numpy as np

from .tensor import Tensor, TensorSpecs


class Variable(Tensor):
    """
    Represents a stateful tensor with a value that can be modified.
    """
    def __init__(self, value: np.ndarray):
        specs = TensorSpecs(dtype=value.dtype, shape=value.shape)
        super().__init__(specs=specs, parent_op=None, input_tensors=[])
        self.data = value
