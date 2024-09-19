import numpy as np

from .tensor import Tensor, TensorSpecs


class Parameter(Tensor):
    """
    Represents a stateful tensor with a value that can be modified.
    """
    @staticmethod
    def from_tensor(t: Tensor):
        return Parameter(t.data)

    def __init__(self, value: np.ndarray):
        specs = TensorSpecs(dtype=value.dtype, shape=value.shape)
        super().__init__(data=value, specs=specs, parent_op=None, input_tensors=[], requires_grad=True)

    def assign(self, t: Tensor):
        self._data = t.data.copy()
        return self
    
    def __iadd__(self, t: Tensor):
        self._data += t.data
        return self
