import numpy as np

from .abstract_tensor import DEVICE_CPU, DEVICE_GPU
from .tensor import Tensor, TensorSpecs
from .op import Op, Kernel
from .holder import TensorHolder


def tensor(data: np.ndarray, dtype=None, device=DEVICE_CPU, requires_grad=False):
    if dtype is not None:
        data = np.asarray(data, dtype=dtype)
    else:
        dtype = data.dtype
    specs = TensorSpecs(dtype=dtype, shape=data.shape, device=device)
    return Tensor(data=data, specs=specs, parent_op=None, input_tensors=[], requires_grad=requires_grad)
