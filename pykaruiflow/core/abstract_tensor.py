from abc import abstractmethod
from typing import List, Dict, Union
import numpy as np
from dataclasses import dataclass

DEVICE_CPU = 'cpu'
DEVICE_GPU = 'gpu'


@dataclass
class TensorSpecs:
    dtype: Union[type, str]
    shape: tuple
    device: str = None


class AbstractTensor:
    def __init__(self, data: np.ndarray, specs: TensorSpecs, requires_grad=False):
        self.requires_grad = requires_grad
        self._data = data
        self._specs = specs
        self._grad = None

    @property
    def specs(self) -> TensorSpecs:
        return self._specs

    @property
    def shape(self) -> tuple:
        return self._specs.shape

    @property
    def dtype(self) -> Union[str, type]:
        return self._specs.dtype

    @property
    def data(self) -> np.ndarray:
        assert self._data is not None, 'You must provide the data before running the computations.'
        return self._data

    @property
    def grad(self) -> Union[np.ndarray, None]:
        if not self.requires_grad:
            return None

        assert self._grad is not None, 'Gradient has not been initialized.'
        return self._grad

    @grad.setter
    def grad(self, value: np.ndarray):
        assert isinstance(value, (np.ndarray, np.float32))\
            , f'Gradient value must be a numpy array, but received {type(value)}.'
        assert tuple(value.shape) == tuple(self.specs.shape), \
            f'Shapes are inconsistent. Expected {self.specs.shape} but received {value.shape}.'
        self._grad = value

    @abstractmethod
    def zero_grad(self):
        pass

    @abstractmethod
    def backward(self, outer_grad=None):
        raise NotImplementedError()

    def __repr__(self):
        return f'' \
               f'Tensor(dtype={self.dtype}, shape={self.shape}, data={np.round(self._data, 6)})'
