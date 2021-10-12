from abc import abstractmethod
from typing import List, Dict, Union
import numpy as np
from dataclasses import dataclass


@dataclass
class TensorSpecs:
    dtype: Union[type, str]
    shape: tuple


class AbstractTensor:
    def __init__(self, specs: TensorSpecs):
        self._data = None
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

    @data.setter
    def data(self, value: np.ndarray):
        assert isinstance(value, np.ndarray), f'Data value must be a numpy array, but received {type(value)}.'
        assert tuple(value.shape) == tuple(self.specs.shape), \
            f'Shapes are inconsistent. Expected {self.specs.shape} but received {value.shape}.'
        self._data = value

    @property
    def grad(self) -> np.ndarray:
        assert self._grad is not None, 'Gradient has not been initialized.'
        return self._grad

    @grad.setter
    def grad(self, value: np.ndarray):
        assert isinstance(value, np.ndarray), f'Gradient value must be a numpy array, but received {type(value)}.'
        assert tuple(value.shape) == tuple(self.specs.shape), \
            f'Shapes are inconsistent. Expected {self.specs.shape} but received {value.shape}.'
        self._grad = value

    @abstractmethod
    def zero_grad(self):
        pass

    @abstractmethod
    def forward(self, feed_dict: dict = {}) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def backward(self, outer_grad=None):
        raise NotImplementedError()

