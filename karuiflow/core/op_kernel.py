from typing import List
from abc import abstractmethod
import numpy as np


class Kernel:
    @abstractmethod
    def forward(self, input_values: List[np.ndarray]) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, input_values: List[np.ndarray], outer_grad: np.ndarray) -> List[np.ndarray]:
        pass
