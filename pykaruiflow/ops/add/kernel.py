from pykaruiflow.core import Kernel

import logging
import numpy as np


class AddKernel(Kernel):
    def forward(self, inputs: list):
        assert len(inputs) == 2, f'{self.__class__.__name__}.forward / len(inputs) must be 2, but received {len(inputs)}.'
        A, B = inputs
        output = np.add(A, B)
        logging.debug(f'{self.__class__.__name__}.forward successful.')
        return output

    def backward(self, inputs: list,
                      outerGradient: np.ndarray, ):
        assert len(inputs) == 2, f'{self.__class__.__name__}.backward / len(inputs) must be 2, but received {len(inputs)}.'
        A, B = inputs
        # Output buffers

        dims_to_reduce = []
        for i, dim in enumerate(A.shape):
            if dim == 1:
                dims_to_reduce.append(i)
        outerGradient_A = np.sum(outerGradient, axis=tuple(dims_to_reduce), keepdims=True)
        A_grad = np.ones_like(A) * outerGradient_A

        dims_to_reduce = []
        for i, dim in enumerate(B.shape):
            if dim == 1:
                dims_to_reduce.append(i)
        outerGradient_B = np.sum(outerGradient, axis=tuple(dims_to_reduce), keepdims=True)
        B_grad = np.ones_like(B) * outerGradient_B

        logging.debug(f'{self.__class__.__name__}.backward successful.')
        return [A_grad, B_grad]