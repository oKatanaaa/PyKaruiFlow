from pykaruiflow.core import Kernel

import numpy as np


class MatMulKernel(Kernel):
    def forward(self, input_vals):
        return np.dot(input_vals[0], input_vals[1])

    def backward(self, input_vals, outer_grad):
        grad0 = input_vals[1]
        grad0 = np.dot(outer_grad, grad0.T)

        grad1 = input_vals[0]
        grad1 = np.dot(grad1.T, outer_grad)
        return [grad0, grad1]
