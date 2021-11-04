from karuiflow.core import Kernel

import numpy as np


class SumKernel(Kernel):
    def __init__(self, dim):
        self.axes = dim

    def forward(self, input_vals):
        input_vals = np.sum(input_vals[0], axis=self.axes)
        return input_vals

    def backward(self, input_vals, outer_grad):
        grad0 = np.ones_like(input_vals[0])
        grad0 = grad0 * np.expand_dims(outer_grad, axis=self.axes)
        return [grad0]

