from karuiflow.core import OpKernel

import numpy as np


class ReLUKernel(OpKernel):
    def forward(self, input_vals):
        return np.where(input_vals[0] > 0.0, input_vals[0], 0.0)

    def backward(self, input_vals, outer_grad):
        grad_mask = np.where(input_vals[0] > 0.0, 1.0, 0.0)
        grad0 = outer_grad * grad_mask
        return [grad0]
