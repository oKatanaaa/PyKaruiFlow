from pykaruiflow.core import Kernel

import numpy as np


class LogKernel(Kernel):
    def forward(self, input_vals):
        return np.log(input_vals[0])

    def backward(self, input_vals, outer_grad):
        grad0 = 1. / input_vals[0] * outer_grad
        return [grad0]
