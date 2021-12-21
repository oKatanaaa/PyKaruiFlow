from pykaruiflow.core import Kernel

import numpy as np


class SigmoidKernel(Kernel):
    def forward(self, input_vals):
        exp = np.exp(-input_vals[0])
        return np.divide(1.0, (1.0 + exp), dtype='float32')

    def backward(self, input_vals, outer_grad):
        sig = self.forward(input_vals)
        grad0 = sig * (1 - sig)
        grad0 = grad0 * outer_grad
        return [grad0]

