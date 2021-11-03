from karuiflow.core import OpKernel

import numpy as np


class LogKernel(OpKernel):
    def forward(self, input_vals):
        return np.log(input_vals[0])

    def backward(self, input_vals, outer_grad):
        grad0 = 1. / input_vals * outer_grad
        return [grad0]
