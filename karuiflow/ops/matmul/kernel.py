from karuiflow.core import OpKernel

import numpy as np


class MatMulKernel(OpKernel):
    def forward(self, input_vals):
        return np.dot(input_vals[0], input_vals[1])

    def backward(self, input_vals):
        return [input_vals[1].T, input_vals[0].T]