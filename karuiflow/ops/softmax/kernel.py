from karuiflow.core import Kernel

import numpy as np


class SoftmaxKernel(Kernel):
    def forward(self, input_vals):
        tensor = input_vals[0]
        exp = np.exp(tensor)
        return np.divide(exp, np.sum(exp), dtype='float32')

    def backward(self, input_vals, outer_grad):
        softmax = self.forward(input_vals)
        jacobian = softmax[:, None] * (np.identity(softmax.size) - softmax[None, :])
        input_grad = np.dot(outer_grad, jacobian)
        return [input_grad]

