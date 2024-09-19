from pykaruiflow.core import Kernel

import numpy as np


def flatten(x):
    dim = x.shape[-1]
    return x.reshape(-1, dim)


class SoftmaxKernel(Kernel):
    def forward(self, input_vals):
        tensor = input_vals[0]
        exp = np.exp(tensor - tensor.max(axis=-1, keepdims=True))
        return np.divide(exp, np.sum(exp, axis=-1, keepdims=True), dtype='float32')

    def backward(self, input_vals, outer_grad):
        softmax = flatten(self.forward(input_vals))
        outer_grad = flatten(outer_grad)
        outer_grad = np.expand_dims(outer_grad, axis=1)

        eye = np.identity(softmax.shape[-1])[None, ...]
        jacobian = np.expand_dims(softmax, axis=1) * (eye - softmax[..., None])
        input_grad = np.matmul(outer_grad, jacobian)
        return [input_grad.reshape(input_vals[0].shape)]

