import numpy as np
from typing import List

from .abstract_tensor import AbstractTensor, TensorSpecs
from .op_kernel import OpKernel


class Tensor(AbstractTensor):
    def __init__(self, specs: TensorSpecs, parent_op: OpKernel = None, input_tensors: List[AbstractTensor] = ()):
        self.op = parent_op
        self.input_tensors = input_tensors
        self.grad_initialized = False
        super().__init__(specs)

    def init_grad(self):
        if self.grad_initialized:
            return

        self.grad = np.zeros(shape=self.shape, dtype=self.dtype)
        self.grad_initialized = True

    def forward(self, feed_dict: dict = {}) -> np.ndarray:
        if self in feed_dict.keys():
            self.data = feed_dict[self]
            return self.data

        if self.op is None:
            return self.data

        input_data = [x.forward(feed_dict) for x in self.input_tensors]

        self.data = self.op.forward(input_data)
        return self.data

    def backward(self, outer_grad=None):
        self.init_grad()
        if outer_grad is None:
            outer_grad = np.ones_like(self.grad)

        # Accumulate incoming gradient
        self.grad += outer_grad

        # Compute gradients of this tensor with respect to the input tensors
        if self.op is None:
            return
        else:
            input_vals = [x.data for x in self.input_tensors]
            grads = self.op.backward(input_vals, outer_grad)

        for grad, tensor in zip(grads, self.input_tensors):
            # Invoke computation of the inner gradients
            tensor.backward(grad)

    def zero_grad(self):
        self.grad = np.zeros_like(self.grad)

        for x in self.input_tensors:
            x.zero_grad()
