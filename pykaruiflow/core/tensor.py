import numpy as np
from typing import List

from .abstract_tensor import AbstractTensor, TensorSpecs
from .op_kernel import Kernel
from .registry import get_add_op, get_mul_op


class Tensor(AbstractTensor):
    def __init__(
            self,
            data: np.ndarray,
            specs: TensorSpecs,
            parent_op: Kernel = None,
            input_tensors: List[AbstractTensor] = (),
            requires_grad=False
    ):
        self.op = parent_op
        self.input_tensors = input_tensors
        self.grad_initialized = False
        super().__init__(data, specs, requires_grad)

    def init_grad(self):
        if self.grad_initialized:
            return

        self.grad = np.zeros(shape=self.shape, dtype=self.dtype)
        self.grad_initialized = True

    def backward(self, outer_grad=None):
        if not self.requires_grad:
            return None

        self.init_grad()
        if outer_grad is None:
            outer_grad = np.ones_like(self.grad)

        # Accumulate incoming gradient
        self.grad += outer_grad

        if self.op is None:
            return
        else:
            # Compute gradients of this tensor with respect to the input tensors
            input_vals = [x.data for x in self.input_tensors]
            grads = self.op.backward(input_vals, outer_grad)

        for grad, tensor in zip(grads, self.input_tensors):
            # Invoke computation of the inner gradients
            tensor.backward(grad)

    def zero_grad(self):
        if self.requires_grad and self.grad_initialized:
            self.grad = np.zeros_like(self.grad)

        for x in self.input_tensors:
            x.zero_grad()

    @property
    def is_leaf(self):
        return self.op is None
    
    def __add__(self, other):
        assert isinstance(other, (Tensor, float)), f"Can add Tensor only with Tensor or float, but received {type(other)}"
        if isinstance(other, float):
            other = np.array(other, dtype='float32')
            specs = TensorSpecs(other.dtype, shape=other.shape)
            other = Tensor(other, specs)
        add_op = get_add_op()
        return add_op([self, other])
    
    def __mul__(self, other):
        assert isinstance(other, (Tensor, float)), f"Can multiply Tensor with Tensor or float, but received {type(other)}"
        if isinstance(other, float):
            other = np.array(other, dtype='float32')
            specs = TensorSpecs(other.dtype, shape=other.shape)
            other = Tensor(other, specs)
        mul_op = get_mul_op()
        return mul_op([self, other])
    
    def numpy(self):
        return self.data.copy()

    def __del__(self):
        #print(f'Deallocating tensor with specs={self.specs}.')
        pass
