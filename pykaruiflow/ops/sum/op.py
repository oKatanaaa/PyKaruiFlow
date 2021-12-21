from typing import List

from pykaruiflow.core import Op, Tensor, TensorSpecs, Kernel
from .kernel import SumKernel


class Sum(Op):
    def __init__(self, dim):
        if isinstance(dim, (int, list)):
            dim = tuple(dim)

        self.dim = dim

    def instantiate_kernel(self, inputs: List[Tensor]) -> Kernel:
        return SumKernel(dim=self.dim)

    def infer_output_tensor_specs(self, inputs: List[Tensor]) -> TensorSpecs:
        a = inputs[0]
        new_shape = []
        for i, dim in enumerate(a.shape):
            # The specified axes are being reduced.
            if i in self.dim:
                continue
            new_shape.append(dim)
        dtype = a.dtype
        return TensorSpecs(dtype, tuple(new_shape))
