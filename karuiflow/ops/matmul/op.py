from typing import List

from karuiflow.core import Op, Tensor, TensorSpecs, Kernel
from .kernel import MatMulKernel


class MatMul(Op):
    def instantiate_kernel(self, inputs: List[Tensor]) -> Kernel:
        return MatMulKernel()

    def infer_output_tensor_specs(self, inputs: List[Tensor]) -> TensorSpecs:
        a = inputs[0]
        b = inputs[1]
        new_shape = (a.shape[0], b.shape[1])
        dtype = a.dtype
        return TensorSpecs(dtype, new_shape)
