from typing import List

from karuiflow.core import Op, Tensor, TensorSpecs
from .kernel import MatMulKernel


class MatMul(Op):
    def __init__(self):
        super().__init__(MatMulKernel)

    def infer_output_tensor_specs(self, input_tensors: List[Tensor]) -> TensorSpecs:
        a = input_tensors[0]
        b = input_tensors[1]
        new_shape = (a.shape[0], b.shape[1])
        dtype = a.dtype
        return TensorSpecs(dtype, new_shape)
