from typing import List

from karuiflow.core import Op, Tensor, TensorSpecs
from .kernel import LogKernel


class MatMul(Op):
    def __init__(self):
        super().__init__(LogKernel)

    def infer_output_tensor_specs(self, input_tensors: List[Tensor]) -> TensorSpecs:
        a = input_tensors[0]
        return TensorSpecs(a.dtype, a.shape)
