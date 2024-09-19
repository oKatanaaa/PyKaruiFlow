from typing import List
import numpy as np

from pykaruiflow.core import Op, Tensor, TensorSpecs, Kernel
from pykaruiflow.core.registry import register_mul_op
from .kernel import MulKernel


@register_mul_op
class Mul(Op):
    def instantiate_kernel(self, inputs: List[Tensor]) -> Kernel:
        return MulKernel()

    def infer_output_tensor_specs(self, inputs: List[Tensor]) -> TensorSpecs:
        a = inputs[0]
        return TensorSpecs(a.dtype, a.shape)