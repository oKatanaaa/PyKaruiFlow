from typing import List
import numpy as np

from pykaruiflow.core import Op, Tensor, TensorSpecs, Kernel
from pykaruiflow.core.registry import register_add_op
from .kernel import AddKernel


@register_add_op
class Add(Op):
    def instantiate_kernel(self, inputs: List[Tensor]) -> Kernel:
        return AddKernel()

    def infer_output_tensor_specs(self, inputs: List[Tensor]) -> TensorSpecs:
        a = inputs[0]
        return TensorSpecs(a.dtype, a.shape)