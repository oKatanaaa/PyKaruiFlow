from typing import List

from karuiflow.core import Op, Tensor, TensorSpecs, Kernel
from .kernel import ReLUKernel


class ReLU(Op):
    def instantiate_kernel(self, inputs: List[Tensor]) -> Kernel:
        return ReLUKernel()

    def infer_output_tensor_specs(self, inputs: List[Tensor]) -> TensorSpecs:
        a = inputs[0]
        return TensorSpecs(a.dtype, a.shape)
