from typing import List

from karuiflow.core import Op, Tensor, TensorSpecs, Kernel
from .kernel import LogKernel


class Log(Op):
    def instantiate_kernel(self, inputs: List[Tensor]) -> Kernel:
        return LogKernel()

    def infer_output_tensor_specs(self, inputs: List[Tensor]) -> TensorSpecs:
        a = inputs[0]
        return TensorSpecs(a.dtype, a.shape)
