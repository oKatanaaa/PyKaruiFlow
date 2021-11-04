from typing import List
from copy import deepcopy

from karuiflow.core import Op, Tensor, TensorSpecs, Kernel
from .kernel import SigmoidKernel


class Sigmoid(Op):
    def instantiate_kernel(self, inputs: List[Tensor]) -> Kernel:
        return SigmoidKernel()

    def infer_output_tensor_specs(self, inputs: List[Tensor]) -> TensorSpecs:
        specs = deepcopy(inputs[0].specs)
        return specs
