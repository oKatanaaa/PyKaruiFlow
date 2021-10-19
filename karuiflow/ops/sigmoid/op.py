from typing import List
from copy import deepcopy

from karuiflow.core import Op, Tensor, TensorSpecs
from .kernel import SigmoidKernel


class Sigmoid(Op):
    def __init__(self):
        super().__init__(SigmoidKernel)

    def infer_output_tensor_specs(self, input_tensors: List[Tensor]) -> TensorSpecs:
        specs = deepcopy(input_tensors[0].specs)
        return specs
