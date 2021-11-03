from typing import List

from karuiflow.core import Op, Tensor, TensorSpecs
from .kernel import SumKernel


class Sum(Op):
    def __init__(self, axes):
        if isinstance(axes, (int, list)):
            axes = tuple(axes)

        self.axes = axes
        super().__init__(SumKernel, axes=axes)

    def infer_output_tensor_specs(self, input_tensors: List[Tensor]) -> TensorSpecs:
        a = input_tensors[0]
        new_shape = []
        for i, dim in enumerate(a.shape):
            # The specified axes are being reduced.
            if i in self.axes:
                continue
            new_shape.append(dim)
        dtype = a.dtype
        return TensorSpecs(dtype, new_shape)
