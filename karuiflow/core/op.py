from typing import List, Union
from abc import abstractmethod

from .op_kernel import OpKernel
from .tensor import Tensor, TensorSpecs


TensorList = List[Tensor]


class Op:
    def __init__(self, kernel_type: OpKernel, **kernel_kwargs):
        self.kernel_type = kernel_type
        self.kernel_kwargs = kernel_kwargs

    @abstractmethod
    def infer_output_tensor_specs(self, input_tensors: List[Tensor]) -> TensorSpecs:
        pass

    # noinspection PyCallingNonCallable
    def __call__(self, input_tensors: Union[TensorList, Tensor]) -> Tensor:
        if isinstance(input_tensors, Tensor):
            input_tensors = [input_tensors]
        specs = self.infer_output_tensor_specs(input_tensors)
        return Tensor(specs, self.kernel_type(**self.kernel_kwargs), input_tensors)
