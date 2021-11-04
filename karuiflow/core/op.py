from typing import List, Union
from abc import abstractmethod

from .op_kernel import Kernel
from .tensor import Tensor, TensorSpecs


class Op:
    def assert_device_same(self, inputs: List[Tensor]) -> str:
        """
        Make sure all the tensors are located on the same device.
        """
        device = inputs[0].specs.device
        assert device is not None
        for x in inputs:
            assert device == x.specs.device
        return device

    @abstractmethod
    def infer_output_tensor_specs(self, inputs: List[Tensor]) -> TensorSpecs:
        pass

    def is_requires_grad(self, inputs: List[Tensor]) -> bool:
        requires_grad = False
        for x in inputs:
            requires_grad = requires_grad or x.requires_grad
        return requires_grad

    @abstractmethod
    def instantiate_kernel(self, inputs: List[Tensor]) -> Kernel:
        """
        Instantiates an appropriate kernel for this operation.
        """
        pass

    # noinspection PyCallingNonCallable
    def __call__(self, inputs: List[Tensor]) -> Tensor:
        assert isinstance(inputs, list)
        # Generate meta info about the output tensor
        device = self.assert_device_same(inputs)
        specs = self.infer_output_tensor_specs(inputs)
        specs.device = device
        requires_grad = self.is_requires_grad(inputs)

        kernel = self.instantiate_kernel(inputs)
        in_data = [x.data for x in inputs]
        out_data = kernel.forward(in_data)
        return Tensor(out_data, specs, kernel, inputs, requires_grad=requires_grad)
