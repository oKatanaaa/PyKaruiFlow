from abc import abstractmethod
import numpy as np

from .device import Device
from ..devices import CPU


class Storage:
    """
    Responsible for all the memory management.
    Since we don't want to tie all this hustle with computational logic (Tensor, Operations), all the memory
    business is encapsulated within this class.
    """
    def __init__(self, dtype, shape, device: Device):
        self._dtype = dtype
        self._shape = tuple(shape)
        self._device = device
        self._data = None

        self.initialize()

    def initialize(self):
        # nbytes = compute number of needed bytes...
        # self._data = self._device.allocate_memory(nbytes)
        self._data = np.empty(shape=self._shape, dtype=self._dtype)

    def destroy(self):
        self._device.deallocate_memory(self._data)

    @property
    def shape(self):
        return self._shape

    @property
    def size(self):
        # Returns size of the storage in bytes
        raise NotImplementedError()

    @property
    def dtype(self):
        return self._dtype

    @property
    def device_name(self):
        return self._device.get_device_name()

    def copy_data(self, storage: Storage):
        assert self.shape == storage.shape, 'Cannot copy data into the storage' \
                                            'because data shapes are inconsistent: ' \
                                            f'this_storage.shape={self.shape} and' \
                                            f'other_storage.shape={storage.shape}'
        assert self.dtype == storage.dtype, 'Cannot copy data into the storage' \
                                            'because data dtypes are inconsistent: ' \
                                            f'this_storage.dtype={self.dtype} and' \
                                            f'other_storage.dtype={storage.dtype}'
        this_data = self._data
        other_data = storage._data
        if self.device_name == storage.device_name:
            storage.copy_device_to_device(this_data, other_data)
        elif self.device_name == 'cpu' and storage.device_name != 'cpu':
            storage._device.copy_device_to_cpu(storage._data, self._data)
        elif self.device_name != 'cpu' and storage.device_name == 'cpu':
            self._device.copy_cpu_to_device(other_data, this_data)
        elif self.device_name != storage.device_name:
            # In this case both storages store their data on different devices
            # that are not CPU. For example, different GPUs.
            # In this case we must first transfer data to other's device to CPU, then
            # from CPU to this storage's device
            pass

