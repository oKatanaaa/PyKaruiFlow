from abc import abstractmethod
import numpy as np


class Device:
    """
    To make the storage independent of the specifics of memory allocation on different devices,
    we use this instance for allocation on the given device.
    """
    @abstractmethod
    def allocate_memory(self, nbytes):
        pass

    @abstractmethod
    def deallocate_memory(self, data):
        pass


class Storage:
    """
    Responsible for all the memory management.
    Since we don't want to tie all this hustle with computational logic (Tensor, Operations), all the memory
    business is encapsulated within this class.
    """
    def __init__(self, dtype, shape, device: Device):
        self._dtype = dtype
        self._shape = shape
        self._device = device
        self._data = None

        self.initialize()

    def initialize(self):
        # nbytes = compute number of needed bytes...
        # self._data = self._device.allocate_memory(nbytes)
        self._data = np.empty(shape=self._shape, dtype=self._dtype)

    def deallocate_memory(self):
        self._device.deallocate_memory(self._data)

    def copy_memory(self, storage):
        # How to copy memory from device to device?
        pass

