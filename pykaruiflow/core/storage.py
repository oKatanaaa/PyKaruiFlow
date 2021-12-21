import numpy as np


class Storage:
    """
    Responsible for all the memory management.
    Since we don't want to tie all this hustle with computational logic (Tensor, Operations), all the memory
    business is encapsulated within this class.
    """
    def __init__(self, dtype, shape, device, data=None):
        self._dtype = dtype
        self._shape = shape
        self._device = device
        self._data = data

        if data is None:
            self.allocate_memory()

    def allocate_memory(self):
        self._data = np.empty(shape=self._shape, dtype=self._dtype)

    def deallocate_memory(self):
        pass

