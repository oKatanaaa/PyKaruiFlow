import numpy as np

from ..core import Device


class CPU(Device):
    NAME = 'cpu'

    def allocate_memory(self, nbytes):
        return np.empty(shape=[nbytes], dtype='uint8')

    def deallocate_memory(self, data):
        pass

    def get_device_name(self) -> str:
        return CPU.NAME

    # --- In all the cases here device means cpu.

    def copy_device_to_cpu(self, mem1, mem2):
        np.copyto(mem1, mem2)

    def copy_cpu_to_device(self, mem1, mem2):
        np.copyto(mem1, mem2)

    def copy_device_to_device(self, mem1, mem2):
        np.copyto(mem1, mem2)
