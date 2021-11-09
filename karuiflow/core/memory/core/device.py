from abc import abstractmethod


class Device:
    """
    To make the storage independent of the specifics of memory allocation on different devices,
    we use this instance for allocation on the given device.
    """
    @abstractmethod
    def allocate_memory(self, nbytes):
        # In C++ this method will return a pointer for the allocated memory
        pass

    @abstractmethod
    def deallocate_memory(self, data):
        # In C++ this method will receive a pointer for the memory to free
        pass

    @abstractmethod
    def get_device_name(self) -> str:
        pass

    @abstractmethod
    def copy_device_to_cpu(self, mem1, mem2):
        pass

    @abstractmethod
    def copy_cpu_to_device(self, mem1, mem2):
        pass

    @abstractmethod
    def copy_device_to_device(self, mem1, mem2):
        pass
