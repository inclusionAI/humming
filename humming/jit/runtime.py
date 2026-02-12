import torch
import functools
from humming.jit.compiler import NVCCCompiler
import humming.jit.utils as jit_utils
import cuda.bindings.driver as cbd


class KernelRuntime(object):
    _instances = {}

    def __new__(cls, *args, **kwargs):
        key = (str(cls), args, tuple(sorted(kwargs.items())))
        if key not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[key] = instance
        return cls._instances[key]

    def prepare(self):
        kernel_filename = NVCCCompiler.compile(self.code, sm_version=self.sm_version_str)
        kernel_name = jit_utils.find_kernel_name_in_cubin(kernel_filename, self.name)

        result, lib = cbd.cuLibraryLoadFromFile(kernel_filename.encode(), [], [], 0, [], [], 0)
        assert result == 0, repr(result)
        result, kernel = cbd.cuLibraryGetKernel(lib, kernel_name.encode())
        assert result == 0, repr(result)

        self.kernel = kernel
        self.kernel_name = kernel_name
        self.kernel_filename = kernel_filename

    def _set_sm_version(self, sm_version=None, device_index=None):
        if isinstance(sm_version, (tuple, list)):
            sm_version = str(sm_version[0] * 10 + sm_version[1])
        elif isinstance(sm_version, (str, int)):
            sm_version = int(sm_version)
        else:
            device_props = torch.cuda.get_device_properties(device_index)
            sm_version = device_props.major * 10 + device_props.minor

        self.sm_version = sm_version
        self.sm_version_str = str(sm_version) + "a" if sm_version >= 90 else str(sm_version)

    @functools.lru_cache
    def get_cubin_symbol_value(self, name):
        return jit_utils.read_symbol_value(self.kernel_filename, name)

    @functools.lru_cache(maxsize=1)
    def list_kernel_attr_name_list(self):
        return list(cbd.CUkernel_attribute)

    @functools.lru_cache(maxsize=1)
    def get_kernel_attr_value(self, attr_name, device_index=0):
        device = cbd.CUdevice(device_index)
        attr_enum = getattr(cbd.CUkernel_attribute, attr_name)
        result, value = cbd.cuKernelGetAttribute(attr_enum, self.kernel, device)
        assert result == 0, repr(result)
        return value

    @functools.lru_cache(maxsize=1)
    def list_kernel_all_attrs(self, device_index=0):
        attrs = {}
        for name in self.list_kernel_attr_name_list():
            try:
                attrs[name] = self.get_kernel_attr_value(name, device_index)
            except BaseException:
                continue
        return attrs

    def __call__(self, arg_values, arg_types, kernel_config, device_index=0):
        device = cbd.CUdevice(device_index)
        return cbd.cuLaunchKernelEx(kernel_config, self.kernel, (arg_values, arg_types), device)
