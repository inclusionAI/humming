import dataclasses
import os
import threading
from typing import Any, ClassVar

import cuda.bindings.driver as cbd
import torch

import humming.utils.jit as jit_utils
from humming import dtypes
from humming.jit.compiler import NVCCCompiler, NVRTCCompiler


@dataclasses.dataclass(kw_only=True)
class KernelRuntime:
    disable_fast_math: ClassVar[bool] = False
    _instances: ClassVar[dict[tuple[str, tuple[Any, ...]], "KernelRuntime"]] = {}

    def __new__(cls, *args, **kwargs):
        def get_value(value):
            if isinstance(value, dtypes.DataType):
                return str(value)
            if isinstance(value, list):
                value = tuple(value)
            return value

        args_items = tuple(get_value(x) for x in args)
        kwargs_items = tuple((key, get_value(kwargs[key])) for key in sorted(kwargs.keys()))
        signature = (cls.__name__, args_items + kwargs_items)

        if signature not in cls._instances or not cls._instances[signature].inited:
            instance = super().__new__(cls)
            cls._instances[signature] = instance
            instance.inited = False
            instance.cubin_loaded = False
        return cls._instances[signature]

    def __post_init__(self):
        if self.inited:
            return
        self.init_sm_version()
        self.init_kernel()
        self.inited = True
        if not hasattr(self, "_vars"):
            self._vars = vars(self)
        else:
            for key, value in self._vars.items():
                setattr(self, key, value)

    def init_kernel(self):
        raise NotImplementedError

    def init_sm_version(self):
        device_props = torch.cuda.get_device_properties()
        sm_version = device_props.major * 10 + device_props.minor
        self.sm_version = sm_version
        self.sm_version_str = str(sm_version)
        if self.sm_version >= 90:
            self.sm_version_str += "a"

    @staticmethod
    def _get_compiler():
        compiler = os.environ.get("HUMMING_COMPILER", "").lower()
        if compiler == "nvcc":
            return NVCCCompiler
        elif compiler == "nvrtc":
            return NVRTCCompiler
        else:
            try:
                from cuda.bindings import nvrtc  # noqa

                return NVRTCCompiler
            except Exception:
                return NVCCCompiler

    @staticmethod
    def _ensure_cuda_context():
        torch.cuda.set_device(torch.cuda.current_device())

    def prepare(self):
        self._ensure_cuda_context()
        compiler_cls = self._get_compiler()
        kernel_expr = getattr(self, "kernel_expr", None)
        kernel_filename = compiler_cls.compile(
            self.code,
            sm_version=self.sm_version_str,
            kernel_expr=kernel_expr,
            disable_fast_math=self.disable_fast_math,
            postprocess_cubin=self.postprocess_cubin,
        )
        kernel_name = jit_utils.find_kernel_name_in_cubin(kernel_filename, self.name)
        self.kernel_name = kernel_name
        self.kernel_filename = kernel_filename
        if threading.current_thread() is threading.main_thread():
            self.load_cubin()

    def postprocess_cubin(self, cubin_path: str):
        pass

    def load_cubin(self):
        if self.cubin_loaded:
            return None
        kernel_filename = self.kernel_filename
        kernel_name = self.kernel_name
        result, lib = cbd.cuLibraryLoadFromFile(kernel_filename.encode(), [], [], 0, [], [], 0)
        assert result == 0, repr(result)
        result, kernel = cbd.cuLibraryGetKernel(lib, kernel_name.encode())
        assert result == 0, repr(result)
        self.kernel = kernel
        result, func = cbd.cuKernelGetFunction(kernel)
        assert result == 0, repr(result)
        self.func = func
        self.cubin_loaded = True

    def check_context(self):
        assert threading.current_thread() is threading.main_thread()
        if not self.cubin_loaded:
            self.load_cubin()

    def get_cubin_symbol_value(self, name):
        return jit_utils.read_symbol_value(self.kernel_filename, name)

    def list_kernel_attr_name_list(self):
        return list(cbd.CUkernel_attribute)

    def get_kernel_attr_value(self, attr_name, device_index=0):
        device = cbd.CUdevice(device_index)
        attr_enum = getattr(cbd.CUkernel_attribute, attr_name)
        result, value = cbd.cuKernelGetAttribute(attr_enum, self.kernel, device)
        assert result == 0, repr(result)
        return value

    def list_kernel_all_attrs(self, device_index=0):
        attrs = {}
        for name in self.list_kernel_attr_name_list():
            try:
                attrs[name] = self.get_kernel_attr_value(name, device_index)
            except BaseException:
                continue
        return attrs

    def __call__(self, *args, **kwargs):
        raise NotImplementedError
