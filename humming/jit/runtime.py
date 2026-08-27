import dataclasses
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, ClassVar

import cuda.bindings.driver as cbd
import torch

from humming import dtypes
from humming.jit.compiler import NVCCCompiler, NVRTCCompiler
from humming.utils.cubin import get_cubin_kernel_names


@dataclasses.dataclass(kw_only=True)
class KernelRuntime:
    disable_fast_math: ClassVar[bool] = False
    _instances: ClassVar[dict[tuple[Any, ...], "KernelRuntime"]] = {}

    def __new__(cls, *args, **kwargs):
        def get_value(value):
            if isinstance(value, dtypes.DataType):
                return str(value)
            if isinstance(value, list):
                value = tuple(value)
            return value

        args_items = tuple(get_value(x) for x in args)
        kwargs_items = tuple((key, get_value(kwargs[key])) for key in sorted(kwargs.keys()))
        signature = (cls.__name__, cls.current_context(), args_items + kwargs_items)

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

    @staticmethod
    def current_context():
        result, context = cbd.cuCtxGetCurrent()
        assert result == 0, repr(result)
        return context

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
        self.kernel_filename = kernel_filename
        if threading.current_thread() is threading.main_thread():
            self.load_cubin()

    @staticmethod
    def compile_many(kernel_specs, device=None):
        def compile_kernel(spec):
            kernel_type, kernel_args = spec
            return kernel_type(**kernel_args)

        kernel_specs = list(kernel_specs)
        parallel = len(kernel_specs) > 1
        parallel &= os.environ.get("HUMMING_DISABLE_PARALLEL_BUILD", "0") != "1"
        if parallel:
            device = torch.cuda.current_device() if device is None else device
            workers = min(16, len(kernel_specs))
            with ThreadPoolExecutor(
                max_workers=workers,
                initializer=torch.cuda.set_device,
                initargs=(device,),
            ) as executor:
                kernels = list(executor.map(compile_kernel, kernel_specs))
        else:
            kernels = [compile_kernel(spec) for spec in kernel_specs]

        for kernel in kernels:
            kernel.load_cubin()
        return kernels

    def postprocess_cubin(self, cubin_path: str):
        pass

    def load_cubin(self):
        if self.cubin_loaded:
            return None
        kernel_filename = self.kernel_filename
        result, module = cbd.cuModuleLoad(kernel_filename.encode())
        assert result == 0, repr(result)
        matched = [name for name in get_cubin_kernel_names(kernel_filename) if self.name in name]
        assert len(matched) == 1, (kernel_filename, self.name, matched)
        self.kernel_name = matched[0]
        result, func = cbd.cuModuleGetFunction(module, self.kernel_name.encode())
        assert result == 0, repr(result)
        self.module = module
        self.func = func
        self.kernel = func
        self.cubin_loaded = True

    def check_context(self):
        assert threading.current_thread() is threading.main_thread()
        if not self.cubin_loaded:
            self.load_cubin()

    def set_pdl_launch_attribute(self, config: cbd.CUlaunchConfig, enabled: bool) -> None:
        if not enabled or self.sm_version < 90:
            return
        attr = cbd.CUlaunchAttribute()
        attr.id = cbd.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
        attr.value.programmaticStreamSerializationAllowed = 1
        config.attrs = [attr]
        config.numAttrs = 1

    def __call__(self, *args, **kwargs):
        raise NotImplementedError
