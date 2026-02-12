import torch
import math
import ctypes
import cuda.bindings.driver as cbd
from typing import Optional
from humming.jit.runtime import KernelRuntime


CODE_TEMPLATE = """
#include <humming/kernel/dequant_weight.cuh>

auto ptr = reinterpret_cast<void*>(&dequant_unpacked_fp_type);
"""


class DequantKernel(KernelRuntime):
    name = "dequant_unpacked_fp_type"

    def __init__(self, sm_version=None, device_index=None):
        self._set_sm_version(sm_version, device_index)
        self.code = CODE_TEMPLATE
        self.arg_types = (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_bool,
        )
        self.prepare()

    def __call__(
        self,
        inputs: torch.Tensor,
        outputs: Optional[torch.Tensor],
        exponent_bits: int,
        mantissa_bits: int,
        is_signed: bool,
    ):
        assert inputs.dtype == torch.int32
        assert inputs.is_cuda
        assert inputs.is_contiguous()
        if outputs is None:
            outputs = torch.empty_like(inputs, dtype=torch.float32)
        else:
            assert outputs.is_contiguous()
            assert outputs.shape == inputs.shape
            assert outputs.device.index == inputs.device.index

        device = inputs.device
        total_size = inputs.nelement()
        config = cbd.CUlaunchConfig()
        config.gridDimX = math.ceil(total_size / (32 * 32))
        config.gridDimY = 1
        config.gridDimZ = 1
        config.blockDimX = 32
        config.blockDimY = 1
        config.blockDimZ = 1
        config.hStream = torch.cuda.current_stream().cuda_stream

        arg_values = (
            inputs.data_ptr(),
            outputs.data_ptr(),
            total_size,
            exponent_bits,
            mantissa_bits,
            is_signed,
        )

        cbd.cuLaunchKernelEx(config, self.kernel, (arg_values, self.arg_types), device.index)
        return outputs
