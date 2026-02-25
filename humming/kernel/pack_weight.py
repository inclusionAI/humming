import ctypes
import torch
import cuda.bindings.driver as cbd
from typing import Optional
from humming.jit.runtime import KernelRuntime


CODE_TEMPLATE = """
#include <humming/kernel/pack_weight.cuh>

auto ptr = reinterpret_cast<void*>(&pack_weight_kernel<{num_bits}>);
"""


class PackWeightKernel(KernelRuntime):
    name = "pack_weight"

    def __init__(self, num_bits, sm_version=None, device_index=None):
        if self.inited:
            return
        self._set_sm_version(sm_version, device_index)
        self.num_bits = num_bits
        self.code = CODE_TEMPLATE.format(num_bits=num_bits)
        self.arg_types = (ctypes.c_void_p, ctypes.c_void_p)
        self.prepare()

    def __call__(self, inputs: torch.Tensor, outputs: Optional[torch.Tensor] = None):
        assert inputs.is_cuda
        assert inputs.is_contiguous()
        shape_k = inputs.size(-1)
        assert shape_k * self.num_bits % 32 == 0
        assert inputs.dtype == torch.int32
        device = inputs.device

        output_shape = inputs.shape[:-1] + (shape_k * self.num_bits // 32,)

        if outputs is None:
            outputs = torch.empty(output_shape, dtype=torch.int32, device=device)
        else:
            assert outputs.is_contiguous()
            assert outputs.shape == output_shape
            assert outputs.device.index == device.index

        assert inputs.nelement() % (32 * 32) == 0

        config = cbd.CUlaunchConfig()
        config.gridDimX = inputs.nelement() // (32 * 32)
        config.gridDimY = 1
        config.gridDimZ = 1
        config.blockDimX = 32
        config.blockDimY = 1
        config.blockDimZ = 1
        config.hStream = torch.cuda.current_stream(device).cuda_stream

        arg_values = (inputs.data_ptr(), outputs.data_ptr())

        cbd.cuLaunchKernelEx(config, self.kernel, (arg_values, self.arg_types), 0)
        return outputs
