import ctypes
import torch
import cuda.bindings.driver as cbd
from typing import Optional
from humming.jit.runtime import KernelRuntime


CODE_TEMPLATE = """
#include <humming/kernel/pack_weight.cuh>

auto ptr = reinterpret_cast<void*>(&unpack_weight_kernel<{num_bits}>);
"""


class UnpackWeightKernel(KernelRuntime):
    name = "unpack_weight"

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
        assert inputs.size(-1) % self.num_bits == 0
        assert inputs.dtype == torch.int32

        shape_k = inputs.size(-1) // self.num_bits * 32
        output_shape = inputs.shape[:-1] + (shape_k,)

        if outputs is None:
            outputs = torch.empty(output_shape, dtype=torch.int32, device=inputs.device)
        else:
            assert outputs.is_contiguous()
            assert outputs.shape == output_shape
            assert outputs.device.index == inputs.device.index

        assert outputs.nelement() % (32 * 32) == 0

        device = inputs.device
        config = cbd.CUlaunchConfig()
        config.gridDimX = outputs.nelement() // (32 * 32)
        config.gridDimY = 1
        config.gridDimZ = 1
        config.blockDimX = 32
        config.blockDimY = 1
        config.blockDimZ = 1
        config.hStream = torch.cuda.current_stream(device).cuda_stream

        arg_values = (inputs.data_ptr(), outputs.data_ptr())

        cbd.cuLaunchKernelEx(config, self.kernel, (arg_values, self.arg_types), 0)
        return outputs
