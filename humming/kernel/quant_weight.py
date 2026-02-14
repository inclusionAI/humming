import ctypes
import torch
import cuda.bindings.driver as cbd
from typing import Optional
from humming.jit.runtime import KernelRuntime


CODE_TEMPLATE = """
#include <humming/kernel/quant_weight.cuh>

auto ptr = reinterpret_cast<void*>(&quant_weight<
    {source_dtype},
    {target_dtype},
    {group_size},
    {has_scale},
    {use_e8m0_scale},
    {has_dynamic_zero_point}
  >);
"""


class QuantWeightKernel(KernelRuntime):
    name = "quant_weight"

    def __init__(
        self,
        source_dtype,
        target_dtype,
        group_size,
        has_scale,
        use_e8m0_scale,
        has_dynamic_zero_point=False,
        sm_version=None,
        device_index=None,
    ):
        if hasattr(self, "sm_version"):
            return
        self._set_sm_version(sm_version, device_index)
        self.group_size = group_size
        self.has_scale = has_scale
        self.use_e8m0_scale = use_e8m0_scale
        self.has_dynamic_zero_point = has_dynamic_zero_point
        self.code = CODE_TEMPLATE.format(
            source_dtype=source_dtype.to_cpp_str(),
            target_dtype=target_dtype.to_cpp_str(),
            group_size=group_size,
            has_scale=int(has_scale),
            use_e8m0_scale=int(use_e8m0_scale),
            has_dynamic_zero_point=int(has_dynamic_zero_point),
        )
        self.arg_types = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
        self.prepare()

    def __call__(
        self,
        inputs: torch.Tensor,
        outputs: Optional[torch.Tensor] = None,
        scales: Optional[torch.Tensor] = None,
        zero_point: Optional[torch.Tensor] = None,
    ):
        group_size = self.group_size
        if group_size <= 0:
            group_size = inputs.size(-1)

        assert inputs.is_cuda
        assert inputs.is_contiguous()
        if outputs is None:
            outputs = torch.empty_like(inputs, dtype=torch.float32)
        else:
            assert outputs.is_contiguous()
            assert outputs.shape == inputs.shape
            assert outputs.device.index == inputs.device.index

        if self.has_scale:
            scale_shape = inputs.shape[:-1] + (inputs.size(-1) // group_size,)

            if scales is None:
                scales = torch.empty(scale_shape, device=inputs.device, dtype=torch.float32)
            else:
                assert scales.is_contiguous()
                if self.use_e8m0_scale:
                    assert scales.dtype == torch.float8_e8m0fnu
                else:
                    assert scales.dtype == torch.float32
                assert scales.shape == scale_shape
                assert scales.device.index == inputs.device.index

            if self.has_dynamic_zero_point and zero_point is None:
                zero_point = torch.empty(scale_shape, device=inputs.device, dtype=torch.int32)
            elif self.has_dynamic_zero_point:
                assert zero_point.is_contiguous()
                assert zero_point.dtype == torch.int32
                assert zero_point.shape == scale_shape
                assert zero_point.device.index == inputs.device.index

        device_index = inputs.device.index
        config = cbd.CUlaunchConfig()
        config.gridDimX = inputs.nelement() // group_size
        config.gridDimY = 1
        config.gridDimZ = 1
        config.blockDimX = 32
        config.blockDimY = 1
        config.blockDimZ = 1
        config.hStream = torch.cuda.current_stream().cuda_stream

        arg_values = (
            inputs.data_ptr(),
            outputs.data_ptr(),
            0 if scales is None else scales.data_ptr(),
            0 if zero_point is None else zero_point.data_ptr(),
        )

        cbd.cuLaunchKernelEx(config, self.kernel, (arg_values, self.arg_types), device_index)
        return outputs, scales, zero_point
