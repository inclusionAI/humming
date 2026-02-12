import torch
import ctypes
import math
import cuda.bindings.driver as cbd
from typing import Optional
from humming.jit.runtime import KernelRuntime


CODE_TEMPLATE = """
#include <humming/kernel/process.cuh>

auto ptr = reinterpret_cast<void*>(&weight_repack_nk<
    {weight_bits},
    {activation_bits},
    {is_weight_pakced},
    {should_preprocess_for_int2fp},
    {should_preprocess_with_zp},
    {group_size_zp}
  >);
"""


class WeightRepackKernel(KernelRuntime):
    name = "weight_repack_nk"

    def __init__(
        self,
        weight_bits,
        activation_bits,
        is_weight_pakced,
        should_preprocess_for_int2fp=False,
        should_preprocess_with_zp=False,
        group_size_zp=0,
        sm_version=None,
        device_index=None,
    ):
        self._set_sm_version(sm_version, device_index)
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.is_weight_pakced = is_weight_pakced
        self.should_preprocess_with_zp = should_preprocess_with_zp
        self.group_size_zp = group_size_zp

        if self.should_preprocess_with_zp:
            assert should_preprocess_for_int2fp

        self.code = CODE_TEMPLATE.format(
            weight_bits=weight_bits,
            activation_bits=activation_bits,
            is_weight_pakced=int(is_weight_pakced),
            should_preprocess_for_int2fp=int(should_preprocess_for_int2fp),
            should_preprocess_with_zp=int(should_preprocess_with_zp),
            group_size_zp=group_size_zp,
        )
        self.arg_types = (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
        )
        self.prepare()

    def __call__(
        self,
        inputs: torch.Tensor,
        outputs: Optional[torch.Tensor] = None,
        zero_point: Optional[torch.Tensor] = None,
        padded_shape_n: Optional[int] = None,
        padded_shape_k: Optional[int] = None,
    ):
        assert inputs.ndim in [2, 3]
        assert inputs.is_cuda
        assert inputs.is_contiguous()
        assert inputs.dtype == torch.int32
        device = inputs.device
        num_experts = 1 if inputs.ndim == 2 else inputs.size(0)
        shape_n = inputs.size(-2)
        shape_k = inputs.size(-1)
        if self.is_weight_pakced:
            assert shape_k * 32 % self.weight_bits == 0
            shape_k = shape_k * 32 // self.weight_bits

        padded_shape_n = shape_n if padded_shape_n is None else padded_shape_n
        padded_shape_k = shape_k if padded_shape_k is None else padded_shape_k
        assert shape_n <= padded_shape_n
        assert shape_k <= padded_shape_k

        if self.should_preprocess_with_zp:
            assert zero_point is not None
            group_size_zp = shape_k if self.group_size_zp == 0 else self.group_size_zp
            zero_point_shape = inputs.shape[:-1] + (math.ceil(shape_k / group_size_zp),)

            if self.is_weight_pakced:
                assert shape_n * self.weight_bits % 32 == 0
                packed_shape_n = shape_n * self.weight_bits // 32
                zero_point_shape = zero_point_shape[:-2] + (packed_shape_n,) + zero_point_shape[-1:]

            assert zero_point.shape == zero_point_shape

        pack_size_k = 256 // self.activation_bits
        output_shape = (
            shape_k // pack_size_k,
            shape_n * pack_size_k * self.weight_bits // 32,
        )
        if inputs.ndim == 3:
            output_shape = (num_experts,) + output_shape

        if outputs is None:
            outputs = torch.empty(output_shape, dtype=torch.int32, device=device)
        else:
            assert outputs.device == device
            assert outputs.dtype == torch.int32
            assert outputs.shape == output_shape

        config = cbd.CUlaunchConfig()
        config.gridDimX = math.ceil(shape_n / 64)
        config.gridDimY = math.ceil(shape_k / 64)
        config.gridDimZ = num_experts
        config.blockDimX = 32
        config.blockDimY = 1
        config.blockDimZ = 1
        config.hStream = torch.cuda.current_stream().cuda_stream

        arg_values = (
            inputs.data_ptr(),
            outputs.data_ptr(),
            0 if zero_point is None else zero_point.data_ptr(),
            shape_n,
            shape_k,
            padded_shape_n,
            padded_shape_k,
        )

        cbd.cuLaunchKernelEx(config, self.kernel, (arg_values, self.arg_types), device.index)
        return outputs
