import ctypes
import math
from typing import Optional

import cuda.bindings.driver as cbd
import torch

import humming.dtypes as dtypes
from humming.config import (
    EpilogueConfig,
    MmaConfig,
    MoEConfig,
    PipelineConfig,
    QuantParamConfig,
    SchedulerConfig,
)
from humming.config.enum import MmaType
from humming.config.mma import MmaOpClass
from humming.dtypes import DataType
from humming.jit.runtime import KernelRuntime
from humming.utils import is_power_of_two
from humming.utils.tma import make_tma_desc

CODE_TEMPLATE = """
#if {use_warp_spec}
#include <humming/kernel/humming_ws.cuh>
#else
#include <humming/kernel/humming.cuh>
#endif

class MmaOpClass {{
public:
{mma_op_class}
}};

class SchedulerConfig {{
public:
{scheduler_config}
}};

class PipelineConfig {{
public:
{pipeline_config}
}};

class EpilogueConfig {{
public:
{epilogue_config}
}};

class QuantParamConfig {{
public:
{quant_param_config}
}};

class MoEConfig {{
public:
{moe_config}
}};

{custom_activation_func}

using SharedStorageType = SharedStorage<
    MmaOpClass,
    Shape<{block_m}, {block_n}, {block_k}>,
    Shape<{warp_m}, {warp_n}, {warp_k}>,
    {a_dtype},
    {b_dtype},
    {bs_dtype},
    PipelineConfig,
    EpilogueConfig,
    QuantParamConfig,
    MoEConfig>;

extern "C" __constant__ uint32_t SMEM_SIZE = sizeof(SharedStorageType);
extern "C" __constant__ uint32_t SMEM_SIZE_A = sizeof(SharedStorageType::a);
extern "C" __constant__ uint32_t SMEM_SIZE_B = sizeof(SharedStorageType::b);
extern "C" __constant__ uint32_t SMEM_SIZE_REDUCE = sizeof(SharedStorageType::reduce);

auto ptr = reinterpret_cast<void*>(&humming<
    MmaOpClass,
    Shape<{shape_m}, {shape_n}, {shape_k}>,
    Shape<{block_m}, {block_n}, {block_k}>,
    Shape<{warp_m}, {warp_n}, {warp_k}>,
    {a_dtype},
    {b_dtype},
    {c_dtype},
    {bs_dtype},
    SchedulerConfig,
    PipelineConfig,
    EpilogueConfig,
    QuantParamConfig,
    MoEConfig>);

"""


class HummingKernel(KernelRuntime):
    name = "humming"

    def __init__(
        self, problem_shape, block_shape, warp_shape, a_dtype, b_dtype, c_dtype, bs_dtype, **kwargs
    ):
        sm_version = kwargs.get("sm_version", None)
        device_index = kwargs.get("device_index", None)
        self._set_sm_version(sm_version, device_index)
        self.problem_shape = (0,) + tuple(problem_shape)[1:]
        self.block_shape = tuple(block_shape)
        self.warp_shape = tuple(warp_shape)
        self.num_warps = math.prod(block_shape) // math.prod(warp_shape)
        self.num_math_threads = self.num_warps * 32

        self.a_dtype = DataType.from_str(a_dtype)
        self.b_dtype = DataType.from_str(b_dtype)
        self.c_dtype = DataType.from_str(c_dtype)
        self.bs_dtype = DataType.from_str(bs_dtype)

        config_dict = kwargs.copy()
        config_dict.update(self.__dict__)

        self.scheduler_config = SchedulerConfig.from_dict(config_dict)
        self.pipeline_config = PipelineConfig.from_dict(config_dict)
        self.epilogue_config = EpilogueConfig.from_dict(config_dict)
        self.quant_param_config = QuantParamConfig.from_dict(config_dict)
        self.moe_config = MoEConfig.from_dict(config_dict)
        self.mma_config = MmaConfig.from_dict(config_dict)
        self.num_threads = self.pipeline_config.num_threads

        self.check_shape()
        self.check_dtype()
        self.check_config()
        self.mma_op_class = self.select_mma_op_class()

        epilogue_config = self.epilogue_config
        custom_activation_func = epilogue_config.prepare_custom_activation_func(
            kwargs.get("custom_activation_func_impl", None)
        )
        self.custom_activation_func = custom_activation_func

        self.code = CODE_TEMPLATE.format(
            use_warp_spec=int(self.pipeline_config.use_warp_spec),
            mma_op_class=self.mma_op_class.to_cpp_str(),
            shape_m=self.problem_shape[0],
            shape_n=self.problem_shape[1],
            shape_k=self.problem_shape[2],
            block_m=self.block_shape[0],
            block_n=self.block_shape[1],
            block_k=self.block_shape[2],
            warp_m=self.warp_shape[0],
            warp_n=self.warp_shape[1],
            warp_k=self.warp_shape[2],
            a_dtype=self.a_dtype.to_cpp_str(),
            b_dtype=self.b_dtype.to_cpp_str(),
            c_dtype=self.c_dtype.to_cpp_str(),
            bs_dtype=self.bs_dtype.to_cpp_str(),
            scheduler_config=self.scheduler_config.to_cpp_str(),
            pipeline_config=self.pipeline_config.to_cpp_str(),
            epilogue_config=self.epilogue_config.to_cpp_str(),
            quant_param_config=self.quant_param_config.to_cpp_str(),
            custom_activation_func=custom_activation_func,
            moe_config=self.moe_config.to_cpp_str(),
        )

        self.prepare()
        self.arg_types = (
            None if self.pipeline_config.use_tma_a else ctypes.c_void_p,
            None if self.pipeline_config.use_tma_b else ctypes.c_void_p,
            None if self.pipeline_config.use_tma_c else ctypes.c_void_p,
            ctypes.c_void_p,
            None if self.pipeline_config.use_tma_bs else ctypes.c_void_p,
            None if self.pipeline_config.use_tma_bzp else ctypes.c_void_p,
            None if self.pipeline_config.use_tma_bias else ctypes.c_void_p,
        )
        self.arg_types += (ctypes.c_void_p,) * 6 + (ctypes.c_uint32,)
        self.smem_size = self.get_cubin_symbol_value("SMEM_SIZE")

    def select_mma_op_class(self):
        if self.a_dtype in [dtypes.int4, dtypes.int8]:
            mma_cd_dtype = dtypes.int32
        elif self.mma_config.use_f16_accum:
            mma_cd_dtype = self.c_dtype
        else:
            mma_cd_dtype = dtypes.float32

        mma_shape_m = 64 if self.mma_config.mma_type == MmaType.WGMMA else 16
        mma_shape_n = self.warp_shape[0] if self.mma_config.mma_type == MmaType.WGMMA else 8
        mma_shape_k = 256 // self.a_dtype.num_bits

        input_group_size = self.problem_shape[2]
        weight_group_size = self.problem_shape[2]
        scale_config = self.quant_param_config
        if scale_config.has_input_scale and scale_config.input_scale_group_size > 0:
            input_group_size = self.quant_param_config.input_scale_group_size
        if scale_config.has_weight_scale and scale_config.weight_scale_group_size > 0:
            weight_group_size = self.quant_param_config.weight_scale_group_size
        assert min(input_group_size, weight_group_size) >= mma_shape_k // 2
        if min(input_group_size, weight_group_size) == mma_shape_k // 2:
            mma_shape_k = mma_shape_k // 2

        return MmaOpClass.from_config(
            self.mma_config.mma_type,
            mma_shape_m,
            mma_shape_n,
            mma_shape_k,
            self.a_dtype,
            self.a_dtype,
            mma_cd_dtype,
        )

    def check_shape(self):
        assert self.problem_shape[1] % self.block_shape[1] == 0
        assert self.problem_shape[2] % self.block_shape[2] == 0
        assert self.block_shape[0] % self.warp_shape[0] == 0
        assert self.block_shape[1] % self.warp_shape[1] == 0
        assert self.block_shape[2] % self.warp_shape[2] == 0

        assert self.warp_shape[1] % 16 == 0
        assert is_power_of_two(self.block_shape[1])
        assert is_power_of_two(self.block_shape[2])
        assert is_power_of_two(self.warp_shape[1])
        assert is_power_of_two(self.warp_shape[2])
        assert is_power_of_two(self.block_shape[0] // self.warp_shape[0])
        assert is_power_of_two(self.block_shape[1] // self.warp_shape[1])
        assert is_power_of_two(self.block_shape[2] // self.warp_shape[2])

        assert self.warp_shape[1] <= 64
        if self.a_dtype.num_bits == 16:
            assert self.warp_shape[1] == 64
            assert self.warp_shape[2] >= 32
        elif self.a_dtype.num_bits == 8:
            assert self.warp_shape[1] >= 32
            assert self.warp_shape[2] >= 64
        elif self.a_dtype.num_bits == 4:
            assert self.warp_shape[1] >= 16
            assert self.warp_shape[2] >= 128

    def check_dtype(self):
        dtype_map = {
            dtypes.int4: 80,
            dtypes.int8: 75,
            dtypes.float4e2m1: 120,
            dtypes.float8e4m3: 89,
            dtypes.float8e5m2: 89,
            dtypes.bfloat16: 80,
            dtypes.float16: 75,
        }
        assert self.a_dtype in dtype_map
        assert self.sm_version >= dtype_map[self.a_dtype]
        assert self.b_dtype.num_bits <= 8
        assert self.b_dtype.num_bits <= self.a_dtype.num_bits
        if self.b_dtype.is_integer_type and self.a_dtype.is_integer_type:
            if self.a_dtype.num_bits == self.b_dtype.num_bits:
                assert self.a_dtype == self.b_dtype
            else:
                assert not self.b_dtype.is_signed
        elif self.b_dtype.is_integer_type and self.a_dtype.is_floating_point_type:
            assert not self.b_dtype.is_signed
            if self.quant_param_config.has_dynamic_zero_point:
                assert self.b_dtype.num_bits <= self.a_dtype.mantissa_bits + 1
            else:
                assert self.b_dtype.num_bits <= self.a_dtype.mantissa_bits + 2
        elif self.b_dtype.is_floating_point_type and self.a_dtype.is_floating_point_type:
            assert self.b_dtype.is_signed
            assert self.b_dtype.exponent_bits <= self.a_dtype.exponent_bits
            assert self.b_dtype.mantissa_bits <= self.a_dtype.mantissa_bits
            assert self.b_dtype.exponent_bits >= 1
        elif self.b_dtype.is_floating_point_type and not self.a_dtype.is_integer_type:
            # not implemented yet
            assert False

        if self.mma_config.use_f16_accum:
            if self.a_dtype == dtypes.float8e4m3:
                assert self.b_dtype.is_integer_type or self.b_dtype.exponent_bits <= 3
            elif self.a_dtype == dtypes.float16:
                pass
            else:
                assert False

    def check_config(self):
        # 16-bit activation don't support input scale
        # for 8bit/4-bit activation, we enable input scale by default
        if self.a_dtype.num_bits == 16:
            assert self.quant_param_config.has_input_scale is not True
        if self.quant_param_config.has_input_scale is None:
            self.quant_param_config.has_input_scale = self.a_dtype.num_bits != 16

        if self.pipeline_config.use_warp_spec:
            assert self.pipeline_config.use_mbarrier
        if not self.quant_param_config.has_weight_scale:
            self.pipeline_config.use_tma_bs = False
        if not self.quant_param_config.has_dynamic_zero_point:
            self.pipeline_config.use_tma_bzp = False
        if not self.epilogue_config.has_bias:
            self.pipeline_config.use_tma_bias = False

    def set_smem_size(self, device_index=0):
        cbd.cuKernelSetAttribute(
            cbd.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            self.smem_size,
            self.kernel,
            cbd.CUdevice(device_index),
        )

    def prepare_inputs_tensor(self, inputs: torch.Tensor):
        if not self.pipeline_config.use_tma_a:
            return inputs.data_ptr()

        num_bits = self.a_dtype.num_bits
        block_bytes_k = num_bits * self.block_shape[2] // 8
        block_size_m = 1 if self.moe_config.is_moe else self.block_shape[0]
        tma_smem_dims_k = self.block_shape[2] if block_bytes_k == 64 else 1024 // num_bits
        swizzle_bytes = 128 if block_bytes_k >= 128 else 64

        return make_tma_desc(
            inputs,
            smem_dims=(tma_smem_dims_k, block_size_m),
            swizzle_bytes=swizzle_bytes,
        )

    def prepare_weight_tensor(self, weight: torch.Tensor, device: torch.device):
        assert weight.device == device
        if not self.pipeline_config.use_tma_b:
            return weight.data_ptr()

        num_bits = self.b_dtype.num_bits
        weight = weight.view(-1, weight.size(-1))
        pack_size_k = 256 // self.a_dtype.num_bits
        weight = weight.view(weight.size(0), -1, num_bits * pack_size_k)
        block_size_k = self.block_shape[2]
        block_size_n = self.block_shape[1]

        return make_tma_desc(
            weight,
            smem_dims=(num_bits * pack_size_k, block_size_n // 32, block_size_k // pack_size_k),
        )

    def prepare_outputs_tensor(self, outputs: torch.Tensor, device: torch.device):
        assert outputs.device == device
        if not self.pipeline_config.use_tma_c:
            return outputs.data_ptr()

        block_size_m = 1 if self.moe_config.is_moe else self.block_shape[0]
        return make_tma_desc(
            outputs,
            smem_dims=(64, block_size_m),
            swizzle_bytes=128,
        )

    def prepare_weight_scale_tensor(
        self, weight_scale: Optional[torch.Tensor], device: torch.device
    ):
        if weight_scale is None:
            assert not self.quant_param_config.has_weight_scale
            return 0
        assert weight_scale.device == device
        if not self.pipeline_config.use_tma_bs:
            return weight_scale.data_ptr()

        weight_scale = weight_scale.view(-1, weight_scale.size(-1))
        weight_scale = weight_scale.view(weight_scale.size(0), -1, 16)
        block_size_n = self.block_shape[1]
        block_size_k = self.block_shape[2]
        num_groups = 1
        if self.quant_param_config.weight_scale_group_size > 0:
            group_size = self.quant_param_config.weight_scale_group_size
            num_groups = math.ceil(block_size_k / group_size)

        return make_tma_desc(
            weight_scale,
            smem_dims=(16, block_size_n // 16, num_groups),
        )

    def prepare_zero_point_tensor(
        self,
        zero_point: Optional[torch.Tensor],
        device: torch.device,
    ):
        if zero_point is None:
            assert not self.quant_param_config.has_dynamic_zero_point
            return 0
        assert zero_point.device == device
        if not self.pipeline_config.use_tma_bzp:
            return zero_point.data_ptr()

        num_bits = 4 if self.b_dtype.num_bits <= 4 else 8
        block_size_n = self.block_shape[1]
        block_size_k = self.block_shape[2]
        num_groups = 1
        if self.quant_param_config.weight_scale_group_size > 0:
            group_size = self.quant_param_config.weight_scale_group_size
            num_groups = math.ceil(block_size_k / group_size)

        return make_tma_desc(
            zero_point,
            smem_dims=(block_size_n * num_bits // 32, num_groups),
        )

    def prepare_bias_tensor(self, bias: Optional[torch.Tensor], device: torch.device):
        if bias is None:
            assert not self.epilogue_config.has_bias
            return 0
        assert bias.device == device
        if not self.pipeline_config.use_tma_bias:
            return bias.data_ptr()

        block_size_n = self.block_shape[1]
        bias = bias.view(-1, 64)
        return make_tma_desc(
            bias,
            smem_dims=(64, block_size_n // 64),
        )

    def __call__(
        self,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        outputs: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
        weight_scale: Optional[torch.Tensor] = None,
        zero_point: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        global_scale: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        sorted_token_ids: Optional[torch.Tensor] = None,
        expert_ids: Optional[torch.Tensor] = None,
        num_tokens_past_padded: Optional[torch.Tensor] = None,
        locks: Optional[torch.Tensor] = None,
        num_ctas_per_sm: int = 1,
        num_sms: Optional[int] = None,
    ):
        assert inputs.is_cuda
        device = inputs.device
        self.set_smem_size(device.index)
        assert weight.device == device
        shape_m = inputs.size(0)

        if self.moe_config.is_moe:
            assert sorted_token_ids is not None
            assert expert_ids is not None
            assert num_tokens_past_padded is not None
            sorted_token_ids.device == device
            expert_ids.device == device
            num_tokens_past_padded.device == device

            if self.moe_config.is_moe_down:
                assert topk_weights is not None
                topk_weights.device == device

        if self.scheduler_config.use_stream_k:
            assert locks is not None
            locks.device == device

        if outputs is None:
            output_shape_m = shape_m
            if self.moe_config.is_moe and not self.moe_config.is_moe_down:
                output_shape_m = shape_m * self.moe_config.top_k
            outputs_shape = (output_shape_m, self.problem_shape[1])
            torch_dtype = dtypes.torch_dtype_map[self.c_dtype]
            outputs = torch.empty(outputs_shape, dtype=torch_dtype, device=device)

        arg_values = (
            self.prepare_inputs_tensor(inputs),
            self.prepare_weight_tensor(weight, device),
            self.prepare_outputs_tensor(outputs, device),
            0 if input_scale is None else input_scale.data_ptr(),
            self.prepare_weight_scale_tensor(weight_scale, device),
            self.prepare_zero_point_tensor(zero_point, device),
            self.prepare_bias_tensor(bias, device),
            0 if global_scale is None else global_scale.data_ptr(),
            0 if topk_weights is None else topk_weights.data_ptr(),
            0 if sorted_token_ids is None else sorted_token_ids.data_ptr(),
            0 if expert_ids is None else expert_ids.data_ptr(),
            0 if num_tokens_past_padded is None else num_tokens_past_padded.data_ptr(),
            0 if locks is None else locks.data_ptr(),
            shape_m,
        )

        if num_sms is None:
            num_sms = torch.cuda.get_device_properties(device).multi_processor_count

        config = cbd.CUlaunchConfig()
        config.gridDimX = num_ctas_per_sm * num_sms
        config.gridDimY = 1
        config.gridDimZ = 1
        config.blockDimX = self.num_threads
        config.blockDimY = 1
        config.blockDimZ = 1
        config.sharedMemBytes = self.smem_size
        config.hStream = torch.cuda.current_stream().cuda_stream

        cbd.cuLaunchKernelEx(config, self.kernel, (arg_values, self.arg_types), device.index)

        if "_GLU" in str(self.epilogue_config.activation_type):
            output_shape_m = outputs.size(0)
            num_valid_elems = output_shape_m * self.problem_shape[1] // 2
            outputs = outputs.view(-1)[:num_valid_elems]
            outputs = outputs.view(output_shape_m, -1)

        return outputs
