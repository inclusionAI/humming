import ctypes
import math
import os
from typing import Optional

import torch
import zlib

import torch.utils.cpp_extension
from humming.jit.utils import make_humming_module

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
import humming.jit.utils as jit_utils

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


extern "C" __constant__ uint32_t SMEM_SIZE = sizeof(SharedStorageType);
extern "C" __constant__ uint32_t SMEM_SIZE_A = sizeof(SharedStorageType::a);
extern "C" __constant__ uint32_t SMEM_SIZE_B = sizeof(SharedStorageType::b);
extern "C" __constant__ uint32_t SMEM_SIZE_REDUCE = sizeof(SharedStorageType::reduce);

extern "C" __constant__ uint32_t PROBLEM_SHAPE_M = {shape_m};
extern "C" __constant__ uint32_t PROBLEM_SHAPE_N = {shape_n};
extern "C" __constant__ uint32_t PROBLEM_SHAPE_K = {shape_k};

extern "C" __constant__ uint32_t BLOCK_SHAPE_M = {block_m};
extern "C" __constant__ uint32_t BLOCK_SHAPE_N = {block_n};
extern "C" __constant__ uint32_t BLOCK_SHAPE_K = {block_k};

extern "C" __constant__ uint32_t WARP_SHAPE_M = {warp_m};
extern "C" __constant__ uint32_t WARP_SHAPE_N = {warp_n};
extern "C" __constant__ uint32_t WARP_SHAPE_K = {warp_k};

extern "C" __constant__ uint32_t A_DTYPE_ID = {a_dtype}::kId;
extern "C" __constant__ uint32_t B_DTYPE_ID = {b_dtype}::kId;
extern "C" __constant__ uint32_t C_DTYPE_ID = {c_dtype}::kId;
extern "C" __constant__ uint32_t BS_DTYPE_ID = {bs_dtype}::kId;

extern "C" __constant__ uint32_t IS_GLU_ACTIVATION = {is_glu_activation};

{scheduler_config_extern}

{pipeline_config_extern}

{epilogue_config_extern}

{quant_param_config_extern}

{moe_config_extern}

"""


def init_humming_launcher():
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "../csrc/launcher/launcher.cpp")
    filename = os.path.abspath(filename)

    torch.utils.cpp_extension.load(
        name="humming_extension",
        sources=[filename],
        extra_include_paths=["/usr/local/cuda/include"],
        extra_ldflags=[
            "-lcuda",
            "-L/usr/local/cuda/lib64",
            "-lc10_cuda",
            "-ltorch_cuda",
        ],
        extra_cflags=["-O3"],
    )


class HummingKernel(KernelRuntime):
    name = "humming"

    def __init__(
        self,
        problem_shape,
        block_shape,
        warp_shape,
        a_dtype,
        b_dtype,
        c_dtype,
        bs_dtype,
        **kwargs,
    ):
        if self.inited:
            return
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
        is_glu_activation = "_glu" in str(self.epilogue_config.activation_type).lower()
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
            moe_config=self.moe_config.to_cpp_str(),
            scheduler_config_extern=self.scheduler_config.to_extern_cpp_str(),
            pipeline_config_extern=self.pipeline_config.to_extern_cpp_str(),
            epilogue_config_extern=self.epilogue_config.to_extern_cpp_str(),
            quant_param_config_extern=self.quant_param_config.to_extern_cpp_str(),
            moe_config_extern=self.moe_config.to_extern_cpp_str(),
            is_glu_activation=int(is_glu_activation),
            custom_activation_func=custom_activation_func,
        )

        init_humming_launcher()
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
        self.torch_dtype = dtypes.torch_dtype_map[self.c_dtype]
        self.prepare()
        self.smem_size = self.get_cubin_symbol_value("SMEM_SIZE")

    def load_cubin(self, kernel_filename, kernel_name):
        self.kernel_id = torch.ops.humming.register_kernel(kernel_filename, kernel_name)
        self.kernel_dirname = os.path.dirname(kernel_filename)
        ref_kernel_id = zlib.crc32(kernel_filename.encode()) << 30
        ref_kernel_id += zlib.crc32(kernel_name.encode())
        assert ref_kernel_id == self.kernel_id
        module = make_humming_module("get_kernel_id", self.kernel_id)
        self.get_kernel_id = module.get_kernel_id

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
        assert jit_utils.is_power_of_two(self.block_shape[1])
        assert jit_utils.is_power_of_two(self.block_shape[2])
        assert jit_utils.is_power_of_two(self.warp_shape[1])
        assert jit_utils.is_power_of_two(self.warp_shape[2])
        assert jit_utils.is_power_of_two(self.block_shape[0] // self.warp_shape[0])
        assert jit_utils.is_power_of_two(self.block_shape[1] // self.warp_shape[1])
        assert jit_utils.is_power_of_two(self.block_shape[2] // self.warp_shape[2])

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
        num_tokens_post_padded: Optional[torch.Tensor] = None,
        locks: Optional[torch.Tensor] = None,
        num_ctas_per_sm: int = 1,
        num_sms: Optional[int] = None,
    ):

        # We need to integrate the module containing the kernel_id
        # into the forward path. This ensures that when the kernel changes,
        # torch.compile can recognize it and update the cache accordingly.
        return torch.ops.humming.launch_humming(
            [self.get_kernel_id()],
            inputs,
            weight,
            outputs,
            input_scale,
            weight_scale,
            zero_point,
            bias,
            global_scale,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            locks,
            num_ctas_per_sm,
            num_sms,
        )
