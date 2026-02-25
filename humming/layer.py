import dataclasses
from typing import Optional

import torch

import humming.kernel.quant_input  # noqa: F401
from humming import dtypes
from humming.jit.utils import make_humming_module
from humming.kernel.humming import HummingKernel
from humming.kernel.pack_weight import PackWeightKernel
from humming.utils.weight import (
    prepare_humming_bias,
    prepare_humming_weight,
    prepare_humming_weight_scale,
    prepare_humming_zero_point,
    quantize_weight,
)


@dataclasses.dataclass
class HummingLayerMeta(object):
    a_dtype: dtypes.DataType
    b_dtype: dtypes.DataType
    c_dtype: dtypes.DataType
    bs_dtype: dtypes.DataType
    shape_n: int
    shape_k: int
    pad_shape_n: int = 0
    pad_shape_k: int = 0
    num_experts: Optional[int] = None
    has_input_scale: Optional[bool] = None
    has_weight_scale: bool = True
    input_scale_group_size: int = 0
    weight_scale_group_size: int = 0
    has_dynamic_zp: bool = False
    has_bias: bool = False
    has_global_scale: bool = False
    mma_type: Optional[str] = "mma"
    sublayer_name: str = ""

    @property
    def name_prefix(self):
        return self.sublayer_name + "_" if self.sublayer_name else ""

    @property
    def weight_name(self):
        return self.name_prefix + "weight"

    @property
    def zero_point_name(self):
        return self.name_prefix + "zero_point"

    @property
    def weight_scale_name(self):
        return self.name_prefix + "weight_scale"

    @property
    def global_scale_name(self):
        return self.name_prefix + "global_scale"

    @property
    def bias_name(self):
        return self.name_prefix + "bias"

    def __post_init__(self):
        if self.a_dtype.num_bits != 16 and self.has_input_scale is None:
            self.has_input_scale = True


class HummingMethod(torch.nn.Module):
    @classmethod
    def set_param(cls, layer: torch.nn.Module, name: str, data: torch.Tensor):
        setattr(layer, name, torch.nn.Parameter(data, requires_grad=False))

    @classmethod
    def set_param_data(
        cls,
        layer: torch.nn.Module,
        name: str,
        data: torch.Tensor,
        offset: Optional[int] = None,
        expert_id: Optional[int] = None,
    ):
        param = getattr(layer, name, None)
        if param is None:
            return

        assert data.dtype == param.dtype

        no_equals = data.nelement() != param.nelement() 
        if expert_id is None and data.size(0) == param.size(0) and no_equals:
            for expert_id in range(data.size(0)):
                data_tmp = data[expert_id].to(param.device).view(-1)
                part_tensor = param.data[expert_id]
                part_tensor = part_tensor.view(-1)[offset or 0 :]
                part_tensor[: data_tmp.size(0)] = data_tmp
        else:
            data = data.to(param.device).view(-1)
            part_tensor = param.data if expert_id is None else param.data[expert_id]
            part_tensor = part_tensor.view(-1)[offset or 0 :]
            part_tensor[: data.size(0)] = data

    @classmethod
    def create_weights(cls, layer: torch.nn.Module, meta: HummingLayerMeta):
        packed_size_k = 256 // meta.a_dtype.num_bits

        repacked_shape_k = meta.shape_k // packed_size_k
        repacked_shape_n = meta.shape_n * packed_size_k * meta.b_dtype.num_bits // 32
        weight_shape = (repacked_shape_k, repacked_shape_n)
        group_size = meta.weight_scale_group_size or meta.shape_k

        num_groups = meta.shape_k // group_size
        weight_scale_shape = (num_groups, meta.shape_n)
        num_zp_bits = 4 if meta.b_dtype.num_bits <= 4 else 8
        zero_point_shape = (num_groups, meta.shape_n * num_zp_bits // 32)
        bias_shape = (meta.shape_n,)
        global_scale_shape = (1,)

        if meta.num_experts is not None:
            weight_shape = (meta.num_experts,) + weight_shape
            weight_scale_shape = (meta.num_experts,) + weight_scale_shape
            zero_point_shape = (meta.num_experts,) + zero_point_shape
            bias_shape = (meta.num_experts,) + bias_shape
            global_scale_shape = (meta.num_experts,) + global_scale_shape

        weight = torch.empty(weight_shape, dtype=torch.int32)
        cls.set_param(layer, meta.weight_name, weight)

        if meta.has_weight_scale:
            torch_dtype = dtypes.torch_dtype_map[meta.bs_dtype]
            weight_scale = torch.empty(weight_scale_shape, dtype=torch_dtype)
            cls.set_param(layer, meta.weight_scale_name, weight_scale)

            if meta.has_dynamic_zp:
                zero_point = torch.empty(zero_point_shape, dtype=torch.int32)
                cls.set_param(layer, meta.zero_point_name, zero_point)

        if meta.has_global_scale:
            global_scale = torch.empty(global_scale_shape, dtype=torch.float32)
            cls.set_param(layer, meta.global_scale_name, global_scale)

        if meta.has_bias:
            torch_dtype = dtypes.torch_dtype_map[meta.c_dtype]
            bias = torch.empty(bias_shape, dtype=torch_dtype)
            cls.set_param(layer, meta.bias_name, bias)

        layer.locks = torch.nn.Buffer(torch.zeros(1024, dtype=torch.int32))
        if not hasattr(layer, "humming_metas"):
            layer.humming_metas = {}

        layer.humming_metas[meta.sublayer_name] = meta

    @classmethod
    def pad_tensor(cls, tensor: torch.Tensor, target_size: int) -> torch.Tensor:
        pad_value = 0 if tensor.dtype != torch.float8_e8m0fnu else 127
        return torch.nn.functional.pad(
            tensor,
            pad=(0, target_size - tensor.size(-1)),
            value=pad_value,
        )

    @classmethod
    def load_weight(
        cls,
        layer: torch.nn.Module,
        weight: Optional[torch.Tensor] = None,
        weight_scale: Optional[torch.Tensor] = None,
        zero_point: Optional[torch.Tensor] = None,
        global_scale: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        offset_n: Optional[int] = None,
        expert_id: Optional[int] = None,
        sublayer_name: str = "",
        packed: bool = False,
        padded: bool = False,
    ):
        meta = layer.humming_metas[sublayer_name]
        ckpt_shape_n = meta.shape_n if padded else (meta.shape_n - meta.pad_shape_n)
        ckpt_shape_k = meta.shape_k if padded else (meta.shape_k - meta.pad_shape_k)

        float_dtypes = [torch.float16, torch.bfloat16, torch.float32]
        if weight is not None and weight.dtype in float_dtypes:
            assert weight_scale is None
            assert zero_point is None

            weight_shape = (ckpt_shape_n, ckpt_shape_k)
            if meta.num_layers is not None and expert_id is None:
                weight_shape = (meta.num_layers,) + weight_shape

            assert weight.shape == weight_shape
            if not padded:
                weight = cls.pad_tensor(weight, meta.shape_k)
            quanted_weight, weight_scale, zero_point, global_scale = quantize_weight(
                weight,
                dtype=meta.b_dtype,
                scale_dtype=meta.bs_dtype,
                group_size=meta.weight_scale_group_size,
                has_dynamic_zp=meta.has_dynamic_zp,
                has_global_scale=meta.has_global_scale,
            )

            return cls.load_weight(
                layer=layer,
                weight=quanted_weight,
                weight_scale=weight_scale,
                zero_point=zero_point,
                global_scale=global_scale,
                bias=bias,
                offset_n=offset_n,
                expert_id=expert_id,
                sublayer_name=sublayer_name,
            )

        if weight is not None:
            weight_param = getattr(layer, meta.weight_name)
            weight = weight.to(weight_param.device)
            assert weight.dtype == torch.int32

            expected_shape_n = ckpt_shape_n if offset_n is None else weight.size(-2)
            shape = (expected_shape_n, ckpt_shape_k)
            if meta.num_experts is not None and expert_id is None:
                shape = (meta.num_experts,) + shape
            packed_shape = shape[:-1] + (ckpt_shape_k * meta.b_dtype.num_bits // 32,)

            assert weight.shape == (packed_shape if packed else shape)

            offset = (offset_n or 0) * ckpt_shape_k * meta.b_dtype.num_bits // 32

            if not packed:
                kernel = PackWeightKernel(meta.b_dtype.num_bits)
                weight = kernel(weight)

            if not padded:
                target_size = meta.shape_k * meta.b_dtype.num_bits // 32
                weight = cls.pad_tensor(weight, target_size)

            cls.set_param_data(layer, meta.weight_name, weight, offset, expert_id)

        if weight_scale is not None:
            weight_scale_param = getattr(layer, meta.weight_scale_name)
            num_groups = weight_scale_param.size(-2)
            if not padded and num_groups != 1:
                num_groups = num_groups * ckpt_shape_k // meta.shape_k
            expected_shape_n = ckpt_shape_n if offset_n is None else weight_scale.size(-2)
            shape = (expected_shape_n, num_groups)
            if meta.num_experts is not None and expert_id is None:
                shape = (meta.num_experts,) + shape
            assert weight_scale.shape == shape
            weight_scale = weight_scale.to(device=weight_scale_param.device)
            if weight_scale.element_size() == 1 or weight_scale_param.element_size() == 1:
                assert weight_scale.dtype == weight_scale_param.dtype
            else:
                assert weight_scale.dtype in float_dtypes
                assert weight_scale_param.dtype in float_dtypes
                weight_scale = weight_scale.to(dtype=weight_scale_param.dtype)

            if not padded and meta.shape_k != ckpt_shape_k:
                target_size = weight_scale_param.size(-2)
                weight_scale = cls.pad_tensor(weight_scale, target_size)

            offset = (offset_n or 0) * num_groups
            cls.set_param_data(layer, meta.weight_scale_name, weight_scale, offset, expert_id)

        if zero_point is not None:
            zero_point_param = getattr(layer, meta.zero_point_name)
            zero_point = zero_point.to(device=zero_point_param.device)

            if packed:
                expected_shape_n = zero_point.size(-2) * 32 // meta.b_dtype.num_bits
            else:
                expected_shape_n = zero_point.size(-2)
            expected_shape_n = ckpt_shape_n if offset_n is None else expected_shape_n

            num_groups = zero_point_param.size(-2)
            if not padded and num_groups != 1:
                num_groups = num_groups * ckpt_shape_k // meta.shape_k

            shape = (expected_shape_n, num_groups)
            packed_shape = (expected_shape_n * meta.b_dtype.num_bits // 32, num_groups)
            if meta.num_experts is not None and expert_id is None:
                shape = (meta.num_experts,) + shape
                packed_shape = (meta.num_experts,) + packed_shape

            assert zero_point.shape == (packed_shape if packed else shape)
            if not packed:
                zero_point = zero_point.cuda()
                zero_point = zero_point.transpose(-1, -2).contiguous()
                zero_point = zero_point.squeeze().view(*zero_point.shape)
                kernel = PackWeightKernel(meta.b_dtype.num_bits)
                zero_point = kernel(zero_point)
                zero_point = zero_point.transpose(-1, -2).contiguous()
                zero_point = zero_point.squeeze().view(*zero_point.shape)

            if not padded and meta.shape_k != ckpt_shape_k:
                target_size = weight_scale_param.size(-2)
                zero_point = cls.pad_tensor(zero_point, target_size)

            assert zero_point.dtype == torch.int32
            offset = (offset_n or 0) * meta.b_dtype.num_bits // 32 * num_groups
            cls.set_param_data(layer, meta.zero_point_name, zero_point, offset, expert_id)

        if global_scale is not None:
            global_scale_param = getattr(layer, meta.global_scale_name)
            global_scale = global_scale.to(
                device=global_scale_param.device,
                dtype=global_scale_param.dtype,
            )
            cls.set_param_data(layer, meta.global_scale_name, global_scale, expert_id)

        if bias is not None:
            bias_param = getattr(layer, meta.bias_name, None)
            expected_shape_n = ckpt_shape_n if offset_n is None else bias.size(-1)
            shape = (expected_shape_n,)
            if meta.num_experts is not None and expert_id is None:
                shape = (meta.num_experts,) + shape
            assert bias.shape == shape
            assert bias.dtype in [torch.float16, torch.bfloat16, torch.float32]
            bias = bias.to(device=bias_param.device, dtype=bias_param.dtype)
            cls.set_param_data(layer, meta.bias_name, bias, offset_n, expert_id)

    @classmethod
    def finish_load(cls, layer: torch.nn.Module, sublayer_name: str = ""):
        meta = layer.humming_metas[sublayer_name]
        weight = getattr(layer, meta.weight_name)
        weight_scale = getattr(layer, meta.weight_scale_name, None)
        zero_point = getattr(layer, meta.zero_point_name, None)
        bias = getattr(layer, meta.bias_name, None)

        num_experts = meta.num_experts or 1
        weight = weight.view(num_experts, meta.shape_n, -1)
        if zero_point is not None and zero_point.size(0):
            num_groups = zero_point.size(-2)
            padded_bits = 4 if meta.b_dtype.num_bits <= 4 else 8
            zero_point = zero_point.view(num_experts, padded_bits, -1)
            zero_point = zero_point[:, : meta.b_dtype.num_bits].contiguous()
            zero_point = zero_point.view(num_experts, -1, num_groups)
        if weight_scale is not None:
            weight_scale = weight_scale.view(num_experts, meta.shape_n, -1)

        weight = prepare_humming_weight(
            weight=weight,
            b_dtype=meta.b_dtype,
            a_dtype=meta.a_dtype,
            zero_point=zero_point,
            packed=True,
        )

        cls.set_param_data(layer, meta.weight_name, weight)

        if weight_scale is not None and weight_scale.size(0):
            weight_scale_group_size = meta.weight_scale_group_size

            if meta.mma_type == "mma":
                to_apply_on_c = weight_scale_group_size == 0 or meta.a_dtype.num_bits != 16
            elif meta.mma_type == "wgmma":
                to_apply_on_c = weight_scale_group_size == 0

            weight_scale = prepare_humming_weight_scale(
                weight_scale=weight_scale,
                to_apply_on_c=to_apply_on_c,
            )

            cls.set_param_data(layer, meta.weight_scale_name, weight_scale)

        if zero_point is not None and zero_point.size(0):
            b_dtype = meta.b_dtype
            zero_point = prepare_humming_zero_point(
                zero_point,
                dtype=b_dtype,
                packed=True,
            )
            cls.set_param_data(layer, meta.zero_point_name, zero_point)

        if bias is not None and bias.size(0):
            bias = prepare_humming_bias(bias)
            cls.set_param_data(layer, meta.bias_name, bias)

    @classmethod
    def prepare_default_kernel_configs(
        cls,
        layer: torch.nn.Module,
        sublayer_name: str = "",
        **kwargs,
    ):
        meta = layer.humming_metas[sublayer_name]
        warp_shape_nk = (64, 32)
        block_shape_nk1 = (128, 128)
        block_shape_nk2 = (256, 64)
        if meta.a_dtype.num_bits == 8:
            warp_shape_nk = (32, 64)
        elif meta.a_dtype.num_bits == 4:
            warp_shape_nk = (32, 128)
            block_shape_nk1 = (128, 256)
            block_shape_nk2 = (256, 128)

        kernel_configs = [
            [0, 16, (16, *block_shape_nk1), (16, *warp_shape_nk)],
            [16, 32, (32, *block_shape_nk2), (32, *warp_shape_nk)],
            [32, 48, (48, *block_shape_nk2), (48, *warp_shape_nk)],
            [48, 64, (64, *block_shape_nk2), (64, *warp_shape_nk)],
            [64, 96, (48, *block_shape_nk2), (48, *warp_shape_nk)],
            [96, None, (64, *block_shape_nk2), (64, *warp_shape_nk)],
        ]

        if "num_stages" not in kwargs:
            kwargs["num_stages"] = 4

        for min_shape_m, max_shape_m, block_shape, warp_shape in kernel_configs:
            if max_shape_m is None:
                max_shape_m = 1 << 31

            if meta.num_experts is not None:
                min_shape_m = int(0.9 * min_shape_m * meta.num_experts / 4)
                import math
                max_shape_m = math.ceil(0.9 * max_shape_m * meta.num_experts / 4)

            cls.add_kernel_config(
                layer=layer,
                min_shape_m=min_shape_m,
                max_shape_m=max_shape_m,
                block_shape=block_shape,
                warp_shape=warp_shape,
                sublayer_name=sublayer_name,
                **kwargs,
            )

    @classmethod
    def add_kernel_config(
        cls,
        layer: torch.nn.Module,
        min_shape_m: int,
        max_shape_m: int,
        block_shape: tuple[int],
        warp_shape: tuple[int],
        sublayer_name: str = "",
        **kwargs,
    ):
        meta: HummingLayerMeta = layer.humming_metas[sublayer_name]
        kernel = HummingKernel(
            problem_shape=(0, meta.shape_n, meta.shape_k),
            block_shape=block_shape,
            warp_shape=warp_shape,
            pad_shape=(0, meta.pad_shape_n, meta.pad_shape_k),
            a_dtype=meta.a_dtype,
            b_dtype=meta.b_dtype,
            c_dtype=meta.c_dtype,
            bs_dtype=meta.bs_dtype,
            has_input_scale=meta.has_input_scale,
            has_weight_scale=meta.has_weight_scale,
            input_scale_group_size=meta.input_scale_group_size,
            weight_scale_group_size=meta.weight_scale_group_size,
            has_dynamic_zero_point=meta.has_dynamic_zp,
            has_bias=meta.has_bias,
            has_global_scale=meta.has_global_scale,
            mma_type=meta.mma_type,
            **kwargs,
        )
        if not hasattr(layer, "humming_kernel_config_modules"):
            layer.humming_kernel_config_modules = {}
        if not hasattr(layer, "humming_block_size_configs"):
            layer.humming_block_size_configs = {}
        old_kernel_configs = []
        if sublayer_name in layer.humming_kernel_config_modules:
            old_kernel_configs = layer.humming_kernel_config_modules[sublayer_name]()
        old_kernel_configs += [min_shape_m, max_shape_m, kernel.kernel_id]

        block_size_configs = []
        if sublayer_name in layer.humming_block_size_configs:
            block_size_configs = layer.humming_block_size_configs[sublayer_name]
        block_size_configs += [min_shape_m, max_shape_m, block_shape[0]]
        module = make_humming_module("get_kernel_configs", old_kernel_configs)
        layer.humming_kernel_config_modules[sublayer_name] = module.get_kernel_configs
        layer.humming_block_size_configs[sublayer_name] = block_size_configs

    @classmethod
    def may_quant_input(
        cls,
        meta: HummingLayerMeta,
        inputs: torch.Tensor,
        input_scale: Optional[torch.Tensor] = None,
    ):
        assert meta.a_dtype.num_bits
        if meta.a_dtype.num_bits == 16:
            return inputs, None
        if input_scale is not None:
            return inputs, input_scale
        return torch.ops.humming.humming_quant_input(
            inputs=inputs,
            dtype=str(meta.a_dtype),
            group_size=None,
        )

    @classmethod
    def forward_layer(
        cls,
        layer: torch.nn.Module,
        inputs: torch.Tensor,
        outputs: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        sorted_token_ids: Optional[torch.Tensor] = None,
        expert_ids: Optional[torch.Tensor] = None,
        num_tokens_post_padded: Optional[torch.Tensor] = None,
        sublayer_name: str = "",
    ):
        meta = layer.humming_metas[sublayer_name]
        inputs, input_scale = cls.may_quant_input(meta, inputs, input_scale)
        configs = layer.humming_kernel_config_modules[sublayer_name]()
        return torch.ops.humming.launch_humming(
            configs,
            inputs,
            getattr(layer, meta.weight_name),
            outputs,
            input_scale,
            getattr(layer, meta.weight_scale_name, None),
            getattr(layer, meta.zero_point_name, None),
            getattr(layer, meta.bias_name, None),
            getattr(layer, meta.global_scale_name, None),
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            layer.locks,
        )


class HummingLayer(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.meta = HummingLayerMeta(**kwargs)
        HummingMethod.create_weights(self, self.meta)

    def load_weight(
        self,
        weight: Optional[torch.Tensor] = None,
        weight_scale: Optional[torch.Tensor] = None,
        zero_point: Optional[torch.Tensor] = None,
        global_scale: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        expert_id: Optional[int] = None,
    ):
        HummingMethod.load_weight(
            layer=self,
            weight=weight,
            weight_scale=weight_scale,
            zero_point=zero_point,
            global_scale=global_scale,
            bias=bias,
            expert_id=expert_id,
        )

    def finish_load(self):
        HummingMethod.finish_load(self)
        HummingMethod.prepare_default_kernel_configs(self)

    def forward(self, **kwargs):
        return HummingMethod.forward_layer(self, **kwargs)
