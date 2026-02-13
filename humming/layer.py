import torch
from humming import dtypes
from typing import Optional
from humming.kernel.humming import HummingKernel
from humming.kernel.pack_weight import PackWeightKernel
from humming.utils.weight import (
    quantize_weight,
    prepare_humming_weight,
    prepare_humming_weight_scale,
    prepare_humming_zero_point,
    prepare_humming_bias,
)
import dataclasses


@dataclasses.dataclass
class HummingLayerMeta(object):
    a_dtype: dtypes.DataType
    b_dtype: dtypes.DataType
    c_dtype: dtypes.DataType
    bs_dtype: dtypes.DataType
    shape_n: int
    shape_k: int
    num_experts: Optional[int] = None
    has_input_scale: Optional[bool] = False
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

        data = data.to(param.device).view(-1)
        assert data.dtype == param.dtype

        part_tensor = param.data if expert_id is None else param.data[expert_id]
        part_tensor = part_tensor.view(-1)[offset or 0:]
        part_tensor[:data.size(0)] = data

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
            bias = torch.empty(bias_shape, dtype=torch.float32)
            cls.set_param(layer, meta.bias_name, bias)

        layer.locks = torch.nn.Buffer(torch.zeros(1024, dtype=torch.int32))
        if not hasattr(layer, "humming_metas"):
            layer.humming_metas = {}

        layer.humming_metas[meta.sublayer_name] = meta

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
    ):
        meta = layer.humming_metas[sublayer_name]

        if weight is not None and weight.dtype in [torch.float16, torch.bfloat16, torch.float32]:
            assert weight_scale is None
            assert zero_point is None

            weight_shape = (meta.shape_n, meta.shape_k)
            if meta.num_layers is not None and expert_id is None:
                weight_shape = (meta.num_layers,) + weight_shape

            assert weight.shape == weight_shape
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

            expected_shape_n = meta.shape_n if offset_n is None else weight.size(-2)
            shape = (expected_shape_n, meta.shape_k)
            if meta.num_experts is not None and expert_id is None:
                shape = (meta.num_experts,) + shape
            packed_shape = shape[:-1] + (meta.shape_k * meta.b_dtype.num_bits // 32,)
            assert weight.shape == (packed_shape if packed else shape)

            offset = (offset_n or 0) * meta.shape_k * meta.b_dtype.num_bits // 32
            print(offset_n, offset, meta.shape_k, meta.shape_n)

            if not packed:
                kernel = PackWeightKernel(meta.b_dtype.num_bits)
                weight = kernel(weight)

            cls.set_param_data(layer, meta.weight_name, weight, offset, expert_id)

        if weight_scale is not None:
            weight_scale_param = getattr(layer, meta.weight_scale_name)
            num_groups = weight_scale_param.size(-2)
            expected_shape_n = meta.shape_n if offset_n is None else weight_scale.size(-2)
            shape = (expected_shape_n, num_groups)
            if meta.num_experts is not None and expert_id is None:
                shape = (meta.num_experts,) + shape
            assert weight_scale.shape == shape
            weight_scale = weight_scale.to(device=weight_scale_param.device)
            if weight_scale.element_size == 1 or weight_scale_param.element_size == 1:
                assert weight_scale.dtype == weight_scale_param.dtype
            else:
                assert weight_scale.dtype in [torch.float16, torch.bfloat16, torch.float32]
                assert weight_scale_param.dtype in [torch.float16, torch.bfloat16, torch.float32]
                weight_scale = weight_scale.to(dtype=weight_scale_param.dtype)

            offset = (offset_n or 0) * num_groups
            print(offset_n, offset)

            cls.set_param_data(layer, meta.weight_scale_name, weight_scale, offset, expert_id)

        if zero_point is not None:
            zero_point_param = getattr(layer, meta.zero_point_name)
            zero_point = zero_point.to(device=zero_point_param.device)

            num_groups = zero_point_param.size(-2)
            shape = (meta.shape_n, num_groups)
            packed_shape = (meta.shape_n * meta.b_dtype.num_bits // 32, num_groups)
            if meta.num_experts is not None and expert_id is None:
                shape = (meta.num_experts,) + shape
                packed_shape = (meta.num_experts,) + packed_shape

            if zero_point.shape == shape:
                zero_point = zero_point.transpose(-1, -2).contiguous()
                zero_point = zero_point.squeeze().view(*zero_point.shape)
                kernel = PackWeightKernel(meta.b_dtype.num_bits)
                zero_point = kernel(zero_point)
                zero_point = zero_point.transpose(-1, -2).contiguous()
                zero_point = zero_point.squeeze().view(*zero_point.shape)

            assert zero_point.shape == packed_shape
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
            expected_shape_n = meta.shape_n if offset_n is None else bias.size(-1)
            shape = (expected_shape_n, weight_scale_param.size(-1))
            if meta.num_experts is not None and expert_id is None:
                shape = (meta.num_experts,) + shape
            assert weight_scale.shape == shape
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
        if zero_point is not None:
            num_groups = zero_point.size(-2)
            padded_bits = 4 if meta.b_dtype.num_bits <= 4 else 8
            zero_point = zero_point.view(padded_bits, -1)[: meta.b_dtype.num_bits]
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

        if weight_scale is not None:
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

        if zero_point is not None:
            b_dtype = meta.b_dtype
            zero_point = prepare_humming_zero_point(
                zero_point,
                dtype=b_dtype,
                packed=True,
            )
            cls.set_param_data(layer, meta.zero_point_name, zero_point)

        if bias is not None:
            bias = prepare_humming_bias(bias)
            cls.set_param_data(layer, meta.bias_name, bias)

    @classmethod
    def forward_layer(
        self,
        layer: torch.nn.Module,
        inputs: torch.Tensor,
        outputs: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        sorted_token_ids: Optional[torch.Tensor] = None,
        expert_ids: Optional[torch.Tensor] = None,
        num_tokens_past_padded: Optional[torch.Tensor] = None,
        sublayer_name: str = "",
        **kwargs,
    ):
        meta = layer.humming_metas[sublayer_name]

        humming_kernel = HummingKernel(
            problem_shape=(0, meta.shape_n, meta.shape_k),
            a_dtype=meta.a_dtype,
            b_dtype=meta.b_dtype,
            c_dtype=meta.c_dtype,
            bs_dtype=meta.bs_dtype,
            has_input_scale=meta.has_input_scale,
            has_weight_scale=meta.has_weight_scale,
            input_scale_group_size=meta.input_scale_group_size,
            weight_scale_group_size=meta.weight_scale_group_size,
            has_bias=meta.has_bias,
            has_global_scale=meta.has_global_scale,
            has_dynamic_zero_point=meta.has_dynamic_zp,
            is_moe=meta.num_experts is not None,
            device_index=inputs.device.index,
            mma_type=meta.mma_type,
            **kwargs,
        )

        return humming_kernel(
            inputs=inputs,
            weight=getattr(layer, meta.weight_name),
            outputs=outputs,
            input_scale=input_scale,
            weight_scale=getattr(layer, meta.weight_scale_name, None),
            zero_point=getattr(layer, meta.zero_point_name, None),
            bias=getattr(layer, meta.bias_name, None),
            global_scale=getattr(layer, meta.global_scale_name, None),
            locks=layer.locks,
            topk_weights=topk_weights,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_past_padded=num_tokens_past_padded,
            num_sms=kwargs.get("num_sms", None),
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

    def forward(self, **kwargs):
        HummingMethod.forward_layer(self, **kwargs)
