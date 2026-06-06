import dataclasses
import json
import math
import os
import re
from typing import Any, Callable

import torch

from humming import dtypes, ops
from humming.config import GemmType, LayerConfig, MmaType, WeightScaleType
from humming.schema import (
    BaseInputSchema,
    BaseWeightSchema,
    HummingInputSchema,
    HummingWeightSchema,
)
from humming.tune import get_heuristics_config
from humming.utils.device import estimate_compute_bound_threshold
from humming.utils.weight import (
    prepare_humming_bias,
    prepare_humming_weight,
    prepare_humming_weight_scale,
    prepare_humming_zero_point,
)


def get_default_f16_torch_dtype() -> torch.dtype:
    torch_dtype = torch.get_default_dtype()
    if torch_dtype not in [torch.float16, torch.bfloat16]:
        if torch.cuda.get_device_capability()[0] >= 8:
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float16
    return torch_dtype


@dataclasses.dataclass(kw_only=True, unsafe_hash=True)
class HummingLayerMeta(LayerConfig):
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

    @property
    def param_dtype(self):
        if self.c_dtype == dtypes.float16:
            return torch.float16
        elif self.c_dtype == dtypes.bfloat16:
            return torch.bfloat16
        else:
            raise ValueError(f"unsupported c_dtype: {self.c_dtype}")

    @property
    def should_apply_bs_on_c(self):
        if self.use_fused_e8m0_scale:
            return False
        elif self.mma_type == MmaType.MMA:
            return self.weight_scale_group_size == 0 or self.a_dtype.num_bits != 16
        elif self.mma_type == MmaType.WGMMA:
            return self.weight_scale_group_size == 0
        else:
            raise ValueError(f"unsupported mma_type: {self.mma_type}")

    @property
    def weight_nbytes(self):
        nbytes1 = self.shape_n * self.shape_k * self.b_dtype.num_bits // 8
        num_groups = self.shape_k / (self.weight_scale_group_size or self.shape_k)
        assert self.bs_dtype is not None
        nbytes2 = self.shape_n * num_groups * self.bs_dtype.num_bits // 8
        nbytes3 = self.shape_n * num_groups * (math.ceil(self.b_dtype.num_bits / 4) * 4) // 8
        nbytes = nbytes1 + nbytes2
        if self.has_zero_point and self.is_fp_zero_point:
            nbytes = nbytes + nbytes2
        elif self.has_zero_point:
            nbytes = nbytes + nbytes3
        return nbytes * (self.num_experts or 1)

    def estimate_bound_min_shape_m(self, use_f16_accum: bool = False):
        return estimate_compute_bound_threshold(
            weight_nbytes=self.weight_nbytes // (self.num_experts or 1),
            shape_n=self.shape_n,
            shape_k=self.shape_k,
            dtype=str(self.a_dtype),
            use_f16_accum=use_f16_accum,
        )

    def to_str(self) -> str:
        if hasattr(self, "_meta_str"):
            return self._meta_str
        return super().to_str()

    def __setattr__(self, name, value):
        if hasattr(self, "_meta_str"):
            raise AttributeError(f"Instance is frozen, cannot set {name}")
        super().__setattr__(name, value)

    def __post_init__(self):
        super().__post_init__()

        if isinstance(self.b_dtype, dtypes.IntegerType):
            if isinstance(self.a_dtype, dtypes.FloatingPointType):
                self.b_dtype = dataclasses.replace(self.b_dtype, is_signed=False)
            elif self.a_dtype.num_bits == self.b_dtype.num_bits:
                self.b_dtype = dataclasses.replace(self.b_dtype, is_signed=True)
            else:
                self.b_dtype = dataclasses.replace(self.b_dtype, is_signed=False)

        if not self.use_fused_e8m0_scale:
            self.use_fused_e8m0_scale = (
                self.a_dtype in [dtypes.float8e4m3, dtypes.int8]
                and self.weight_scale_group_size > 0
                and self.b_dtype in [dtypes.float4e2m1]
                and self.bs_dtype in [dtypes.float8e8m0]
            )

        if not self.use_int_weight_scale and not self.use_fused_e8m0_scale:
            self.use_int_weight_scale = (
                self.a_dtype in [dtypes.int8, dtypes.int4]
                and self.input_scale_group_size == 0
                and self.weight_scale_group_size > 0
            )

        if self.use_int_weight_scale:
            self.weight_scale_type = WeightScaleType.GROUP_TENSOR
            self.is_group_weight_scale = True
            self.is_tensor_weight_scale = True
            self.bs_dtype = self.c_dtype

        if self.use_fused_e8m0_scale:
            self.weight_scale_type = WeightScaleType.GROUP_TENSOR
            self.is_group_weight_scale = True
            self.is_tensor_weight_scale = True

        self._meta_str = self.to_str()


class HummingModule(torch.nn.Module):
    humming_block_size_configs: dict[str, list[int]]
    humming_kernel_config_modules: dict[str, Callable]
    humming_metas: dict[str, HummingLayerMeta]
    locks: torch.Tensor | None


class HummingLayerMethod:
    completed_layer_configs: set[tuple[HummingLayerMeta, tuple[str, ...]]] = set()

    @classmethod
    def may_set_param(cls, layer: torch.nn.Module, name: str, tensor: torch.Tensor | None):
        if tensor is None:
            return
        param = torch.nn.Parameter(tensor, requires_grad=False)
        setattr(layer, name, param)

    @classmethod
    def prepare_layer_meta(
        cls,
        layer: HummingModule | torch.nn.Module,
        shape_n: int,
        shape_k: int,
        weight_schema: HummingWeightSchema,
        input_schema: HummingInputSchema | None = None,
        num_experts: int | None = None,
        pad_n_to_multiple: int = 1,
        pad_k_to_multiple: int = 1,
        has_bias: bool = False,
        torch_dtype: torch.dtype | None = None,
        sublayer_name: str = "",
    ):
        if torch_dtype is None:
            torch_dtype = get_default_f16_torch_dtype()
        f16_dtype = dtypes.DataType.from_torch_dtype(torch_dtype)
        pad_shape_n = math.ceil(shape_n / pad_n_to_multiple) * pad_n_to_multiple - shape_n
        pad_shape_k = math.ceil(shape_k / pad_k_to_multiple) * pad_k_to_multiple - shape_k

        if input_schema is None:
            input_schema = HummingInputSchema(a_dtype=f16_dtype)

        assert isinstance(input_schema, HummingInputSchema)
        assert isinstance(weight_schema, HummingWeightSchema)

        meta = HummingLayerMeta(
            a_dtype=input_schema.a_dtype or f16_dtype,
            b_dtype=weight_schema.b_dtype,
            bs_dtype=weight_schema.bs_dtype or f16_dtype,
            c_dtype=f16_dtype,
            shape_n=shape_n + pad_shape_n,
            shape_k=shape_k + pad_shape_k,
            pad_shape_n=pad_shape_n,
            pad_shape_k=pad_shape_k,
            num_experts=num_experts or 0,
            has_bias=has_bias,
            input_scale_group_size=input_schema.input_scale_group_size,
            weight_scale_group_size=weight_schema.weight_scale_group_size,
            weight_scale_group_size_n=weight_schema.weight_scale_group_size_n,
            weight_scale_type=weight_schema.weight_scale_type,
            has_zero_point=weight_schema.has_zero_point,
            is_fp_zero_point=weight_schema.is_fp_zero_point,
            sublayer_name=sublayer_name,
        )

        if not hasattr(layer, "humming_metas"):
            layer.humming_metas = {}
        assert isinstance(layer.humming_metas, dict)
        layer.humming_metas[sublayer_name] = meta

        return meta

    @classmethod
    def check_and_pad_tensors(cls, tensors: dict[str, torch.Tensor], meta: HummingLayerMeta):
        tensors = tensors.copy()
        schema = HummingWeightSchema(
            b_dtype=meta.b_dtype,
            bs_dtype=meta.bs_dtype,
            weight_scale_group_size=meta.weight_scale_group_size,
            weight_scale_group_size_n=meta.weight_scale_group_size_n,
            weight_scale_type=meta.weight_scale_type,
            has_zero_point=meta.has_zero_point,
            is_fp_zero_point=meta.is_fp_zero_point,
        )

        if meta.use_int_weight_scale:
            dtype = dtypes.torch_dtype_map[meta.bs_dtype]
            tensors["weight_scale"] = tensors["weight_scale"].to(dtype)
            if "global_scale" not in tensors:
                tensors["global_scale"] = torch.ones(
                    (meta.num_experts or 1),
                    device=tensors["weight_scale"].device,
                    dtype=torch.float32,
                )

        if meta.use_fused_e8m0_scale:
            if "global_scale" not in tensors:
                tensors["global_scale"] = torch.ones(
                    (meta.num_experts or 1),
                    device=tensors["weight_scale"].device,
                    dtype=torch.float32,
                )

        schema.validate_tensors(
            tensors,
            shape_n=meta.shape_n - meta.pad_shape_n,
            shape_k=meta.shape_k - meta.pad_shape_k,
            num_experts=meta.num_experts,
            param_dtype=meta.param_dtype,
            has_bias=meta.has_bias,
        )

        tensors_attrs = schema.get_tensors_attrs(
            shape_n=meta.shape_n,
            shape_k=meta.shape_k,
            num_experts=meta.num_experts,
            param_dtype=meta.param_dtype,
            has_bias=meta.has_bias,
        )

        for key, attrs in tensors_attrs.items():
            shape = attrs["shape"]
            tensor = tensors[key]
            padding: list[int] = []
            value = 0 if tensor.dtype != torch.float8_e8m0fnu else 1
            for i in range(1, len(shape) + 1):
                padding += (0, shape[-i] - tensor.shape[-i])

            tensors[key] = torch.nn.functional.pad(tensor, pad=padding, value=value)

        return tensors

    @classmethod
    def may_process_int_weight_scale(
        cls,
        meta: HummingLayerMeta,
        weight_scale: torch.Tensor,
        global_scale: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if meta.bs_dtype is not None and meta.bs_dtype.num_bits == 8:
            assert weight_scale is not None
            torch_dtype = dtypes.torch_dtype_map[meta.c_dtype]
            weight_scale = weight_scale.to(torch_dtype)

        assert weight_scale is not None
        dtype = weight_scale.dtype
        assert dtype in [torch.float16, torch.bfloat16]
        scale_factor = weight_scale.float().abs().max() / 1024
        weight_scale = (weight_scale.float() / scale_factor).round().to(torch.int16)
        weight_scale = weight_scale.view(dtype)

        if global_scale is not None:
            assert global_scale is not None
            out_global_scale = global_scale * scale_factor
        else:
            out_global_scale = torch.full(
                (meta.num_experts or 1,),
                fill_value=scale_factor.item(),
                device=weight_scale.device,
            )

        return weight_scale, out_global_scale

    @classmethod
    def may_process_fused_e8m0_scale(
        cls,
        meta: HummingLayerMeta,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        global_scale: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        origin_dtype = weight_scale.dtype
        origin_shape = weight_scale.shape
        assert origin_dtype in [torch.uint8, torch.float8_e8m0fnu]
        weight_scale = weight_scale.view(torch.uint8).view(meta.num_experts or 1, -1)

        scale_max = weight_scale.max(1)[0].unsqueeze(-1)
        scale_min = weight_scale.min(1)[0].unsqueeze(-1)
        scale_range = scale_max - scale_min
        if meta.a_dtype == dtypes.int8:
            max_range_val = 3
        elif meta.a_dtype == dtypes.float8e4m3:
            max_range_val = 11
        else:
            raise ValueError(f"unsupported a_dtype: {meta.a_dtype}")

        max_range = torch.tensor(max_range_val, dtype=torch.uint8, device=scale_range.device)
        scale_range = scale_range.minimum(max_range)
        scale_min_new = scale_max - scale_range
        delta_scale_offsets = weight_scale.maximum(scale_min_new) - weight_scale
        weight_scale = weight_scale.maximum(scale_min_new) - scale_min_new
        if meta.a_dtype == dtypes.float8e4m3:
            weight_scale = weight_scale + 1
        weight_scale = weight_scale.view(origin_dtype).view(origin_shape)
        ops.process_mxfp4_w4a8_weight(weight, delta_scale_offsets, inplace=True)

        scale_factor = 2 ** (scale_min_new.view(-1).float() - 127)
        scale_factor = scale_factor / 2
        if global_scale is not None:
            assert global_scale is not None
            out_global_scale = global_scale * scale_factor
        else:
            out_global_scale = scale_factor

        return weight, weight_scale, out_global_scale

    @classmethod
    def get_default_tuning_configs(
        cls,
        layer: HummingModule | torch.nn.Module,
        use_f16_accum: bool = False,
        use_batch_invariant: bool = False,
        gemm_type: GemmType | str = GemmType.DENSE,
        sublayer_name: str = "",
    ) -> list[Any]:
        assert isinstance(layer.humming_metas, dict)
        meta = layer.humming_metas[sublayer_name]
        return get_heuristics_config(
            meta=meta,
            use_f16_accum=use_f16_accum,
            gemm_type=gemm_type,
            use_batch_invariant=use_batch_invariant,
        )

    @classmethod
    def transform_humming_layer(
        cls,
        layer: HummingModule | torch.nn.Module,
        sublayer_name: str = "",
        already_padded: bool = False,
    ):
        assert isinstance(layer.humming_metas, dict)
        meta = layer.humming_metas[sublayer_name]
        prefix = meta.name_prefix
        tensors = dict(
            (key.removeprefix(prefix), value)
            for key, value in layer.state_dict().items()
            if key.startswith(prefix)
        )

        if not already_padded:
            tensors = cls.check_and_pad_tensors(tensors, meta)

        weight = tensors["weight"]
        zero_point = tensors["zero_point"] if meta.has_zero_point else None
        weight_scale: torch.Tensor | None = None
        if meta.weight_scale_type != WeightScaleType.TENSOR:
            weight_scale = tensors["weight_scale"]
        bias = tensors["bias"] if meta.has_bias else None
        if "TENSOR" in str(meta.weight_scale_type):
            global_scale = tensors.get("global_scale", None)
        else:
            global_scale = None

        if meta.use_fused_e8m0_scale:
            assert weight_scale is not None
            weight, weight_scale, global_scale = cls.may_process_fused_e8m0_scale(
                meta,
                weight=weight,
                weight_scale=weight_scale,
                global_scale=global_scale,
            )

        interleave_mode = 3
        if meta.use_fused_e8m0_scale and meta.a_dtype == dtypes.float8e4m3:
            interleave_mode = 2

        weight = prepare_humming_weight(
            weight=weight,
            b_dtype=meta.b_dtype,
            a_dtype=meta.a_dtype,
            zero_point=zero_point,
            use_wgmma=meta.mma_type == MmaType.WGMMA,
            use_fused_e8m0_scale=meta.use_fused_e8m0_scale,
            packed=True,
            interleave_mode=interleave_mode,
        )

        if weight_scale is not None:
            weight_scale = prepare_humming_weight_scale(
                weight_scale,
                to_apply_on_c=meta.should_apply_bs_on_c,
                is_blockwise=meta.weight_scale_type == WeightScaleType.BLOCK,
            )

        if zero_point is not None:
            zero_point = prepare_humming_zero_point(zero_point, meta.b_dtype, packed=True)

        if bias is not None:
            bias = prepare_humming_bias(bias)

        if meta.use_int_weight_scale:
            assert weight_scale is not None
            weight_scale, global_scale = cls.may_process_int_weight_scale(
                meta,
                weight_scale=weight_scale,
                global_scale=global_scale,
            )

        cls.may_set_param(layer, meta.weight_name, weight)
        cls.may_set_param(layer, meta.weight_scale_name, weight_scale)
        cls.may_set_param(layer, meta.zero_point_name, zero_point)
        cls.may_set_param(layer, meta.global_scale_name, global_scale)
        cls.may_set_param(layer, meta.bias_name, bias)

    @classmethod
    def may_quant_input(
        cls,
        layer: HummingModule | torch.nn.Module,
        inputs: torch.Tensor,
        input_scale: torch.Tensor | None = None,
        quanted_input: torch.Tensor | None = None,
        sublayer_name: str = "",
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert isinstance(layer.humming_metas, dict)
        meta = layer.humming_metas[sublayer_name]
        if meta.a_dtype.num_bits == 16:
            return inputs, None
        if input_scale is not None:
            return inputs, input_scale
        quanted_input, input_scale = ops.quant_input(
            inputs=inputs,
            outputs=quanted_input,
            dtype=str(meta.a_dtype),
            group_size=None,
        )
        return quanted_input, (input_scale if input_scale.numel() else None)

    @classmethod
    def forward_layer(
        cls,
        layer: HummingModule | torch.nn.Module,
        inputs: torch.Tensor,
        outputs: torch.Tensor | None = None,
        input_scale: torch.Tensor | None = None,
        sorted_ids: torch.Tensor | None = None,
        expert_ids: torch.Tensor | None = None,
        num_tokens_padded: torch.Tensor | None = None,
        expert_layout: torch.Tensor | None = None,
        top_k: int = 1,
        valid_shape_m: int = 0,
        compute_config: dict | str | None = None,
        tuning_config: dict | list | str | None = None,
        sublayer_name: str = "",
    ):
        assert isinstance(layer.humming_metas, dict)
        meta = layer.humming_metas[sublayer_name]
        inputs, input_scale = cls.may_quant_input(
            layer=layer,
            inputs=inputs,
            input_scale=input_scale,
            sublayer_name=sublayer_name,
        )

        if isinstance(compute_config, dict):
            compute_config = json.dumps(compute_config)

        if isinstance(tuning_config, (list, dict)):
            tuning_config = json.dumps(tuning_config)

        return ops.humming_gemm(
            layer_config=meta.to_str(),
            compute_config=compute_config,
            tuning_config=tuning_config,
            inputs=inputs,
            weight=getattr(layer, meta.weight_name),
            outputs=outputs,
            input_scale=input_scale,
            weight_scale=getattr(layer, meta.weight_scale_name, None),
            zero_point=getattr(layer, meta.zero_point_name, None),
            bias=getattr(layer, meta.bias_name, None),
            global_scale=getattr(layer, meta.global_scale_name, None),
            sorted_ids=sorted_ids,
            expert_ids=expert_ids,
            num_tokens_padded=num_tokens_padded,
            expert_layout=expert_layout,
            locks=layer.locks,
            top_k=top_k,
            valid_shape_m=valid_shape_m,
        )


class HummingMethod(HummingLayerMethod):
    pass


@dataclasses.dataclass(repr=False, eq=False)
class HummingLayer(HummingModule):
    shape_n: int
    shape_k: int
    weight_config: BaseWeightSchema | dict[str, Any]
    input_config: BaseInputSchema | dict[str, Any] | None = None
    pad_n_to_multiple: int = 1
    pad_k_to_multiple: int = 1
    num_experts: int | None = None
    has_bias: bool = False
    torch_dtype: torch.dtype | None = None

    def __post_init__(self) -> None:
        super().__init__()

        if self.torch_dtype is None:
            self.torch_dtype = get_default_f16_torch_dtype()
        assert self.torch_dtype in [torch.float16, torch.bfloat16], self.torch_dtype

        self.input_config = self.input_config or {}

        if isinstance(self.input_config, dict):
            if "quant_method" not in self.input_config:
                self.input_config["quant_method"] = "humming"
            if "dtype" not in self.input_config:
                self.input_config["dtype"] = dtypes.DataType.from_torch_dtype(self.torch_dtype)
        if isinstance(self.weight_config, dict) and "quant_method" not in self.weight_config:
            self.weight_config["quant_method"] = "humming"

        self.input_schema: BaseInputSchema = (
            self.input_config
            if isinstance(self.input_config, BaseInputSchema)
            else BaseInputSchema.from_config(self.input_config)
        )

        self.weight_schema: BaseWeightSchema = (
            self.weight_config
            if isinstance(self.weight_config, BaseWeightSchema)
            else BaseWeightSchema.from_config(self.weight_config)
        )

        tensors_attrs = self.weight_schema.get_tensors_attrs(
            shape_n=self.shape_n,
            shape_k=self.shape_k,
            param_dtype=self.torch_dtype,
            num_experts=self.num_experts,
            has_bias=self.has_bias,
        )

        for name, attrs in tensors_attrs.items():
            tensor = torch.empty(attrs["shape"], dtype=attrs["dtype"])
            param = torch.nn.Parameter(tensor, requires_grad=False)
            for key, value in attrs.items():
                if key not in ["shape", "dtype"]:
                    setattr(param, key, value)
            setattr(self, name, param)

        locks = torch.zeros((1024), dtype=torch.int32, device=torch.cuda.current_device())
        self.register_buffer("locks", locks)

    @staticmethod
    def filter_tensors(
        tensors: dict[str, torch.Tensor], prefix: str = ""
    ) -> dict[str, torch.Tensor]:
        tensors_new = {}
        for key in tensors:
            if key.startswith(prefix):
                key_new = key.removeprefix(prefix).lstrip(".")
                tensors_new[key_new] = tensors[key]
        return tensors_new

    def load_from_unquantized(self, tensor: torch.Tensor):
        assert isinstance(self.weight_schema, HummingWeightSchema)
        assert tensor.dtype in [torch.float16, torch.bfloat16, torch.float32]
        expected_shape: tuple[int, ...] = (self.shape_n, self.shape_k)
        if self.num_experts is not None and self.num_experts != 0:
            expected_shape = (self.num_experts,) + expected_shape
        assert tensor.shape == expected_shape

        from humming.utils.weight import quantize_weight

        f16_dtype = dtypes.DataType.from_torch_dtype(self.torch_dtype)
        weight, weight_scale, zero_point, global_scale = quantize_weight(
            weight=tensor,
            dtype=self.weight_schema.b_dtype,
            scale_dtype=self.weight_schema.bs_dtype or f16_dtype,
            group_size=self.weight_schema.weight_scale_group_size,
            has_zero_point=self.weight_schema.has_zero_point,
            has_global_scale="TENSOR" in str(self.weight_schema.weight_scale_type),
            is_fp_zero_point=self.weight_schema.is_fp_zero_point,
            pack=True,
        )

        tensors = {"weight": weight}
        if weight_scale is not None:
            tensors["weight_scale"] = weight_scale
        if zero_point is not None:
            tensors["zero_point"] = zero_point
        if global_scale is not None:
            tensors["global_scale"] = global_scale

        self.load_from_tensors(tensors)

    def load_from_tensors(self, tensors: dict[str, torch.Tensor], prefix: str = ""):
        tensors = self.filter_tensors(tensors, prefix)
        self.load_state_dict(tensors, strict=False)

    def load_from_safetensors(self, name: str, prefix: str = ""):
        assert os.path.exists(name)
        import safetensors.torch

        if os.path.isfile(name):
            tensors = safetensors.torch.load_file(name)
            return self.load_from_tensors(tensors, prefix)

        filename = os.path.join(name, "model.safetensors")
        index_filename = os.path.join(name, "model.safetensors.index.json")
        if os.path.exists(filename):
            return self.load_from_safetensors(filename, prefix)

        assert os.path.exists(index_filename)
        with open(index_filename, "r") as f:
            index_data = json.load(f)
        loaded_filenames = set()
        for key, filename in index_data["weight_map"].items():
            filename = os.path.join(name, filename)
            if filename in loaded_filenames:
                continue
            if key.startswith(prefix):
                self.load_from_safetensors(filename, prefix)
                loaded_filenames.add(filename)

    @classmethod
    def from_safetensors(
        cls,
        name: str,
        prefix: str = "",
        pad_n_to_multiple: int = 1,
        pad_k_to_multiple: int = 1,
        torch_dtype: torch.dtype | None = None,
    ):
        assert os.path.isdir(name)
        import safetensors.torch

        config_filename = os.path.join(name, "config.json")
        with open(config_filename, "r") as f:
            config = json.load(f)
            if torch_dtype is None and config.get("torch_dtype", "") == "float16":
                torch_dtype = torch.float16

            assert "quantization_config" in config, "not a quantization model"
            config = config["quantization_config"]

        keys = ["ignored_layers", "ignore", "modules_to_not_convert"]
        for key in keys:
            ignore_layers = config.get(key, []) or []
            assert not any(x in prefix for x in ignore_layers), f"layer {prefix} is unquantized"

        layer_config = config.copy()
        for regex in config.get("dynamic", {}):
            if regex[:1] != "-":
                assert not re.match(regex[2:], prefix), f"layer {prefix} is unquantized"
            elif re.match(regex[2:], prefix):
                layer_config.update(config["dynamic"][regex])
                break

        if config["quant_method"] in ["compressed-tensors", "modelopt"]:
            target_group_config = None
            for group_config in config["config_groups"].values():
                if "Linear" in group_config["targets"]:
                    target_group_config = group_config["weights"].copy()
                    break
            assert target_group_config is not None, f"layer {prefix} is unquantized"
            target_group_config["quant_method"] = config["quant_method"]
            if "format" in config:
                target_group_config["format"] = config["format"]
            if "quant_algo" in config:
                target_group_config["quant_algo"] = config["quant_algo"]
            layer_config = target_group_config

        schema = BaseWeightSchema.from_config(layer_config)

        filename = os.path.join(name, "model.safetensors")
        index_filename = os.path.join(name, "model.safetensors.index.json")
        if os.path.exists(filename):
            tensors = safetensors.torch.load_file(filename)
            tensors = cls.filter_tensors(tensors, prefix)
        else:
            assert os.path.exists(index_filename)
            with open(index_filename, "r") as f:
                index_data = json.load(f)
            loaded_filenames = set()
            tensors = {}
            for key, filename in index_data["weight_map"].items():
                filename = os.path.join(name, filename)
                if filename in loaded_filenames:
                    continue
                if key.startswith(prefix):
                    tensors2 = safetensors.torch.load_file(filename)
                    tensors.update(cls.filter_tensors(tensors2, prefix))
                    loaded_filenames.add(filename)

        shape_n, shape_k, num_experts, has_bias = schema.infer_shape(tensors)

        layer = cls(
            shape_n=shape_n,
            shape_k=shape_k,
            weight_config=schema,
            num_experts=num_experts or 0,
            pad_n_to_multiple=pad_n_to_multiple,
            pad_k_to_multiple=pad_k_to_multiple,
            has_bias=has_bias,
            torch_dtype=torch_dtype,
        )

        layer.load_from_tensors(tensors)
        return layer

    def transform(self):
        if not isinstance(self.weight_schema, HummingWeightSchema):
            assert self.torch_dtype is not None
            self.weight_schema, tensors = self.weight_schema.convert_humming(
                tensors=self.state_dict(),
                shape_n_stacks=[self.shape_n],
                shape_k_stacks=[self.shape_k],
                param_dtype=self.torch_dtype,
            )

            self.input_schema, _ = self.input_schema.convert_humming(
                tensors=self.state_dict(),
                shape_n_stacks=[self.shape_n],
                shape_k_stacks=[self.shape_k],
                param_dtype=self.torch_dtype,
            )

            for name, _ in list(self.named_parameters()):
                delattr(self, name)

            for name, tensor in tensors.items():
                param = torch.nn.Parameter(tensor, requires_grad=False)
                setattr(self, name, param)

        assert isinstance(self.input_schema, HummingInputSchema)
        HummingLayerMethod.prepare_layer_meta(
            layer=self,
            shape_n=self.shape_n,
            shape_k=self.shape_k,
            weight_schema=self.weight_schema,
            input_schema=self.input_schema,
            num_experts=self.num_experts,
            pad_n_to_multiple=self.pad_n_to_multiple,
            pad_k_to_multiple=self.pad_k_to_multiple,
            torch_dtype=self.torch_dtype,
            has_bias=self.has_bias,
        )

        HummingLayerMethod.transform_humming_layer(self)

    def forward(
        self,
        inputs: torch.Tensor,
        outputs: torch.Tensor | None = None,
        input_scale: torch.Tensor | None = None,
        sorted_ids: torch.Tensor | None = None,
        expert_ids: torch.Tensor | None = None,
        num_tokens_padded: torch.Tensor | None = None,
        expert_layout: torch.Tensor | None = None,
        top_k: int = 1,
        valid_shape_m: int = 0,
        compute_config: dict | str | None = None,
        tuning_config: dict | list | str | None = None,
    ) -> torch.Tensor:
        return HummingLayerMethod.forward_layer(
            layer=self,
            inputs=inputs,
            outputs=outputs,
            input_scale=input_scale,
            sorted_ids=sorted_ids,
            expert_ids=expert_ids,
            num_tokens_padded=num_tokens_padded,
            expert_layout=expert_layout,
            top_k=top_k,
            valid_shape_m=valid_shape_m,
            compute_config=compute_config,
            tuning_config=tuning_config,
        )
