import dataclasses
from typing import Any

import torch

from humming import dtypes, ops
from humming.config import InputQuantizationMode
from humming.config.enum import WeightScale2Type, WeightScaleType
from humming.schema.base import BaseInputSchema, BaseWeightSchema
from humming.schema.humming import HummingInputSchema, HummingWeightSchema

_FLOAT_DTYPES = {
    "fp4": dtypes.float4e2m1,
    "fp6_e2m3": dtypes.float6e2m3,
    "fp6_e3m2": dtypes.float6e3m2,
    "fp8_e4m3": dtypes.float8e4m3,
    "fp8_e5m2": dtypes.float8e5m2,
}

_STORAGE_DTYPES = {
    "int2": torch.int32,
    "int3": torch.uint8,
    "int4": torch.int32,
    "uint4": torch.int32,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "fp4": torch.uint8,
    "fp6_e2m3": torch.uint8,
    "fp6_e3m2": torch.uint8,
    "fp8_e4m3": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
}

_SCALE_DTYPES: dict[str | None, torch.dtype] = {
    "e8m0": torch.uint8,
    "e5m3": torch.uint8,
    "fp8_e5m3": torch.uint8,
    "fp8_e4m3": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
}


def parse_quark_tensor_config(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, list):
        if len(value) == 1:
            value = value[0]
        elif len(value) == 2:
            # Preserve checkpoint stage order while keeping schema fields flat.
            scale_stage = 1 if value[1].get("is_scale_quant") else 0
            tensor_stage = 1 - scale_stage
            scale_spec = value[scale_stage]
            value = dict(value[tensor_stage])
            value.update(
                scale_dtype=scale_spec["dtype"],
                scale_qscheme=scale_spec["qscheme"],
                scale_ch_axis=scale_spec.get("ch_axis"),
                scale_is_dynamic=scale_spec.get("is_dynamic", False),
                scale_stage=scale_stage,
            )
            if scale_spec.get("symmetric") is False:
                raise ValueError("Quark secondary scales must use symmetric quantization")
        else:
            raise ValueError("Quark conversion supports at most two quantization stages")
    if not isinstance(value, dict):
        raise ValueError("Quark tensor configuration must be a mapping")
    return dict(value, quant_method="quark")


def _unpack_weight(tensor: torch.Tensor, num_bits: int, reorder: bool = False) -> torch.Tensor:
    storage_bits = tensor.element_size() * 8
    output_shape = (*tensor.shape[:-1], tensor.size(-1) * storage_bits // num_bits)
    num_values = tensor.numel() * storage_bits // num_bits
    packed_bytes = tensor.contiguous().view(torch.uint8).reshape(-1)
    packed_bytes = torch.nn.functional.pad(packed_bytes, (0, -packed_bytes.numel() % 4))
    packed = packed_bytes.view(torch.int32).reshape(1, -1)
    packed = torch.nn.functional.pad(packed, (0, -packed.numel() % num_bits))
    values = ops.unpack_weight(packed, num_bits).flatten()[:num_values].view(output_shape)
    if reorder:
        values = values.reshape(*values.shape[:-1], -1, 8)[..., [0, 4, 1, 5, 2, 6, 3, 7]]
        values = values.flatten(-2)
    return values.contiguous()


def _pack_weight(tensor: torch.Tensor, num_bits: int) -> torch.Tensor:
    output_shape = (*tensor.shape[:-1], tensor.size(-1) * num_bits // 32)
    num_words = tensor.numel() * num_bits // 32
    values = tensor.reshape(1, -1).to(torch.int32)
    values = torch.nn.functional.pad(values, (0, -values.numel() % 32))
    return ops.pack_weight(values, num_bits).flatten()[:num_words].view(output_shape).contiguous()


def _decode_scale(tensor: torch.Tensor, scale_format: str | None) -> torch.Tensor:
    if scale_format == "e8m0":
        return tensor.view(torch.float8_e8m0fnu)
    if scale_format in ("e5m3", "fp8_e5m3"):
        # Quark E5M3 is unsigned, with bias 15 and subnormal exponent -14.
        values = tensor.to(torch.int32)
        exponent, mantissa = values >> 3, values & 7
        significand = torch.where(exponent == 0, mantissa.float() / 8, 1 + mantissa.float() / 8)
        scale = torch.ldexp(significand, exponent.clamp_min(1) - 15)
        return scale.masked_fill(values == 255, float("nan"))
    if scale_format in ("fp8_e4m3", "fp8_e5m2"):
        return tensor.view(_SCALE_DTYPES[scale_format])
    return tensor.float()


@dataclasses.dataclass(kw_only=True)
class QuarkWeightSchema(BaseWeightSchema):
    quant_method: str = "quark"
    dtype: str
    qscheme: str | None = None
    ch_axis: int | None = None
    group_size: int | None = None
    block_size: tuple[int, int] | list[int] | None = None
    symmetric: bool | None = None
    is_dynamic: bool = False
    scale_format: str | None = None
    is_scale_quant: bool = False
    mx_element_dtype: str | None = None
    pack_method: str = "reorder"
    scale_dtype: str | None = None
    scale_qscheme: str | None = None
    scale_ch_axis: int | None = None
    scale_is_dynamic: bool = False
    scale_stage: int = 1

    def __post_init__(self):
        if self.is_dynamic or self.scale_is_dynamic:
            raise ValueError("Quark requires static quantized weights")
        if self.pack_method not in ("order", "reorder"):
            raise ValueError(f"unsupported Quark pack_method: {self.pack_method}")
        if self.dtype == "mx" and self.mx_element_dtype in ("fp4", "fp6_e2m3", "fp6_e3m2"):
            self.qscheme = "per_group"
            self.scale_format = "e8m0"
            if self.group_size != 32:
                raise ValueError("Quark embedded MX weights require group_size=32")
        elif self.dtype not in _STORAGE_DTYPES:
            raise ValueError(f"unsupported Quark weight dtype: {self.dtype}")

        if self.qscheme not in ("per_tensor", "per_channel", "per_group", "per_block"):
            raise ValueError(f"unsupported Quark weight qscheme: {self.qscheme}")
        if self.qscheme == "per_channel" and self.ch_axis not in (0, -2):
            raise ValueError("Quark per-channel weights must select output channels")
        if self.qscheme == "per_group":
            if self.ch_axis not in (-1, 1):
                raise ValueError("Quark per-group weights must group along K")
            if not self.group_size or self.group_size < 16 or self.group_size & (self.group_size - 1):
                raise ValueError("Quark weight group_size must be a power of two >= 16")
        else:
            self.group_size = 0
        if self.qscheme == "per_block":
            if self.dtype not in ("fp8_e4m3", "fp8_e5m2") or not self.block_size:
                raise ValueError("Quark block quantization requires FP8 weights and block_size")

        if self.scale_format not in (None, "float32", "float16", "bfloat16", "e8m0", "e5m3"):
            raise ValueError(f"unsupported Quark scale format: {self.scale_format}")
        if self.scale_dtype is not None:
            if self.scale_dtype not in ("fp8_e4m3", "fp8_e5m2", "fp8_e5m3"):
                raise ValueError(f"unsupported Quark secondary scale dtype: {self.scale_dtype}")
            if self.scale_qscheme not in ("per_tensor", "per_channel"):
                raise ValueError("Quark secondary scales require per-tensor or per-output-channel scales")
            if self.scale_qscheme == "per_channel" and self.scale_ch_axis not in (0, -2):
                raise ValueError("Quark secondary scales must select output channels")
        self.has_zero_point = self.is_integer and (self.symmetric is False or self.dtype.startswith("uint"))

    @property
    def is_integer(self) -> bool:
        return self.dtype.startswith(("int", "uint"))

    @property
    def weight_dtype(self) -> dtypes.DataType:
        if self.is_integer:
            return dtypes.IntegerType.from_str("uint" + self.dtype.removeprefix("uint").removeprefix("int"))
        element_dtype = self.mx_element_dtype if self.dtype == "mx" else self.dtype
        assert element_dtype is not None
        return _FLOAT_DTYPES[element_dtype]

    @property
    def storage_dtype(self) -> torch.dtype:
        return torch.uint8 if self.dtype == "mx" else _STORAGE_DTYPES[self.dtype]

    @property
    def transposes_groups(self) -> bool:
        return self.qscheme == "per_group" and self.dtype in ("int2", "int4", "uint4", "int8", "uint8")

    @property
    def tensor_scale_name(self) -> str:
        return "weight_scale_2" if self.scale_dtype is not None and self.scale_stage == 0 else "weight_scale"

    @property
    def zero_point_name(self) -> str:
        return self.tensor_scale_name.replace("scale", "zero_point")

    def _packed_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        if self.dtype == "mx":
            block_bytes = 1 + self.weight_dtype.num_bits * 4
            return (*shape[:-1], shape[-1] // 32 * block_bytes)
        if self.dtype in ("int2", "int8", "uint8", "fp8_e4m3", "fp8_e5m2"):
            return shape
        num_bits = self.weight_dtype.num_bits
        storage_bits = torch.empty((), dtype=self.storage_dtype).element_size() * 8
        return (*shape[:-1], shape[-1] * num_bits // storage_bits)

    def get_tensors_attrs(
        self,
        shape_n: int,
        shape_k: int,
        param_dtype: torch.dtype,
        num_experts: int | None = None,
        has_bias: bool = False,
        stack_size: int = 1,
    ) -> dict[str, dict[str, Any]]:
        num_bits = self.weight_dtype.num_bits
        if shape_k * num_bits % 32 or self.group_size and shape_k % self.group_size:
            raise ValueError("Quark weight K must align to its packed storage and quantization groups")
        shape = (shape_k, shape_n) if self.transposes_groups else (shape_n, shape_k)
        input_dim, output_dim = (0, 1) if self.transposes_groups else (1, 0)
        weight_attrs: dict[str, Any] = {"input_dim": input_dim, "output_dim": output_dim}
        packed_shape = self._packed_shape(shape)
        storage_bits = torch.empty((), dtype=self.storage_dtype).element_size() * 8
        is_element_packed = self.dtype not in ("mx", "int2")
        is_storage_aligned = packed_shape[-1] * storage_bits == shape[-1] * num_bits
        if is_element_packed and not is_storage_aligned:
            raise ValueError("Quark weight shape must align to its packed element storage")
        if packed_shape != shape:
            weight_attrs.update(packed_dim=1, packed_factor=shape[-1] / packed_shape[-1])

        scale_shape: tuple[int, ...]
        if self.qscheme == "per_group":
            assert self.group_size is not None
            scale_shape = (shape_n, shape_k // self.group_size)
            scale_attrs = {"output_dim": 0, "input_dim": 1, "scale_type": "group"}
            if self.transposes_groups:
                scale_shape = scale_shape[::-1]
                scale_attrs.update(output_dim=1, input_dim=0)
        elif self.qscheme == "per_block":
            assert self.block_size is not None
            scale_shape = (shape_n // self.block_size[0], shape_k // self.block_size[1])
            scale_attrs = {"output_dim": 0, "input_dim": 1, "scale_type": "block"}
        elif self.qscheme == "per_channel":
            scale_shape = (shape_n,)
            scale_attrs = {"output_dim": 0, "scale_type": "channel"}
        else:
            scale_shape = (stack_size,)
            scale_attrs = {"scale_type": "tensor"}

        scale_format = self.scale_dtype if self.scale_dtype and self.scale_stage == 1 else self.scale_format
        attrs: dict[str, dict[str, Any]] = {
            "weight": {"shape": packed_shape, "dtype": self.storage_dtype, "extra_attrs": weight_attrs},
            self.tensor_scale_name: {
                "shape": scale_shape,
                "dtype": _SCALE_DTYPES.get(scale_format, torch.float32),
                "extra_attrs": scale_attrs,
            },
        }
        if self.has_zero_point:
            zero_shape = scale_shape if self.qscheme == "per_tensor" else self._packed_shape(scale_shape)
            zero_attrs = dict(scale_attrs)
            if zero_shape != scale_shape:
                zero_attrs.update(
                    packed_dim=len(scale_shape) - 1,
                    packed_factor=scale_shape[-1] / zero_shape[-1],
                )
            attrs[self.zero_point_name] = {
                "shape": zero_shape,
                "dtype": self.storage_dtype,
                "extra_attrs": zero_attrs,
            }
        if self.scale_dtype is not None:
            scale_name = "weight_scale" if self.scale_stage == 0 else "weight_scale_2"
            is_channel = self.scale_qscheme == "per_channel"
            if is_channel:
                secondary_scale_attrs = {"output_dim": 0, "scale_type": "channel"}
            else:
                secondary_scale_attrs = {"scale_type": "tensor"}
            attrs[scale_name] = {
                "shape": (shape_n,) if is_channel else (stack_size,),
                "dtype": torch.float32,
                "extra_attrs": secondary_scale_attrs,
            }
        if has_bias:
            attrs["bias"] = {"shape": (shape_n,), "dtype": param_dtype, "extra_attrs": {"output_dim": 0}}
        if self.dtype == "mx":
            del attrs[self.tensor_scale_name]
        return self.may_add_expert_dim(attrs, num_experts)

    def infer_shape(self, tensors: dict[str, torch.Tensor]) -> tuple[int, int, int | None, bool]:
        weight = tensors["weight"]
        if weight.ndim not in (2, 3):
            raise ValueError("Quark weight must be a matrix or a stack of expert matrices")
        shape_n, shape_k = weight.shape[-2:]
        if self.dtype == "mx":
            shape_k = shape_k // (1 + self.weight_dtype.num_bits * 4) * 32
        elif self.dtype not in ("int2", "int8", "uint8", "fp8_e4m3", "fp8_e5m2"):
            shape_k = shape_k * weight.element_size() * 8 // self.weight_dtype.num_bits
        if self.transposes_groups:
            shape_n, shape_k = shape_k, shape_n
        return shape_n, shape_k, weight.size(0) if weight.ndim == 3 else None, "bias" in tensors

    def process_loaded_weight(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        expected = None
        if name in ("weight", self.zero_point_name):
            expected = self.storage_dtype
        elif name == self.tensor_scale_name:
            has_quantized_scale = bool(self.scale_dtype) and self.scale_stage == 1
            scale_format = self.scale_dtype if has_quantized_scale else self.scale_format
            expected = _SCALE_DTYPES.get(scale_format)
            if expected == torch.uint8 and tensor.dtype == torch.float8_e8m0fnu:
                tensor = tensor.view(torch.uint8)
        if expected is not None:
            if tensor.dtype != expected:
                raise ValueError(f"Quark {name} requires {expected}, got {tensor.dtype}")
        elif "scale" in name:
            tensor = tensor.float()
        if tensor.ndim == 0:
            tensor = tensor.reshape(1)
        return tensor.contiguous()

    def to_humming_schema(self, param_dtype: torch.dtype) -> HummingWeightSchema:
        scale_dtype = None
        if self.scale_format == "e8m0":
            scale_dtype = dtypes.float8e8m0
        has_native_scale = self.scale_dtype in ("fp8_e4m3", "fp8_e5m2")
        if self.scale_stage == 1 and has_native_scale and self.group_size:
            assert self.scale_dtype is not None
            scale_dtype = _FLOAT_DTYPES[self.scale_dtype]
        is_block = self.qscheme == "per_block"
        block_size_n, block_size_k = self.block_size or (0, 0)
        return HummingWeightSchema(
            b_dtype=self.weight_dtype,
            bs_dtype=dtypes.float32 if is_block else scale_dtype,
            weight_scale_group_size=block_size_k if is_block else (self.group_size or 0),
            weight_scale_group_size_n=block_size_n if is_block else 0,
            weight_scale_type=WeightScaleType.BLOCK if is_block else None,
            has_zero_point=self.has_zero_point,
        )

    def _unpack_integer_weight(self, tensor: torch.Tensor, is_zero_point: bool = False) -> torch.Tensor:
        num_bits = self.weight_dtype.num_bits
        is_scalar_zero = is_zero_point and self.qscheme == "per_tensor"
        if self.dtype in ("int3", "int4", "uint4") and not is_scalar_zero:
            tensor = _unpack_weight(tensor, num_bits, self.pack_method == "reorder" and num_bits == 4)
        else:
            tensor = tensor.to(torch.int32)
        tensor = tensor & ((1 << num_bits) - 1)
        if self.dtype.startswith("int"):
            tensor = tensor ^ (1 << (num_bits - 1))
        if self.transposes_groups:
            tensor = tensor.transpose(-1, -2)
        return tensor.contiguous()

    def _unpack_mx(self, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        num_bits = self.weight_dtype.num_bits
        blocks = weight.reshape(*weight.shape[:-1], -1, 1 + num_bits * 4)
        exponent = blocks[..., 0].contiguous().view(torch.int8).to(torch.int16)
        scale = (exponent + 127).to(torch.uint8).view(torch.float8_e8m0fnu)
        payload = blocks[..., 1:].contiguous()
        if num_bits == 4:
            weight = payload.flatten(-2).view(torch.int32)
        else:
            assert num_bits == 6
            # The old MXFP6 layout separates the high two and low four element bits.
            payload = payload.reshape(*blocks.shape[:-1], 8, 3)
            high_bits = payload[..., :1].reshape(*blocks.shape[:-1], 8, 1)
            low_bits = payload[..., 1:].reshape(*blocks.shape[:-1], 16, 1)
            high_shifts = torch.tensor([6, 4, 2, 0], device=weight.device)
            low_shifts = torch.tensor([4, 0], device=weight.device)
            high_bits = ((high_bits >> high_shifts) & 3).flatten(-2)
            low_bits = ((low_bits >> low_shifts) & 15).flatten(-2)
            weight = ((high_bits << 4) | low_bits).flatten(-2)
            weight = _pack_weight(weight, num_bits)
        return weight, scale

    def _convert_humming(
        self,
        tensors: dict[str, torch.Tensor],
        shape_n_stacks: list[int],
        shape_k_stacks: list[int],
        param_dtype: torch.dtype,
        num_experts: int | None = None,
    ) -> tuple[HummingWeightSchema, dict[str, torch.Tensor]]:
        tensors = {name: self.process_loaded_weight(tensor, name) for name, tensor in tensors.items()}
        schema = self.to_humming_schema(param_dtype)
        weight = tensors["weight"]
        scale_format = self.scale_dtype if self.scale_dtype and self.scale_stage == 1 else self.scale_format
        if self.dtype == "mx":
            weight, scale = self._unpack_mx(weight)
        else:
            scale = _decode_scale(tensors[self.tensor_scale_name], scale_format)

        if self.transposes_groups:
            scale = scale.transpose(-1, -2)
        if self.qscheme == "per_tensor":
            scale = self._may_process_global_scale(
                scale,
                shape_n_stacks,
                shape_k_stacks,
                num_experts or None,
                force_repeat=True,
            )
            if len(shape_k_stacks) > 1:
                schema.weight_scale_group_size = sum(shape_k_stacks) // scale.size(-1)
                schema.weight_scale_type = WeightScaleType.GROUP
        elif self.qscheme == "per_channel":
            scale = scale.unsqueeze(-1)

        result = {}
        if self.is_integer:
            weight = self._unpack_integer_weight(weight)
            weight = _pack_weight(weight, self.weight_dtype.num_bits)
            if self.has_zero_point:
                zero_point = self._unpack_integer_weight(tensors[self.zero_point_name], is_zero_point=True)
                if self.qscheme == "per_tensor":
                    zero_point = self._may_process_global_scale(
                        zero_point,
                        shape_n_stacks,
                        shape_k_stacks,
                        num_experts or None,
                        force_repeat=True,
                    )
                elif self.qscheme == "per_channel":
                    zero_point = zero_point.unsqueeze(-1)
                if not self.group_size:
                    schema.weight_scale_group_size = sum(shape_k_stacks) // scale.size(-1)
                    schema.weight_scale_type = WeightScaleType.GROUP
                zero_point = zero_point.transpose(-1, -2).contiguous()
                packed_zero = _pack_weight(zero_point, self.weight_dtype.num_bits)
                result["zero_point"] = packed_zero.transpose(-1, -2).contiguous()
        elif self.dtype != "mx":
            # Quark FP4/FP6/FP8 store consecutive little-endian element bits.
            weight = weight.view(torch.int32)

        if self.scale_dtype is not None:
            scale_name = "weight_scale" if self.scale_stage == 0 else "weight_scale_2"
            scale_2 = tensors[scale_name]
            if self.scale_qscheme == "per_channel":
                scale = scale.float() * scale_2.unsqueeze(-1)
                schema.bs_dtype = None
            else:
                scale_2 = self._may_process_global_scale(
                    scale_2,
                    shape_n_stacks,
                    shape_k_stacks,
                    num_experts or None,
                    target_group_size=schema.weight_scale_group_size or None,
                )
                is_group_scale = schema.weight_scale_type == WeightScaleType.GROUP
                keep_scale_2 = schema.bs_dtype is not None and is_group_scale
                if keep_scale_2 and scale_2.numel() == (num_experts or 1):
                    schema.weight_scale_2_type = WeightScale2Type.TENSOR
                    result["weight_scale_2"] = scale_2.reshape(num_experts or 1).float()
                else:
                    if scale_2.numel() == (num_experts or 1):
                        scale_2 = scale_2.reshape(-1, 1, 1) if num_experts else scale_2.reshape(())
                    scale = scale.float() * scale_2
                    schema.bs_dtype = None

        if schema.weight_scale_type == WeightScaleType.BLOCK:
            scale = scale.float()
        elif schema.bs_dtype == dtypes.float8e8m0:
            scale = scale.view(torch.float8_e8m0fnu)
        elif schema.bs_dtype in (dtypes.float8e4m3, dtypes.float8e5m2):
            scale = scale.to(dtypes.torch_dtype_map[schema.bs_dtype])
        else:
            scale = scale.to(param_dtype)
        result.update(weight=weight.contiguous(), weight_scale=scale.contiguous())
        if "bias" in tensors:
            result["bias"] = tensors["bias"].to(param_dtype)
        return schema, result


@dataclasses.dataclass(kw_only=True)
class QuarkInputSchema(BaseInputSchema):
    quant_method: str = "quark"
    dtype: str
    qscheme: str | None = None
    ch_axis: int | None = None
    group_size: int | None = None
    symmetric: bool | None = None
    is_dynamic: bool = False
    scale_format: str | None = None
    is_scale_quant: bool = False
    mx_element_dtype: str | None = None
    scale_dtype: str | None = None
    scale_qscheme: str | None = None
    scale_ch_axis: int | None = None
    scale_is_dynamic: bool = False
    scale_stage: int = 1

    def __post_init__(self):
        if self.dtype in ("float16", "bfloat16"):
            return
        element_dtype = self.mx_element_dtype if self.dtype == "mx" else self.dtype
        if self.dtype == "mx" and element_dtype == "fp4":
            self.qscheme = "per_group"
            self.scale_format = "e8m0"
            self.is_dynamic = True
        if element_dtype not in ("fp4", "fp8_e4m3", "fp8_e5m2", "int4", "int8"):
            raise ValueError(f"unsupported Quark activation dtype: {self.dtype}")
        if self.symmetric is False or self.dtype in ("int4", "int8") and self.symmetric is not True:
            raise ValueError("Quark activations require symmetric quantization")
        if self.scale_dtype is not None:
            is_nvfp4 = self.dtype == "fp4" and self.group_size == 16 and self.scale_dtype == "fp8_e4m3"
            static_scale = self.scale_qscheme == "per_tensor" and not self.scale_is_dynamic
            token_scale = self.scale_qscheme == "per_channel" and self.scale_ch_axis in (0, -2)
            token_scale = token_scale and self.scale_is_dynamic
            if is_nvfp4 and self.is_dynamic and self.scale_stage == 1 and (static_scale or token_scale):
                return
            raise ValueError("Quark activation scale quantization requires NVFP4 tensor/token scaling")
        if self.scale_format not in (None, "float32", "float16", "bfloat16", "e8m0"):
            raise ValueError(f"unsupported Quark activation scale format: {self.scale_format}")
        if element_dtype == "fp4" and self.scale_format != "e8m0":
            raise ValueError("Quark FP4 activations require MXFP4 or NVFP4 scales")
        if self.scale_format == "e8m0" and not (self.is_dynamic and self.qscheme == "per_group"):
            raise ValueError("Quark E8M0 activation scales require dynamic group quantization")
        if not self.is_dynamic and self.qscheme == "per_tensor":
            return
        if self.is_dynamic and self.qscheme == "per_channel" and self.ch_axis in (0, -2):
            return
        is_grouped_k = self.qscheme == "per_group" and self.ch_axis in (-1, 1)
        if self.is_dynamic and is_grouped_k:
            min_group_size = 64 if self.dtype == "int4" else 32
            if self.group_size and min_group_size <= self.group_size <= 512:
                if not self.group_size & (self.group_size - 1):
                    if self.scale_format != "e8m0" or self.group_size == 32:
                        return
        raise ValueError(f"unsupported Quark input specification: {self}")

    @property
    def activation_dtype(self) -> dtypes.DataType:
        if self.dtype == "mx":
            assert self.mx_element_dtype is not None
            return _FLOAT_DTYPES[self.mx_element_dtype]
        if self.dtype in _FLOAT_DTYPES:
            return _FLOAT_DTYPES[self.dtype]
        return dtypes.DataType.from_str(self.dtype)

    def get_activation_bits(self) -> int:
        return self.activation_dtype.num_bits

    def to_humming_schema(self, param_dtype: torch.dtype) -> HummingInputSchema:
        if self.dtype in ("float16", "bfloat16"):
            return HummingInputSchema(a_dtype=dtypes.DataType.from_torch_dtype(param_dtype))
        group_size = self.group_size if self.qscheme == "per_group" else 0
        scale_dtype = dtypes.float8e8m0 if self.scale_format == "e8m0" else None
        if self.scale_dtype is not None:
            if self.scale_is_dynamic:
                quant_mode = InputQuantizationMode.DynamicGroupToken
            else:
                quant_mode = InputQuantizationMode.StaticTensorDynamicGroup
            scale_dtype = dtypes.float8e4m3
        elif not self.is_dynamic:
            quant_mode = InputQuantizationMode.StaticTensor
        elif group_size:
            quant_mode = InputQuantizationMode.DynamicGroup
        else:
            quant_mode = InputQuantizationMode.DynamicToken
        return HummingInputSchema(
            a_dtype=self.activation_dtype,
            input_scale_group_size=group_size or 0,
            input_scale_dtype=scale_dtype,
            input_quant_mode=quant_mode,
        )

    def get_tensors_attrs(
        self,
        shape_k: int,
        param_dtype: torch.dtype,
        num_experts: int | None = None,
        stack_size: int = 1,
    ) -> dict[str, dict[str, Any]]:
        schema = self.to_humming_schema(param_dtype)
        if schema.input_scale_group_size and shape_k % schema.input_scale_group_size:
            raise ValueError("Quark activation K must be divisible by group_size")
        if schema.static_tensor_scale_name is not None:
            source_name = "input_scale_2" if self.scale_dtype is not None else "input_scale"
            return self._get_input_scale_attrs(num_experts, stack_size, input_scale_name=source_name)
        return {}

    def _convert_humming(
        self,
        tensors: dict[str, torch.Tensor],
        shape_n_stacks: list[int],
        shape_k_stacks: list[int],
        param_dtype: torch.dtype,
        num_experts: int | None = None,
    ) -> tuple[HummingInputSchema, dict[str, torch.Tensor]]:
        schema = self.to_humming_schema(param_dtype)
        if schema.static_tensor_scale_name is None:
            return schema, {}
        source_name = "input_scale_2" if self.scale_dtype is not None else "input_scale"
        scale = tensors[source_name].float()
        if not torch.all(scale == scale.flatten()[0]):
            raise ValueError("Quark fused layers/experts require equal static input scales")
        return schema, {schema.static_tensor_scale_name: scale.flatten()[:1].contiguous()}
