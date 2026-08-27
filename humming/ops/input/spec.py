"""Shared dtype and quantization metadata for process-input operations."""

import torch

from .enums import QuantizationMode

HIDDEN_BASE = {"float8e3m4": "float8e5m2", "float4e0m3": "float4e2m1"}

QUANT_STORAGE = {
    "int4": (torch.uint8, 2),
    "float4e2m1": (torch.uint8, 2),
    "float4e0m3": (torch.uint8, 2),
    "int8": (torch.int8, 1),
    "float8e4m3": (torch.float8_e4m3fn, 1),
    # E3M4 is emitted as E5M2 and patched after compilation. The launch-time
    # tensor carries the base dtype; the public result is a uint8 view.
    "float8e3m4": (torch.float8_e5m2, 1),
    "float8e5m2": (torch.float8_e5m2, 1),
}

E8M0_TORCH_DTYPE = getattr(torch, "float8_e8m0fnu", torch.uint8)
SCALE_TORCH_DTYPE = {
    "float32": torch.float32,
    "float8e4m3": torch.float8_e4m3fn,
    "float8e8m0": E8M0_TORCH_DTYPE,
}

STATIC_TENSOR, STATIC_GROUP = 1, 2

# (dynamic scale mode, static scale bitmask). A scale argument has exactly one
# role in each mode: static input or dynamic output.
QUANT_MODE_SPECS = {
    QuantizationMode.Disabled: ("none", 0),
    QuantizationMode.StaticTensor: ("static", STATIC_TENSOR),
    QuantizationMode.StaticGroup: ("static", STATIC_GROUP),
    QuantizationMode.StaticTensorGroup: ("static", STATIC_TENSOR | STATIC_GROUP),
    QuantizationMode.DynamicToken: ("token", 0),
    QuantizationMode.DynamicGroup: ("group", 0),
    QuantizationMode.StaticTensorDynamicGroup: ("group", STATIC_TENSOR),
    QuantizationMode.DynamicGroupToken: ("group_token", 0),
}
