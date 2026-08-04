from humming.schema.autoround import AutoRoundWeightSchema
from humming.schema.awq import AWQWeightSchema
from humming.schema.base import BaseInputSchema, BaseWeightSchema
from humming.schema.bitnet import BitnetWeightSchema
from humming.schema.compressed_tensors import (
    CompressedTensorsInputSchema,
    CompressedTensorsWeightSchema,
)
from humming.schema.fp8 import Fp8InputSchema, Fp8WeightSchema
from humming.schema.gpt_oss_mxfp4 import GptOssMxfp4WeightSchema
from humming.schema.gptq import GPTQWeightSchema
from humming.schema.humming import HummingInputSchema, HummingWeightSchema
from humming.schema.modelopt import ModeloptInputSchema, ModeloptWeightSchema
from humming.schema.mxfp4 import Mxfp4WeightSchema

WEIGHT_SCHEMA_MAP: dict[str, type[BaseWeightSchema]] = {
    "auto-round": AutoRoundWeightSchema,
    "auto_round": AutoRoundWeightSchema,
    "awq": AWQWeightSchema,
    "bitnet": BitnetWeightSchema,
    "compressed-tensors": CompressedTensorsWeightSchema,
    "fp8": Fp8WeightSchema,
    "gptq": GPTQWeightSchema,
    "humming": HummingWeightSchema,
    "modelopt": ModeloptWeightSchema,
    "mxfp4": Mxfp4WeightSchema,
    "gpt_oss_mxfp4": GptOssMxfp4WeightSchema,
}

INPUT_SCHEMA_MAP: dict[str, type[BaseInputSchema]] = {
    "compressed-tensors": CompressedTensorsInputSchema,
    "fp8": Fp8InputSchema,
    "humming": HummingInputSchema,
    "modelopt": ModeloptInputSchema,
}

BaseWeightSchema.WEIGHT_SCHEMA_MAP = WEIGHT_SCHEMA_MAP
BaseInputSchema.INPUT_SCHEMA_MAP = INPUT_SCHEMA_MAP


__all__ = [
    "BaseInputSchema",
    "BaseWeightSchema",
    "HummingInputSchema",
    "HummingWeightSchema",
]
