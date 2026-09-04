from enum import Enum


class QuantizationMode(str, Enum):
    Disabled = "none"
    StaticTensor = "static_tensor"
    DynamicToken = "dynamic_token"
    DynamicGroup = "dynamic_group"
    StaticTensorDynamicGroup = "static_tensor_dynamic_group"
    DynamicGroupToken = "dynamic_group_token"

    @property
    def should_quantize(self) -> bool:
        return self != QuantizationMode.Disabled

    @property
    def has_static_tensor_scale(self) -> bool:
        return self in (QuantizationMode.StaticTensor, QuantizationMode.StaticTensorDynamicGroup)

    @property
    def has_dynamic_token_scale(self) -> bool:
        return self in (QuantizationMode.DynamicToken, QuantizationMode.DynamicGroupToken)

    @property
    def has_dynamic_group_scale(self) -> bool:
        modes = (
            QuantizationMode.DynamicGroup,
            QuantizationMode.StaticTensorDynamicGroup,
            QuantizationMode.DynamicGroupToken,
        )
        return self in modes

    @property
    def has_dynamic_scale(self) -> bool:
        return self.has_dynamic_token_scale or self.has_dynamic_group_scale

    @property
    def dynamic_scale_mode(self) -> str | None:
        if self.has_dynamic_token_scale:
            return "group_token" if self.has_dynamic_group_scale else "token"
        if self.has_dynamic_group_scale:
            return "group"
        return None

    @property
    def uses_token_scale(self) -> bool:
        return self.has_static_tensor_scale or self.has_dynamic_token_scale

    @property
    def uses_group_scale(self) -> bool:
        return self.has_dynamic_group_scale


class QuantizationPhase(str, Enum):
    Fused = "fused"
    CollectAbsmax = "collect_absmax"
    Quantize = "quantize"


class ActivationType(str, Enum):
    None_ = "none"
    Unary = "unary"
    BinarySplit = "binary_split"
    BinaryInterleaved = "binary_interleaved"

    @property
    def cpp_name(self) -> str:
        return self.name.removesuffix("_")


class LayoutType(str, Enum):
    Normal = "normal"
    Grouped = "grouped"
    Permute = "permute"
    GroupedPadded = "grouped_padded"
    Scatter = "scatter"


class GroupScaleLayout(str, Enum):
    RowMajor = "row_major"
    MMajor = "m_major"
    MxPacked = "mx_packed"
