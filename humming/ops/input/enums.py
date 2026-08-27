from enum import Enum


class QuantizationMode(str, Enum):
    Disabled = "none"
    StaticTensor = "static_tensor"
    StaticGroup = "static_group"
    StaticTensorGroup = "static_tensor_group"
    DynamicToken = "dynamic_token"
    DynamicGroup = "dynamic_group"
    StaticTensorDynamicGroup = "static_tensor_dynamic_group"
    DynamicGroupToken = "dynamic_group_token"


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
