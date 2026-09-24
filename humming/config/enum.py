import enum


class InputQuantizationMode(str, enum.Enum):
    Disabled = "none"
    StaticTensor = "static_tensor"
    DynamicToken = "dynamic_token"
    DynamicGroup = "dynamic_group"
    StaticTensorDynamicGroup = "static_tensor_dynamic_group"
    DynamicGroupToken = "dynamic_group_token"

    @property
    def should_quantize(self) -> bool:
        return self != InputQuantizationMode.Disabled

    @property
    def has_tensor_scale(self) -> bool:
        return self in (InputQuantizationMode.StaticTensor, InputQuantizationMode.StaticTensorDynamicGroup)

    @property
    def has_token_scale(self) -> bool:
        return self in (InputQuantizationMode.DynamicToken, InputQuantizationMode.DynamicGroupToken)

    @property
    def has_group_scale(self) -> bool:
        return self in (
            InputQuantizationMode.DynamicGroup,
            InputQuantizationMode.StaticTensorDynamicGroup,
            InputQuantizationMode.DynamicGroupToken,
        )

    @property
    def has_dynamic_scale(self) -> bool:
        return self.has_token_scale or self.has_group_scale

    @property
    def dynamic_scale_mode(self) -> str | None:
        if self.has_token_scale:
            return "group_token" if self.has_group_scale else "token"
        if self.has_group_scale:
            return "group"
        return None

    @property
    def has_secondary_scale(self) -> bool:
        return self.has_group_scale and (self.has_tensor_scale or self.has_token_scale)


class MmaType(enum.Enum):
    MMA = "mma"
    WGMMA = "wgmma"
    UMMA = "umma"
    MXMMA = "mxmma"


class WeightScaleType(enum.Enum):
    GROUP = "group"
    BLOCK = "block"
    CHANNEL = "channel"
    TENSOR = "tensor"


class WeightScale2Type(enum.Enum):
    NONE = "none"
    CHANNEL = "channel"
    TENSOR = "tensor"


class GemmType(enum.Enum):
    DENSE = "dense"
    INDEXED = "indexed"
    GROUPED_CONTIGUOUS = "grouped_contiguous"
    GROUPED_MASKED = "grouped_masked"


class ProcessInputQuantizationPhase(str, enum.Enum):
    Fused = "fused"
    CollectAbsmax = "collect_absmax"
    Quantize = "quantize"


class ActivationType(str, enum.Enum):
    None_ = "none"
    Unary = "unary"
    BinarySplit = "binary_split"
    BinaryInterleaved = "binary_interleaved"

    @property
    def cpp_name(self) -> str:
        return self.name.removesuffix("_")

    @property
    def is_unary(self) -> bool:
        return self == ActivationType.Unary

    @property
    def is_binary(self) -> bool:
        return self in (ActivationType.BinarySplit, ActivationType.BinaryInterleaved)


class ProcessInputLayoutType(str, enum.Enum):
    Normal = "normal"
    GroupedMask = "grouped_mask"
    Scatter = "scatter"


class SmemReuseMode(str, enum.Enum):
    NONE = "none"
    LAST_STAGE = "last_stage"
    ALL_STAGES = "all_stages"
