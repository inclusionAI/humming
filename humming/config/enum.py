import enum


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
