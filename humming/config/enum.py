import enum


class MmaType(enum.Enum):
    MMA = enum.auto()
    WGMMA = enum.auto()


class ActivationType(enum.Enum):
    NONE = enum.auto()
    SIGMOID = enum.auto()
    TANH = enum.auto()
    RELU = enum.auto()
    GELU = enum.auto()
    FASTGELU = enum.auto()
    QUICKGELU = enum.auto()
    SILU = enum.auto()
    CUSTOM = enum.auto()
    SILU_GLU = enum.auto()
    CUSTOM_GLU = enum.auto()
