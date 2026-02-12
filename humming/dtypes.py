import re
import torch


class DataType(object):
    def __init__(self):
        self.is_integer_type = False
        self.is_floating_point_type = False

    @classmethod
    def from_str(cls, s):
        if isinstance(s, DataType):
            return s
        if "float" in s:
            return FloatingPointType.from_str(s)
        elif "int" in s:
            return InergerType.from_str(s)
        else:
            raise NotImplementedError

    @classmethod
    def from_torch_dtype(cls, torch_dtype):
        if "float" in str(torch_dtype):
            return FloatingPointType.from_torch_dtype(torch_dtype)
        elif "int" in str(torch_dtype):
            return InergerType.from_torch_dtype(torch_dtype)
        else:
            raise NotImplementedError

    def to_str(self, lowercase=True):
        raise NotImplementedError

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(self) == str(other)

    def __repr__(self):
        return self.to_str()


class InergerType(DataType):
    def __init__(self, is_signed, num_bits):
        super().__init__()
        self.is_integer_type = True
        self.is_signed = is_signed
        self.num_bits = num_bits

    @classmethod
    def from_str(cls, s):
        if isinstance(s, InergerType):
            return s
        s = s.lower()
        re_res = re.findall("^(u*)int(\\d+)$", s)
        if not re_res:
            raise ValueError(f"invalid integer dtype: {s}")
        re_res = re_res[0]
        return cls(is_signed=re_res[0] == "", num_bits=int(re_res[1]))

    def to_str(self, lowercase=True):
        s = "Int" if self.is_signed else "UInt"
        s += str(self.num_bits)
        return s.lower() if lowercase else s

    def to_cpp_str(self):
        return "IntegerType<{is_signed}, {num_bits}>".format(
            is_signed=str(self.is_signed).lower(),
            num_bits=str(self.num_bits),
        )

    @classmethod
    def from_torch_dtype(cls, torch_dtype):
        assert isinstance(torch_dtype, torch.dtype)
        dtype_str = str(torch_dtype)[6:]
        assert "int" in dtype_str
        return cls.from_str(dtype_str)


class FloatingPointType(DataType):
    def __init__(self, num_bits, exponent_bits, mantissa_bits):
        super().__init__()
        self.is_floating_point_type = True
        self.num_bits = num_bits
        self.sign_bits = num_bits - exponent_bits - mantissa_bits
        assert self.sign_bits in (0, 1)
        self.is_signed = self.sign_bits != 0
        self.exponent_bits = exponent_bits
        self.mantissa_bits = mantissa_bits

    @classmethod
    def from_str(cls, s):
        if isinstance(s, FloatingPointType):
            return s
        s = s.lower()
        if s in ("float16", "half"):
            return cls(16, 5, 10)
        elif s == "bfloat16":
            return cls(16, 8, 7)
        elif s == "float32":
            return cls(32, 8, 23)

        re_res = re.findall("^float(\\d+)_*e(\\d+)m(\\d+)$", s)
        if not re_res:
            raise ValueError(f"invalid floating point dtype: {s}")

        re_res = re_res[0]
        return cls(
            num_bits=int(re_res[0]),
            exponent_bits=int(re_res[1]),
            mantissa_bits=int(re_res[2]),
        )

    def to_str(self, lowercase=True):
        s = f"Float{self.num_bits}E{self.exponent_bits}M{self.mantissa_bits}"
        if s == "Float16E5M10":
            s = "Float16"
        elif s == "Float16E8M7":
            s = "BFloat16"
        elif s == "Float32E8M23":
            s = "Float32"

        return s.lower() if lowercase else s

    def to_cpp_str(self):
        return "FloatingPointType<{num_bits}, {exponent_bits}, {mantissa_bits}>".format(
            num_bits=self.num_bits,
            exponent_bits=self.exponent_bits,
            mantissa_bits=self.mantissa_bits,
        )

    @classmethod
    def from_torch_dtype(cls, torch_dtype):
        assert isinstance(torch_dtype, torch.dtype)
        dtype_str = str(torch_dtype)[6:]
        # note that fnuz format / packed format / padded format are not supported
        dtype_str = dtype_str.replace("fn", "").replace("fnu", "")
        return cls.from_str(dtype_str)


uint1 = InergerType.from_str("uint1")
uint2 = InergerType.from_str("uint2")
uint3 = InergerType.from_str("uint3")
uint4 = InergerType.from_str("uint4")
uint5 = InergerType.from_str("uint5")
uint6 = InergerType.from_str("uint6")
uint7 = InergerType.from_str("uint7")
uint8 = InergerType.from_str("uint8")

int2 = InergerType.from_str("int2")
int3 = InergerType.from_str("int3")
int4 = InergerType.from_str("int4")
int6 = InergerType.from_str("int6")
int8 = InergerType.from_str("int8")
int32 = InergerType.from_str("int32")

float4e2m1 = FloatingPointType.from_str("float4e2m1")
float6e2m3 = FloatingPointType.from_str("float6e2m3")
float6e3m2 = FloatingPointType.from_str("float6e3m2")
float8e4m3 = FloatingPointType.from_str("float8e4m3")
float8e5m2 = FloatingPointType.from_str("float8e5m2")
float8e8m0 = FloatingPointType.from_str("float8e8m0")

float16 = FloatingPointType.from_str("float16")
bfloat16 = FloatingPointType.from_str("bfloat16")
float32 = FloatingPointType.from_str("float32")


torch_dtype_map = {
    float8e8m0: torch.float8_e8m0fnu,
    float8e4m3: torch.float8_e4m3fn,
    float8e5m2: torch.float8_e5m2,
    float16: torch.float16,
    bfloat16: torch.bfloat16,
}
