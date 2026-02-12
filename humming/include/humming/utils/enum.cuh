#pragma once


enum class ScaleType : uint32_t {
  NONE,
  CHANNELWISE,
  GROUPWISE
};


enum class ActivationType : uint32_t {
  NONE,
  SIGMOID,
  TANH,
  RELU,
  GELU,
  FASTGELU,
  QUICKGELU,
  SILU,
  CUSTOM,
  SILU_GLU,
  CUSTOM_GLU
};


enum class MmaType : uint32_t {
  MMA,
  WGMMA
};
