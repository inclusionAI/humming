#pragma once


enum class ProcessInputLayoutType : uint32_t {
  Normal = 0,
  GroupedMask = 3,
  Scatter = 4,
};


enum class ProcessInputQuantizationPhase : uint32_t {
  Fused = 0,
  CollectAbsmax = 1,
  Quantize = 2,
};


enum class InputQuantizationMode : uint32_t {
  Disabled = 0,
  StaticTensor = 1,
  DynamicToken = 2,
  DynamicGroup = 3,
  StaticTensorDynamicGroup = 4,
  DynamicGroupToken = 5,
};


enum class WeightScaleType : uint32_t {
  GROUP,
  BLOCK,
  CHANNEL,
  TENSOR,
};


enum class WeightScale2Type : uint32_t {
  NONE,
  CHANNEL,
  TENSOR,
};


enum class MmaType : uint32_t {
  MMA,
  WGMMA,
  UMMA,
  MXMMA
};


enum class GemmType : uint32_t {
  DENSE,
  INDEXED,
  GROUPED_CONTIGUOUS,
  GROUPED_MASKED,
};


enum class SmemReuseMode : uint32_t {
  NONE = 0,
  LAST_STAGE = 1,
  ALL_STAGES = 2,
};
