#pragma once

#include <cuda_fp8.h>

#include <humming/datatype/dtypes.cuh>
#include <humming/utils/all.cuh>


enum class ScaleMode : uint32_t {
  Static = 0,
  DynamicToken = 1,
  DynamicGroup = 2,
  DynamicGroupToken = 3,
};


enum class QuantizationMode : uint32_t {
  Disabled = 0,
  StaticTensor = 1,
  DynamicToken = 2,
  DynamicGroup = 3,
  StaticTensorDynamicGroup = 4,
  DynamicGroupToken = 5,
};


enum class QuantizationPhase : uint32_t {
  Fused = 0,
  CollectAbsmax = 1,
  Quantize = 2,
};


enum class GroupScaleLayout : uint32_t {
  RowMajor = 0,
  MMajor = 1,
  MxPacked = 2,
};


template <uint32_t kBytes>
CUDA_INLINE void store_packed(uint8_t *output, const uint8_t *packed, bool zero) {
  using StoreType = typename LoadTypeChooser<kBytes>::Type;
  constexpr uint32_t kStoreBytes = sizeof(StoreType);
  static_assert(kBytes % kStoreBytes == 0);
  PRAGMA_UNROLL
  for (uint32_t byte = 0; byte < kBytes; byte += kStoreBytes) {
    StoreType stored{};
    if (!zero) stored = *reinterpret_cast<const StoreType *>(packed + byte);
    *reinterpret_cast<StoreType *>(output + byte) = stored;
  }
}


// Internal two-byte storage for the first stage of DynamicGroup E4M3 +
// DynamicToken.  It has BF16's exponent range, but only three mantissa bits
// are retained so division by the power-of-two token scale is exactly
// representable by E4M3 unless the result underflows.
struct M3BFloat16 {};


template <class ScaleType>
constexpr bool public_scale_type =
    std::is_same<ScaleType, Float32>::value ||
    std::is_same<ScaleType, Float8E4M3>::value ||
    std::is_same<ScaleType, Float8E8M0>::value;


template <class ScaleType>
constexpr bool supported_scale_type = public_scale_type<ScaleType> || std::is_same<ScaleType, M3BFloat16>::value;


template <class ScaleType>
using ScaleStorage = typename std::conditional<
    std::is_same<ScaleType, Float32>::value,
    float,
    typename std::conditional<std::is_same<ScaleType, M3BFloat16>::value, uint16_t, uint8_t>::type>::type;


template <class ScaleType>
CUDA_INLINE float decode_scale(ScaleStorage<ScaleType> encoded) {
  static_assert(supported_scale_type<ScaleType>);
  if constexpr (std::is_same<ScaleType, Float32>::value) {
    return encoded;
  } else if constexpr (std::is_same<ScaleType, Float8E4M3>::value) {
    __nv_fp8_e4m3 value;
    value.__x = encoded;
    return static_cast<float>(value);
  } else if constexpr (std::is_same<ScaleType, M3BFloat16>::value) {
    return __uint_as_float(static_cast<uint32_t>(encoded) << 16);
  } else {
    return __uint_as_float(static_cast<uint32_t>(encoded) << 23);
  }
}


template <class ScaleType>
CUDA_INLINE ScaleStorage<ScaleType> encode_scale(float value) {
  static_assert(supported_scale_type<ScaleType>);
  if constexpr (std::is_same<ScaleType, Float32>::value) {
    return value;
  } else if constexpr (std::is_same<ScaleType, Float8E4M3>::value) {
    return static_cast<__nv_fp8_e4m3>(value).__x;
  } else if constexpr (std::is_same<ScaleType, M3BFloat16>::value) {
    uint32_t bits = __float_as_uint(value);
    uint32_t retained_lsb = (bits >> 20) & 1u;
    bits = (bits + 0x0007FFFFu + retained_lsb) & 0xFFF00000u;
    return static_cast<uint16_t>(bits >> 16);
  } else {
    uint32_t bits = __float_as_uint(value);
    bits = (bits + 0x007FFFFFu) & 0x7F800000u;
    return static_cast<uint8_t>(bits >> 23);
  }
}


template <class ScaleType>
CUDA_INLINE float load_scale(const void *scales, uint64_t index) {
  return decode_scale<ScaleType>(reinterpret_cast<const ScaleStorage<ScaleType> *>(scales)[index]);
}


template <class ScaleType>
CUDA_INLINE void store_scale(void *scales, uint64_t index, ScaleStorage<ScaleType> value) {
  reinterpret_cast<ScaleStorage<ScaleType> *>(scales)[index] = value;
}


// 448 = 1.75 * 2^8.  M3 scales have only three mantissa bits, so the
// power-of-two token scale is determined exactly by the exponent and whether
// the significand is greater than 1.75.  Keep variants for both live formats
// to avoid converting between FP32 bits and the compact scale on hot paths.
CUDA_INLINE float token_scale_from_m3_bits(uint32_t scale_bits) {
  if (scale_bits == 0) return 0.f;
  int32_t exponent = static_cast<int32_t>(scale_bits & 0x7F800000u) - 0x04000000;
  if ((scale_bits & 0x007F0000u) > 0x00600000u) exponent += 0x00800000;
  exponent = max(0x00800000, min(0x7F000000, exponent));
  return __uint_as_float(static_cast<uint32_t>(exponent));
}


CUDA_INLINE float token_scale_from_m3(ScaleStorage<M3BFloat16> scale) {
  if (scale == 0) return 0.f;
  int32_t exponent = static_cast<int32_t>(scale & 0x7F80u) - 0x400;
  if ((scale & 0x7Fu) > 0x60u) exponent += 0x80;
  exponent = max(0x80, min(0x7F00, exponent));
  return __uint_as_float(static_cast<uint32_t>(exponent) << 16);
}


CUDA_INLINE float warp_max(float value) {
#if __CUDA_ARCH__ >= 800
  return __uint_as_float(__reduce_max_sync(0xFFFFFFFFu, __float_as_uint(value)));
#else
  PRAGMA_UNROLL
  for (uint32_t step = 1; step < 32; step <<= 1)
    value = fmaxf(value, __shfl_xor_sync(0xFFFFFFFFu, value, step));
  return value;
#endif
}


CUDA_INLINE uint32_t warp_max(uint32_t value) {
#if __CUDA_ARCH__ >= 800
  return __reduce_max_sync(0xFFFFFFFFu, value);
#else
  PRAGMA_UNROLL
  for (uint32_t step = 1; step < 32; step <<= 1)
    value = max(value, __shfl_xor_sync(0xFFFFFFFFu, value, step));
  return value;
#endif
}


template <uint32_t kValues>
CUDA_INLINE float lane_absmax(const float *values) {
  static_assert(kValues >= 1 && (kValues & (kValues - 1)) == 0);
  constexpr uint32_t kChains = kValues < 4 ? kValues : 4;
  float partial[kChains];
  PRAGMA_UNROLL
  for (uint32_t chain = 0; chain < kChains; chain++)
    partial[chain] = fabsf(values[chain]);
  PRAGMA_UNROLL
  for (uint32_t value = kChains; value < kValues; value++)
    partial[value % kChains] = fmaxf(
        partial[value % kChains], fabsf(values[value]));
  PRAGMA_UNROLL
  for (uint32_t step = kChains / 2; step > 0; step >>= 1)
    for (uint32_t chain = 0; chain < step; chain++)
      partial[chain] = fmaxf(partial[chain], partial[chain + step]);
  return partial[0];
}


template <uint32_t kValuesPerLane, uint32_t kGroupSize, uint32_t kNumWarps, uint32_t kWarpOffset>
CUDA_INLINE float group_absmax(const float *values, float *shared) {
  static_assert(kValuesPerLane >= 1);
  static_assert((kValuesPerLane & (kValuesPerLane - 1)) == 0);
  static_assert(kGroupSize >= kValuesPerLane);
  static_assert(kGroupSize % kValuesPerLane == 0);
  static_assert(kNumWarps >= 1);

  constexpr uint32_t kLanes = kGroupSize / kValuesPerLane;
  constexpr uint32_t kThreads = kNumWarps * 32;
  constexpr uint32_t kGroups = kThreads / kLanes;
  static_assert(kLanes >= 1 && kLanes <= 1024);
  static_assert(kLanes < 32 ? (kLanes & (kLanes - 1)) == 0 : kLanes % 32 == 0);
  static_assert(kThreads % kLanes == 0);
  static_assert(kThreads <= 1024);

  float maximum = lane_absmax<kValuesPerLane>(values);

  if constexpr (kLanes < 32) {
    PRAGMA_UNROLL
    for (uint32_t step = 1; step < kLanes; step <<= 1)
      maximum = fmaxf(maximum, __shfl_xor_sync(0xFFFFFFFFu, maximum, step, kLanes));
  } else {
    maximum = warp_max(maximum);
  }

  if constexpr (kLanes > 32) {
    constexpr uint32_t kWarpsPerGroup = kLanes / 32;
    static_assert(kGroups <= 16);

    uint32_t thread = threadIdx.x - kWarpOffset * 32;
    uint32_t warp = thread / 32;
    uint32_t lane = thread & 31;
    uint32_t group = thread / kLanes;
    uint32_t group_warp = group * kWarpsPerGroup;
    uint32_t warp_in_group = warp - group_warp;

    if (lane == 0) shared[warp] = maximum;
    if constexpr (kGroups == 1) {
      asm volatile("bar.sync 0, %0;" : : "r"(kLanes) : "memory");
    } else {
      asm volatile("bar.sync %0, %1;" : : "r"(group), "r"(kLanes) : "memory");
    }

    if constexpr (kWarpsPerGroup < 32) {
// SM80/89 favor uniform shared-memory broadcasts; SM100+ favors distributed loads followed by REDUX.
#if __CUDA_ARCH__ >= 1000
      maximum = shared[group_warp + lane % kWarpsPerGroup];
      maximum = warp_max(maximum);
#else
      PRAGMA_UNROLL
      for (uint32_t warp = 0; warp < kWarpsPerGroup; warp++)
        maximum = fmaxf(maximum, shared[group_warp + warp]);
#endif
    } else {
      if (warp_in_group == 0) {
        maximum = lane < kWarpsPerGroup ? shared[group_warp + lane] : 0.f;
        maximum = warp_max(maximum);
        if (lane == 0) shared[group_warp] = maximum;
      }

      asm volatile("bar.sync %0, %1;" : : "r"(group), "r"(kLanes) : "memory");
      maximum = shared[group_warp];
    }
  }

  return maximum;
}


template <
    uint32_t kGroupSize,
    uint32_t kValuesPerLane,
    uint32_t kThreadsPerToken,
    uint32_t kNumWarps,
    uint32_t kWarpOffset>
CUDA_INLINE float token_group_scale_max(float group_scale, float *shared) {
  constexpr uint32_t kGroupLanes = kGroupSize / kValuesPerLane;
  constexpr uint32_t kWarpsPerToken = kThreadsPerToken / 32;
  static_assert(kGroupSize % kValuesPerLane == 0);
  static_assert(kGroupLanes >= 1 && (kGroupLanes & (kGroupLanes - 1)) == 0);
  static_assert(kThreadsPerToken >= 32 && kThreadsPerToken % 32 == 0);
  static_assert(kWarpsPerToken >= 1 && (kWarpsPerToken & (kWarpsPerToken - 1)) == 0);
  static_assert(kNumWarps % kWarpsPerToken == 0);

  if constexpr (kGroupLanes < 32) {
#if __CUDA_ARCH__ >= 800
    group_scale = warp_max(group_scale);
#else
    PRAGMA_UNROLL
    for (uint32_t step = kGroupLanes; step < 32; step <<= 1)
      group_scale = fmaxf(group_scale, __shfl_xor_sync(0xFFFFFFFFu, group_scale, step));
#endif
  }

  if constexpr (kWarpsPerToken > 1) {
    uint32_t thread = threadIdx.x - kWarpOffset * 32;
    uint32_t warp = thread / 32;
    uint32_t lane = thread & 31;
    uint32_t token = thread / kThreadsPerToken;
    uint32_t first_warp = token * kWarpsPerToken;
    if (lane == 0) shared[warp] = group_scale;
    asm volatile("bar.sync %0, %1;" : : "r"(token), "r"(kThreadsPerToken) : "memory");
    group_scale = shared[first_warp + lane % kWarpsPerToken];
#if __CUDA_ARCH__ >= 800
    group_scale = warp_max(group_scale);
#else
    PRAGMA_UNROLL
    for (uint32_t step = 1; step < kWarpsPerToken; step <<= 1)
      group_scale = fmaxf(group_scale, __shfl_xor_sync(0xFFFFFFFFu, group_scale, step, kWarpsPerToken));
#endif
  }
  return group_scale;
}


template <class TargetType>
__host__ __device__ constexpr float target_maximum() {
  if constexpr (std::is_same<TargetType, Float8E3M4>::value)
    return 30.f;
  else if constexpr (std::is_same<TargetType, Float8E4M3>::value)
    return 448.f;
  else if constexpr (std::is_same<TargetType, Float8E5M2>::value)
    return 57344.f;
  else if constexpr (std::is_same<TargetType, Float4E0M3>::value)
    return 7.f;
  else if constexpr (std::is_same<TargetType, Float4E2M1>::value)
    return 6.f;
  else {
    static_assert(TargetType::kIsIntegerType && TargetType::kIsSigned);
    static_assert(TargetType::kBits == 4 || TargetType::kBits == 8);
    return static_cast<float>((1 << (TargetType::kBits - 1)) - 1);
  }
}


template <class TargetType, uint32_t kValues>
CUDA_INLINE void pack_float(const float *values, uint8_t *packed) {
  constexpr bool kFp8 = std::is_same<TargetType, Float8E3M4>::value || std::is_same<TargetType, Float8E4M3>::value || std::is_same<TargetType, Float8E5M2>::value;
  constexpr bool kFp4 = std::is_same<TargetType, Float4E0M3>::value || std::is_same<TargetType, Float4E2M1>::value;
  static_assert(kFp8 || kFp4);
  static_assert(kValues % 2 == 0);

  if constexpr (kFp8) {
#if __CUDA_ARCH__ >= 890
    if constexpr (kValues % 4 == 0) {
      PRAGMA_UNROLL
      for (uint32_t value = 0; value < kValues; value += 4) {
        uint32_t output;
        if constexpr (std::is_same<TargetType, Float8E4M3>::value)
          asm("{ .reg .b16 p0, p1;"
              "cvt.rn.satfinite.e4m3x2.f32 p0, %2, %1;"
              "cvt.rn.satfinite.e4m3x2.f32 p1, %4, %3;"
              "mov.b32 %0, {p0, p1}; }"
              : "=r"(output)
              : "f"(values[value + 0]), "f"(values[value + 1]),
                "f"(values[value + 2]), "f"(values[value + 3]));
        else
          asm("{ .reg .b16 p0, p1;"
              "cvt.rn.satfinite.e5m2x2.f32 p0, %2, %1;"
              "cvt.rn.satfinite.e5m2x2.f32 p1, %4, %3;"
              "mov.b32 %0, {p0, p1}; }"
              : "=r"(output)
              : "f"(values[value + 0]), "f"(values[value + 1]),
                "f"(values[value + 2]), "f"(values[value + 3]));
        *reinterpret_cast<uint32_t *>(packed + value) = output;
      }
    } else {
      uint16_t output;
      if constexpr (std::is_same<TargetType, Float8E4M3>::value)
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;"
            : "=h"(output)
            : "f"(values[0]), "f"(values[1]));
      else
        asm("cvt.rn.satfinite.e5m2x2.f32 %0, %2, %1;"
            : "=h"(output)
            : "f"(values[0]), "f"(values[1]));
      *reinterpret_cast<uint16_t *>(packed) = output;
    }
#else
    asm("trap;");
#endif
  } else {
#if __CUDA_ARCH__ >= 1000
    if constexpr (kValues % 8 == 0) {
      PRAGMA_UNROLL
      for (uint32_t value = 0; value < kValues; value += 8) {
        uint32_t output;
        asm("{ .reg .b8 p0, p1, p2, p3;"
            "cvt.rn.satfinite.e2m1x2.f32 p0, %2, %1;"
            "cvt.rn.satfinite.e2m1x2.f32 p1, %4, %3;"
            "cvt.rn.satfinite.e2m1x2.f32 p2, %6, %5;"
            "cvt.rn.satfinite.e2m1x2.f32 p3, %8, %7;"
            "mov.b32 %0, {p0, p1, p2, p3}; }"
            : "=r"(output)
            : "f"(values[value + 0]), "f"(values[value + 1]),
              "f"(values[value + 2]), "f"(values[value + 3]),
              "f"(values[value + 4]), "f"(values[value + 5]),
              "f"(values[value + 6]), "f"(values[value + 7]));
        *reinterpret_cast<uint32_t *>(packed + value / 2) = output;
      }
    } else if constexpr (kValues % 4 == 0) {
      PRAGMA_UNROLL
      for (uint32_t value = 0; value < kValues; value += 4) {
        uint16_t output;
        asm("{ .reg .b8 p0, p1;"
            "cvt.rn.satfinite.e2m1x2.f32 p0, %2, %1;"
            "cvt.rn.satfinite.e2m1x2.f32 p1, %4, %3;"
            "mov.b16 %0, {p0, p1}; }"
            : "=h"(output)
            : "f"(values[value + 0]), "f"(values[value + 1]),
              "f"(values[value + 2]), "f"(values[value + 3]));
        *reinterpret_cast<uint16_t *>(packed + value / 2) = output;
      }
    } else {
      uint16_t output;
      asm("{ .reg .b8 p; cvt.rn.satfinite.e2m1x2.f32 p, %2, %1;"
          "cvt.u16.u8 %0, p; }"
          : "=h"(output)
          : "f"(values[0]), "f"(values[1]));
      packed[0] = output;
    }
#else
    asm("trap;");
#endif
  }
}


template <uint32_t kValues>
CUDA_INLINE void pack_int8(const float *values, uint8_t *packed) {
  static_assert(kValues >= 1 && (kValues & (kValues - 1)) == 0);

  if constexpr (kValues % 4 == 0) {
    PRAGMA_UNROLL
    for (uint32_t value = 0; value < kValues; value += 4) {
      uint32_t output;
#if __CUDA_ARCH__ >= 800
      asm("{ .reg .s32 p0, p1, p2, p3; .reg .b32 q;"
          "cvt.rni.s32.f32 p0, %1; cvt.rni.s32.f32 p1, %2;"
          "cvt.rni.s32.f32 p2, %3; cvt.rni.s32.f32 p3, %4;"
          "cvt.pack.sat.s8.s32.b32 q, p3, p2, 0;"
          "cvt.pack.sat.s8.s32.b32 %0, p1, p0, q; }"
          : "=r"(output)
          : "f"(values[value + 0]), "f"(values[value + 1]),
            "f"(values[value + 2]), "f"(values[value + 3]));
#else
      asm("{ .reg .b32 p0, p1, p2, p3, q0, q1;"
          "cvt.rni.s8.f32 p0, %1; cvt.rni.s8.f32 p1, %2;"
          "cvt.rni.s8.f32 p2, %3; cvt.rni.s8.f32 p3, %4;"
          "prmt.b32 q0, p0, p1, 0x4040; prmt.b32 q1, p2, p3, 0x4040;"
          "prmt.b32 %0, q0, q1, 0x5410; }"
          : "=r"(output)
          : "f"(values[value + 0]), "f"(values[value + 1]),
            "f"(values[value + 2]), "f"(values[value + 3]));
#endif
      *reinterpret_cast<uint32_t *>(packed + value) = output;
    }
  } else if constexpr (kValues == 2) {
    uint32_t output;
    asm("{ .reg .b32 p0, p1; cvt.rni.s8.f32 p0, %1; cvt.rni.s8.f32 p1, %2;"
        "prmt.b32 %0, p0, p1, 0x4040; }"
        : "=r"(output)
        : "f"(values[0]), "f"(values[1]));
    *reinterpret_cast<uint16_t *>(packed) = static_cast<uint16_t>(output);
  } else {
    int32_t output;
    asm("cvt.rni.s8.f32 %0, %1;" : "=r"(output) : "f"(values[0]));
    packed[0] = output;
  }
}


template <uint32_t kValues>
CUDA_INLINE void pack_int4(const float *values, uint8_t *packed) {
  static_assert(kValues >= 2 && (kValues & (kValues - 1)) == 0);
  constexpr uint32_t kSelectNibbles = (0xF0 & 0xAA) | (0xCC & ~0xAA);

  if constexpr (kValues % 8 == 0) {
    PRAGMA_UNROLL
    for (uint32_t value = 0; value < kValues; value += 8) {
      float even[4] = {values[value], values[value + 2], values[value + 4], values[value + 6]};
      float odd[4] = {values[value + 1], values[value + 3], values[value + 5], values[value + 7]};
      uint32_t even_bytes, odd_bytes;
      pack_int8<4>(even, reinterpret_cast<uint8_t *>(&even_bytes));
      pack_int8<4>(odd, reinterpret_cast<uint8_t *>(&odd_bytes));
      *reinterpret_cast<uint32_t *>(packed + value / 2) = lop3<kSelectNibbles>(even_bytes, odd_bytes << 4, 0x0F0F0F0Fu);
    }
  } else if constexpr (kValues % 4 == 0) {
    PRAGMA_UNROLL
    for (uint32_t value = 0; value < kValues; value += 4) {
      uint32_t output;
      asm("{ .reg .b32 e0, e1, o0, o1, even, odd;"
          "cvt.rni.s8.f32 e0, %1; cvt.rni.s8.f32 e1, %3;"
          "cvt.rni.s8.f32 o0, %2; cvt.rni.s8.f32 o1, %4;"
          "prmt.b32 even, e0, e1, 0x4040; prmt.b32 odd, o0, o1, 0x4040;"
          "shl.b32 odd, odd, 4; lop3.b32 %0, even, odd, 0x0F0F, 0xE4; }"
          : "=r"(output)
          : "f"(values[value + 0]), "f"(values[value + 1]),
            "f"(values[value + 2]), "f"(values[value + 3]));
      *reinterpret_cast<uint16_t *>(packed + value / 2) = static_cast<uint16_t>(output);
    }
  } else {
    uint32_t output;
    asm("{ .reg .b32 even, odd;"
        "cvt.rni.s8.f32 even, %1; cvt.rni.s8.f32 odd, %2;"
        "shl.b32 odd, odd, 4; lop3.b32 %0, even, odd, 0x0F, 0xE4; }"
        : "=r"(output)
        : "f"(values[0]), "f"(values[1]));
    packed[0] = static_cast<uint8_t>(output);
  }
}


template <class TargetType, class ScaleType, uint32_t kValuesPerLane>
struct QuantGroupResult {
  static_assert(kValuesPerLane * TargetType::kBits % 8 == 0);
  static constexpr uint32_t kPackedBytes = kValuesPerLane * TargetType::kBits / 8;

  alignas(4) uint8_t packed[kPackedBytes];
  ScaleStorage<ScaleType> scale;
};


template <
    class TargetType,
    class ScaleType,
    uint32_t kValuesPerLane,
    uint32_t kScaleSize,
    uint32_t kNumWarps,
    uint32_t kWarpOffset,
    bool kStaticScale,
    ScaleMode kScaleMode,
    QuantizationPhase kPhase = QuantizationPhase::Fused>
CUDA_INLINE QuantGroupResult<TargetType, ScaleType, kValuesPerLane> quant_group(
    float *values,
    float *shared,
    float static_scale,
    ScaleStorage<ScaleType> collected_scale = {}) {
  static_assert(supported_scale_type<ScaleType>);
  static_assert(kScaleMode == ScaleMode::DynamicGroup || std::is_same<ScaleType, Float32>::value);
  static_assert(kPhase == QuantizationPhase::Fused || kScaleMode == ScaleMode::DynamicToken);

  QuantGroupResult<TargetType, ScaleType, kValuesPerLane> result;
  if constexpr (kStaticScale && kScaleMode != ScaleMode::DynamicGroup) {
    float static_multiplier = 1.f / static_scale;
    PRAGMA_UNROLL
    for (uint32_t value = 0; value < kValuesPerLane; value++)
      values[value] *= static_multiplier;
  }

  if constexpr (kPhase == QuantizationPhase::Quantize) {
    result.scale = collected_scale;
  } else if constexpr (kScaleMode != ScaleMode::Static) {
    float maximum = group_absmax<kValuesPerLane, kScaleSize, kNumWarps, kWarpOffset>(values, shared);
    float raw_scale = fmaxf(maximum / target_maximum<TargetType>(), 1e-30f);
    if constexpr (kStaticScale && kScaleMode == ScaleMode::DynamicGroup)
      raw_scale /= static_scale;
    result.scale = encode_scale<ScaleType>(raw_scale);
  }

  if constexpr (kScaleMode != ScaleMode::Static && kPhase != QuantizationPhase::CollectAbsmax) {
    float scale = decode_scale<ScaleType>(result.scale);
    if constexpr (kStaticScale && kScaleMode == ScaleMode::DynamicGroup)
      scale *= static_scale;
    float dynamic_multiplier;
    if constexpr (
        std::is_same<ScaleType, Float32>::value ||
        std::is_same<ScaleType, Float8E8M0>::value ||
        std::is_same<ScaleType, M3BFloat16>::value)
      dynamic_multiplier = 1.f / scale;
    else
      dynamic_multiplier = scale > 0.f ? 1.f / scale : 0.f;
    PRAGMA_UNROLL
    for (uint32_t value = 0; value < kValuesPerLane; value++)
      values[value] *= dynamic_multiplier;
  }

  if constexpr (kPhase != QuantizationPhase::CollectAbsmax) {
    if constexpr (TargetType::kIsFloatingPointType) {
      pack_float<TargetType, kValuesPerLane>(values, result.packed);
    } else if constexpr (TargetType::kBits == 8) {
      pack_int8<kValuesPerLane>(values, result.packed);
    } else {
      static_assert(TargetType::kIsIntegerType && TargetType::kIsSigned && TargetType::kBits == 4);
      pack_int4<kValuesPerLane>(values, result.packed);
    }
  }

  return result;
}
