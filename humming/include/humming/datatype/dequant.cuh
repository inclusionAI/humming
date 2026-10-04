#pragma once

#include <humming/datatype/dequant_fused.cuh>
#include <humming/datatype/dequant_native.cuh>
#include <humming/datatype/dequant_prepare.cuh>
#include <humming/datatype/dequant_single.cuh>
#include <humming/utils/all.cuh>


template <class SourceType, class TargetType, bool kHasZeroPoint, bool kIsFpZeroPoint>
CUDA_INLINE void dequant_b1248(const uint32_t *qb, uint32_t *res, uint32_t j, uint32_t *zp_vals = nullptr) {
  static_assert(SourceType::kBits <= TargetType::kBits);
  constexpr uint32_t kResultPackedNums = 32 / TargetType::kBits;
  constexpr uint32_t kResultPackedNumSourceBits = kResultPackedNums * SourceType::kBits;
  constexpr bool kIsFpToFp = SourceType::kIsFloatingPointType && TargetType::kIsFloatingPointType;
  constexpr uint32_t reverse_pattern = TargetType::kBits / SourceType::kBits;

  PRAGMA_UNROLL
  for (uint32_t i = 0; i < 4; i++) {
    uint32_t index = j * 4 + i;
    uint32_t zp_val = kHasZeroPoint ? zp_vals[i] : 0;
    uint32_t qb_index = kResultPackedNumSourceBits * index / 32;

    uint32_t shift_count;
    if constexpr (SourceType::kBits * 4 % TargetType::kBits == 0) {
      shift_count = SourceType::kBits * i % TargetType::kBits;
    } else {
      shift_count = SourceType::kBits * index % TargetType::kBits;
    }

    uint32_t qb_val = qb[qb_index];
    if (kIsFpToFp && shift_count) qb_val = qb_val << shift_count;
    if (!kIsFpToFp && shift_count) qb_val = qb_val >> shift_count;
    if constexpr (!kIsFpToFp) {
      res[i] = dequant_single<SourceType, TargetType, kHasZeroPoint, kIsFpZeroPoint>(qb_val, zp_val);
    } else {
      uint32_t reversed_index = i / reverse_pattern * reverse_pattern + reverse_pattern - 1 - i % reverse_pattern;
      res[reversed_index] = dequant_single<SourceType, TargetType, kHasZeroPoint, kIsFpZeroPoint>(qb_val, zp_val);
    }
  }
}


template <class SourceType>
CUDA_INLINE void repack_native_mxf8f6f4(const uint32_t *qb, uint32_t *res, uint32_t j) {
  if constexpr (std::is_same<SourceType, Float4E2M1>::value) {
    res[0] = (qb[2 * j] & 0x0F0F0F0Fu) << 2;
    res[1] = (qb[2 * j] & 0xF0F0F0F0u) >> 2;
    res[2] = (qb[2 * j + 1] & 0x0F0F0F0Fu) << 2;
    res[3] = (qb[2 * j + 1] & 0xF0F0F0F0u) >> 2;
  } else {
    static_assert(std::is_same<SourceType, Float6E3M2>::value || std::is_same<SourceType, Float6E2M3>::value);
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < 4; i++) {
      uint32_t index = j * 4 + i;
      res[i] = get_quanted_value_group<6, true>(qb + index / 8 * 6, index % 8);
    }
  }
}


template <class SourceType, class TargetType, bool kHasZeroPoint, bool kIsFpZeroPoint, uint32_t kNumWarpShapeNSplits = 1>
CUDA_INLINE void dequant_b3567(const uint32_t *qb, uint32_t *res, uint32_t j, uint32_t *zp_vals = nullptr) {
  static_assert(SourceType::kBits <= TargetType::kBits);
  constexpr uint32_t kPaddedNumBits = static_next_power_of_2(SourceType::kBits);
  constexpr bool kIsFpToFp = SourceType::kIsFloatingPointType && TargetType::kIsFloatingPointType;
  constexpr uint32_t reverse_pattern = TargetType::kBits / static_next_power_of_2(SourceType::kBits);
  uint32_t qb_val;

  PRAGMA_UNROLL
  for (uint32_t i = 0; i < 4; i++) {
    uint32_t zp_val = kHasZeroPoint ? zp_vals[i] : 0;
    uint32_t index;

    if constexpr (kNumWarpShapeNSplits == 1 || SourceType::kBits == 6) {
      index = j * 4 + i;
      if (index * kPaddedNumBits % TargetType::kBits == 0) {
        const uint32_t idx1 = index / TargetType::kBits;
        const uint32_t idx2 = index * kPaddedNumBits / TargetType::kBits % kPaddedNumBits;
        const uint32_t qb_offset = idx1 * SourceType::kBits;
        qb_val = get_quanted_value_group<SourceType::kNumBits, !kIsFpToFp>(qb + qb_offset, idx2);
      }
    } else if (threadIdx.x / 32 % 2 == 0) {
      index = j * 4 + i;
      if (index * kPaddedNumBits % TargetType::kBits == 0) {
        const uint32_t idx1 = index / TargetType::kBits;
        const uint32_t idx2 = index * kPaddedNumBits / TargetType::kBits % kPaddedNumBits;
        const uint32_t qb_offset = idx1 * SourceType::kBits;
        qb_val = get_quanted_value_group<SourceType::kNumBits, !kIsFpToFp>(qb + qb_offset, idx2);
      }
    } else {
      index = (j + TargetType::kBits / 8) * 4 + i;
      if (index * kPaddedNumBits % TargetType::kBits == 0) {
        const uint32_t idx1 = index / TargetType::kBits;
        const uint32_t idx2 = index * kPaddedNumBits / TargetType::kBits % kPaddedNumBits;
        const uint32_t qb_offset = idx1 * SourceType::kBits;
        qb_val = get_quanted_value_group<SourceType::kNumBits, !kIsFpToFp>(qb + qb_offset, idx2);
      }
    }

    uint32_t shift_count;
    if constexpr (kPaddedNumBits * 4 % TargetType::kBits == 0) {
      shift_count = kPaddedNumBits * i % TargetType::kBits;
    } else {
      shift_count = kPaddedNumBits * index % TargetType::kBits;
    }

    uint32_t qb_val2 = qb_val;
    if (kIsFpToFp && shift_count) qb_val2 = qb_val << shift_count;
    if (!kIsFpToFp && shift_count) qb_val2 = qb_val >> shift_count;

    if constexpr (!kIsFpToFp) {
      res[i] = dequant_single<SourceType, TargetType, kHasZeroPoint, kIsFpZeroPoint>(qb_val2, zp_val);
    } else {
      uint32_t reversed_index = i / reverse_pattern * reverse_pattern + reverse_pattern - 1 - i % reverse_pattern;
      res[reversed_index] = dequant_single<SourceType, TargetType, kHasZeroPoint, kIsFpZeroPoint>(qb_val2, zp_val);
    }
  }
}


template <class SourceType, class TargetType, bool kHasZeroPoint = false, bool kIsFpZeroPoint = false, uint32_t kNumWarpShapeNSplits = 1>
CUDA_INLINE void dequant(const uint32_t *qb, uint32_t *res, uint32_t j, uint32_t *zp_vals = nullptr) {
  if constexpr (SourceType::kBits == static_next_power_of_2(SourceType::kBits)) {
    dequant_b1248<SourceType, TargetType, kHasZeroPoint, kIsFpZeroPoint>(qb, res, j, zp_vals);
  } else {
    dequant_b3567<SourceType, TargetType, kHasZeroPoint, kIsFpZeroPoint, kNumWarpShapeNSplits>(qb, res, j, zp_vals);
  }
}


// Scale bytes have a canonical order independent of the weight dequantization layout.
template <class SourceType, class TargetType>
constexpr bool kNativeScaleDequantSupported =
    kNativeDequantSupported<SourceType, TargetType> ||
    (std::is_same<TargetType, BFloat16>::value && kNativeDequantSupported<SourceType, Float16>);


template <class SourceType, class TargetType>
CUDA_INLINE void dequant_scale(const uint32_t *src, uint32_t *dst, uint32_t index) {
  if constexpr (kNativeDequantSupported<SourceType, TargetType>) {
    dequant_native<SourceType, TargetType>(src, dst, index);
  } else if constexpr (kNativeScaleDequantSupported<SourceType, TargetType>) {
    uint32_t packed[4];
    dequant_native<SourceType, Float16>(src, packed, index);
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < 4; i++) {
      float2 values = __half22float2(*reinterpret_cast<half2 *>(&packed[i]));
      nv_bfloat162 result = __float22bfloat162_rn(values);
      dst[i] = *reinterpret_cast<uint32_t *>(&result);
    }
  } else {
    dequant<SourceType, TargetType>(src, dst, index);
    using scalar_t = typename F16Conversion<TargetType>::scalar_t;
    scalar_t *values = reinterpret_cast<scalar_t *>(dst);
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < 2; i++) {
      scalar_t tmp = values[4 * i + 1];
      values[4 * i + 1] = values[4 * i + 2];
      values[4 * i + 2] = tmp;
    }
    // The software weight path leaves exponent fields biased. Preserve special
    // scale encodings before callers apply the normal exponent compensation.
    if constexpr (std::is_same<SourceType, Float8E4M3>::value || std::is_same<SourceType, Float8E5M2>::value) {
      constexpr uint16_t kInfinity = ((1u << TargetType::kExponentBits) - 1) << TargetType::kMantissaBits;
      uint16_t *bits = reinterpret_cast<uint16_t *>(dst);
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < 8; i++) {
        uint32_t byte = (src[index * 2 + i / 4] >> (8 * (i % 4))) & 0xFFu;
        if constexpr (std::is_same<SourceType, Float8E4M3>::value) {
          if ((byte & 0x7Fu) == 0x7Fu) bits[i] = kInfinity | 1u;
        } else {
          constexpr uint32_t kMantissaMask = (1u << SourceType::kMantissaBits) - 1;
          if (((byte >> SourceType::kMantissaBits) & 31u) == 31u) {
            uint16_t sign = SourceType::kIsSigned ? (byte & 0x80u) << 8 : 0;
            bits[i] = sign | kInfinity | (byte & kMantissaMask);
          }
        }
      }
    }
  }
}


template <class SourceType>
CUDA_INLINE float4 dequant_scale_float4(uint32_t packed) {
  float4 result;
  float *values = reinterpret_cast<float *>(&result);
  if constexpr (kNativeDequantSupported<SourceType, Float16>) {
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < 2; i++) {
      uint32_t converted = dequant_native_x2<SourceType, Float16>(packed >> (16 * i));
      reinterpret_cast<float2 *>(values)[i] = __half22float2(*reinterpret_cast<half2 *>(&converted));
    }
  } else {
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < 4; i++) {
      uint32_t byte = (packed >> (8 * i)) & 0xFFu;
      uint32_t bits = dequant_single<SourceType, Float32, false, false>(byte << 24, 0);
      if constexpr (std::is_same<SourceType, Float8E8M0>::value) {
        // E8M0 has no zero/subnormal encoding; byte zero represents 2^-127.
        values[i] = byte == 0 ? 0x1p-127f : *reinterpret_cast<float *>(&bits);
        if (byte == 255) values[i] = __int_as_float(0x7FC00000);
      } else {
        constexpr uint32_t kExponentMask = (1u << SourceType::kExponentBits) - 1;
        constexpr uint32_t kMantissaMask = (1u << SourceType::kMantissaBits) - 1;
        constexpr int32_t kBias = (1 << (SourceType::kExponentBits - 1)) - 1;
        uint32_t exponent = (byte >> SourceType::kMantissaBits) & kExponentMask;
        if (exponent == 0) {
          // Normalize source subnormals without multiplying an FP32 subnormal under FTZ.
          uint32_t factor_bits = (127 + 1 - kBias - SourceType::kMantissaBits) << 23;
          float value = float(byte & kMantissaMask) * __uint_as_float(factor_bits);
          values[i] = SourceType::kIsSigned && (byte & 0x80u) ? -value : value;
        } else {
          bits += (127 - kBias) << 23;
          values[i] = *reinterpret_cast<float *>(&bits);
        }
        if constexpr (std::is_same<SourceType, Float8E5M2>::value) {
          if (exponent == 31) {
            uint32_t mantissa = (byte & kMantissaMask) << (23 - SourceType::kMantissaBits);
            values[i] = __uint_as_float((bits & 0x80000000u) | 0x7F800000u | mantissa);
          }
        } else if constexpr (std::is_same<SourceType, Float8E4M3>::value) {
          if ((byte & 0x7Fu) == 0x7Fu) values[i] = __int_as_float(0x7FC00000);
        }
      }
    }
  }
  return result;
}
