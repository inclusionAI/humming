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


template <class SourceType, class TargetType>
constexpr bool kNativeScaleDequantSupported =
    kNativeDequantSupported<SourceType, TargetType> ||
    (std::is_same<TargetType, BFloat16>::value && kNativeDequantSupported<SourceType, Float16>);


// Software conversion may retain (0, 2, 1, 3) order for callers that can absorb the permutation.
template <class SourceType, class TargetType, bool kKeepDequantOrder = false>
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
    if constexpr (!kKeepDequantOrder) {
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < 2; i++) {
        uint32_t even = dst[2 * i];
        uint32_t odd = dst[2 * i + 1];
        dst[2 * i] = __byte_perm(even, odd, 0x5410);
        dst[2 * i + 1] = __byte_perm(even, odd, 0x7632);
      }
    }
  }
}


template <class SourceType>
CUDA_INLINE void dequant_scale_float32(const uint32_t *src, uint32_t *dst, uint32_t index) {
  uint32_t pairs[4];
  if constexpr (kNativeDequantSupported<SourceType, Float16>) {
    dequant_native<SourceType, Float16>(src, pairs, index);
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < 4; i++) {
      reinterpret_cast<float2 *>(dst)[i] = __half22float2(*reinterpret_cast<half2 *>(&pairs[i]));
    }
  } else {
    // Expand eight scale bytes into four packed BF16 pairs before widening to FP32.
    dequant_scale<SourceType, BFloat16, true>(src, pairs, index);
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < 4; i++) {
      nv_bfloat162 values = *reinterpret_cast<nv_bfloat162 *>(&pairs[i]);
      constexpr uint32_t kExponentOffset = 128 - (1u << (SourceType::kExponentBits - 1));
      if constexpr (kExponentOffset != 0) {
        constexpr uint32_t kFactorBits = ((127 + kExponentOffset) * 0x00010001u) << 7;
        const nv_bfloat162 factor = *reinterpret_cast<const nv_bfloat162 *>(&kFactorBits);
        values = __hmul2(values, factor);
      }
      float2 converted = __bfloat1622float2(values);
      if constexpr (kNativeScaleDequantSupported<SourceType, BFloat16>) {
        reinterpret_cast<float2 *>(dst)[i] = converted;
      } else {
        // Restore byte order through register indices instead of packed permutations.
        constexpr uint32_t kValuesPerGroup = 4;
        uint32_t first = i / 2 * kValuesPerGroup + i % 2;
        reinterpret_cast<float *>(dst)[first] = converted.x;
        reinterpret_cast<float *>(dst)[first + 2] = converted.y;
      }
    }
  }
}
