#pragma once

#include <humming/datatype/base_conversion.cuh>
#include <humming/datatype/dtypes.cuh>
#include <humming/utils/all.cuh>


template <class TargetType>
CUDA_INLINE uint2 fused_dequant_single_for_mxfp4(const uint32_t qb, const uint32_t exp_offset) {
  static_assert(std::is_same<TargetType, Float8E4M3>::value || std::is_same<TargetType, Int8>::value);
  return {0, 0};
}


template <>
CUDA_INLINE uint2 fused_dequant_single_for_mxfp4<Float8E4M3>(const uint32_t qb, const uint32_t exp_offset) {
  uint32_t exp_offset_buffer1 = (exp_offset * 0x08080800) + ((0x03020100 << 2) - 0x00000400);
  uint32_t exp_offset_buffer2 = (exp_offset * 0x08080808) + (0x07060504 << 2);

  uint32_t exp_offsets[2] = {
      __byte_perm(exp_offset_buffer1, exp_offset_buffer2, qb),
      __byte_perm(exp_offset_buffer1, exp_offset_buffer2, qb >> 16)};

  uint32_t res[2] = {
      lop3_and_or(qb << 4, 0x80808080, exp_offsets[0]),
      lop3_and_or(qb, 0x80808080, exp_offsets[1])};

  return *reinterpret_cast<uint2 *>(res);
}


template <uint32_t kStoredExpBias>
CUDA_INLINE uint2 fused_dequant_single_for_mxfp4_e4m3(const uint32_t qb, const uint32_t stored_exp) {
  // Keep the legacy fused path instruction shape while allowing a canonical
  // E8M0 byte (relative exponent + 127) to share storage with the explicit
  // path.  The bias is folded into the affine LUT constants at compile time.
  constexpr uint32_t kMul1 = 0x08080800;
  constexpr uint32_t kMul2 = 0x08080808;
  constexpr uint32_t kAdd1 = (0x03020100 << 2) - 0x00000400 - kStoredExpBias * kMul1;
  constexpr uint32_t kAdd2 = (0x07060504 << 2) - kStoredExpBias * kMul2;
  uint32_t exp_offset_buffer1 = stored_exp * kMul1 + kAdd1;
  uint32_t exp_offset_buffer2 = stored_exp * kMul2 + kAdd2;

  uint32_t exp_offsets[2] = {
      __byte_perm(exp_offset_buffer1, exp_offset_buffer2, qb),
      __byte_perm(exp_offset_buffer1, exp_offset_buffer2, qb >> 16)};

  uint32_t res[2] = {
      lop3_and_or(qb << 4, 0x80808080, exp_offsets[0]),
      lop3_and_or(qb, 0x80808080, exp_offsets[1])};

  return *reinterpret_cast<uint2 *>(res);
}


template <>
CUDA_INLINE uint2 fused_dequant_single_for_mxfp4<Int8>(const uint32_t qb, const uint32_t exp_offset) {
  uint32_t buffer1 = 0x03020100 << exp_offset;
  uint32_t buffer2 = 0x0C080604 << exp_offset;

  uint32_t res[2];
  uint32_t signs[2] = {qb >> 3, qb >> 7};
  uint32_t int8s[2] = {
      __byte_perm(buffer1, buffer2, qb),
      __byte_perm(buffer1, buffer2, qb >> 16)};

  PRAGMA_UNROLL
  for (uint32_t i = 0; i < 2; i++) {
    uint32_t val = __byte_perm(int8s[0], int8s[1], 0x6420 + 0x1111 * i);
    uint32_t flag = signs[i] & 0x01010101;
    uint32_t mask = flag * 0xFF;
    res[i] = (val - flag) ^ mask;
  }

  return *reinterpret_cast<uint2 *>(res);
}

CUDA_INLINE uint32_t mxfp4_fused_to_group_interleave(uint32_t qb) {
  // Mode 2 keeps magnitudes in [0,1,2,3,4,5,6,7] order while signs are
  // already in WGMMA order.  Mode 3 needs magnitudes in
  // [0,4,1,5,2,6,3,7].  Transpose the two nibble rows with two PRMTs and
  // preserve the already-correct sign bitplane.
  constexpr uint32_t kMagnitudeMask = 0x07070707;
  uint32_t even_magnitudes = qb & kMagnitudeMask;
  uint32_t odd_magnitudes = (qb >> 4) & kMagnitudeMask;
  uint32_t lower = __byte_perm(even_magnitudes, odd_magnitudes, 0x5140);
  uint32_t upper = __byte_perm(even_magnitudes, odd_magnitudes, 0x7362);
  return lower | (upper << 4) | (qb & 0x88888888);
}


template <uint32_t kCount>
CUDA_INLINE void swap_mxfp4_wgmma_register_words(uint32_t *values) {
  PRAGMA_UNROLL
  for (uint32_t index = 0; index < kCount; index++) {
    uint32_t tmp = values[index * 4 + 1];
    values[index * 4 + 1] = values[index * 4 + 2];
    values[index * 4 + 2] = tmp;
  }
}


template <
    class TargetType,
    uint32_t kCount,
    bool kUseWgmma,
    uint32_t kStoredExpBias = 0>
CUDA_INLINE void fused_dequant_for_mxfp4(const uint32_t *qb_ptrs, uint32_t *res_ptrs, uint32_t *scales_ptr) {
  PRAGMA_UNROLL
  for (uint32_t i = 0; i < kCount * 2; i++) {
    uint32_t exp_offset = reinterpret_cast<uint8_t *>(scales_ptr)[i];
    uint32_t qb = qb_ptrs[i];
    uint2 res;
    if constexpr (std::is_same<TargetType, Float8E4M3>::value) {
      res = fused_dequant_single_for_mxfp4_e4m3<kStoredExpBias>(qb, exp_offset);
    } else {
      static_assert(kStoredExpBias == 0, "rebased E8M0 storage only supports FP8 E4M3");
      res = fused_dequant_single_for_mxfp4<TargetType>(qb, exp_offset);
    }
    res_ptrs[i * 2] = res.x;
    res_ptrs[i * 2 + 1] = res.y;
  }

  if constexpr (kUseWgmma) {
    swap_mxfp4_wgmma_register_words<kCount>(res_ptrs);
  }
}
