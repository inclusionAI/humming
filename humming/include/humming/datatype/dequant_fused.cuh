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


template <uint32_t kStoredExpBias>
CUDA_INLINE void fused_dequant_group_interleaved_mxfp4_e4m3(
    const uint32_t *qb,
    uint32_t *res,
    const uint32_t stored_exp0,
    const uint32_t stored_exp1) {
  // Group-friendly mode-3 storage interleaves two WGMMA rows in each packed
  // FP4 word.  The low nibbles map to registers 0/2 and use scale 0, while
  // the high nibbles map to registers 1/3 and use scale 1.  Build one LUT per
  // row and emit the four WGMMA A-fragment registers directly.
  constexpr uint32_t kMul1 = 0x08080800;
  constexpr uint32_t kMul2 = 0x08080808;
  constexpr uint32_t kAdd1 = (0x03020100 << 2) - 0x00000400 - kStoredExpBias * kMul1;
  constexpr uint32_t kAdd2 = (0x07060504 << 2) - kStoredExpBias * kMul2;
  uint32_t exp_buffer10 = stored_exp0 * kMul1 + kAdd1;
  uint32_t exp_buffer20 = stored_exp0 * kMul2 + kAdd2;
  uint32_t exp_buffer11 = stored_exp1 * kMul1 + kAdd1;
  uint32_t exp_buffer21 = stored_exp1 * kMul2 + kAdd2;

  PRAGMA_UNROLL
  for (uint32_t index = 0; index < 2; index++) {
    uint32_t packed = qb[index];
    uint32_t aligned[2] = {packed << 4, packed};
    PRAGMA_UNROLL
    for (uint32_t half = 0; half < 2; half++) {
      uint32_t magnitudes = (aligned[half] & 0x70707070) >> 4;
      uint32_t even = __byte_perm(magnitudes, 0, 0x4420);
      uint32_t odd = __byte_perm(magnitudes, 0, 0x4431);
      uint32_t selectors = even | (odd << 4);
      uint32_t exp = half == 0
                         ? __byte_perm(exp_buffer10, exp_buffer20, selectors)
                         : __byte_perm(exp_buffer11, exp_buffer21, selectors);
      res[index * 2 + half] =
          lop3_and_or(aligned[half], 0x80808080, exp);
    }
  }
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
