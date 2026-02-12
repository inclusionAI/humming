#pragma once

#include <humming/utils/all.cuh>


template <uint32_t swizzle_bytes = 128>
CUDA_INLINE uint64_t make_wgmma_smem_desc(void *smem_ptr, uint32_t iter_id) {
  static_assert(swizzle_bytes == 128 || swizzle_bytes == 64);

  constexpr uint64_t swizzle_type = swizzle_bytes == 128 ? 1 : 2;
  constexpr uint64_t stride = (swizzle_bytes * 8) >> 4;
  constexpr uint64_t desc_base = (swizzle_type << 62) | (stride << 32);

  uint64_t addr = cast_smem_ptr_to_uint(smem_ptr);
  uint64_t desc = desc_base;
  uint64_t base_offset = 0;

  desc |= (addr >> 4);

  return desc;
};


template <
    class MmaOpClass_, class SharedStorage, class ArithClass,
    class BlockShape, class WarpShape,
    class ElementA, class ElementB,
    class QuantParamConfig>
struct WGMMA {
public:
  using MmaOpClass = MmaOpClass_;
  using MmaShape = class MmaOpClass::MmaShape;

  static constexpr bool kHasDynamicZeroPoint = QuantParamConfig::kHasDynamicZeroPoint;

  static constexpr uint32_t kPartMmaShapeK = 256 / ElementA::kBits;
  static constexpr uint32_t M_WARPS = BlockShape::M / WarpShape::M;
  static constexpr uint32_t N_WARPS = BlockShape::N / WarpShape::N;
  static constexpr uint32_t K_WARPS = BlockShape::K / WarpShape::K;
  static constexpr uint32_t kWarpItersK = WarpShape::K / (256 / ElementA::kBits);
  static constexpr uint32_t kSwizzleBytes = ElementA::kBits * BlockShape::K >= 1024 ? 128 : 64;

  SharedStorage &smem;
  ArithClass &arith;
  uint32_t regs_qb[2][ElementB::kBits * (16 / ElementA::kBits)];
  typename MmaOpClass::ARegisters regs_b[2][WarpShape::N * 4 / MmaShape::M][kPartMmaShapeK / MmaShape::K];
  typename MmaOpClass::CRegisters regs_c[2][WarpShape::N * 4 / MmaShape::M][WarpShape::M / MmaShape::N];
  uint32_t smem_offset = 0;

  CUDA_INLINE
  WGMMA(SharedStorage &smem, ArithClass &arith)
      : smem(smem), arith(arith) {
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t m_warp_id = warp_id / N_WARPS % M_WARPS;
    uint32_t k_warp_id = warp_id / (N_WARPS * M_WARPS);

    constexpr uint32_t kSwizzleSizeK = kSwizzleBytes * 8 / ElementA::kBits;
    static_assert(kSwizzleSizeK >= WarpShape::K);

    const uint32_t row_offset = M_WARPS > 1 ? WarpShape::M * m_warp_id : 0;
    const uint32_t col_offset = K_WARPS > 1 ? WarpShape::K * k_warp_id : 0;

    smem_offset = row_offset * (kSwizzleBytes / 16);
    smem_offset += (col_offset % kSwizzleSizeK) * ElementA::kBits / 128;
    smem_offset += (col_offset / kSwizzleSizeK) * (BlockShape::M * kSwizzleBytes / 16);
  }

  CUDA_INLINE
  void zero_accum() {
    uint32_t *regs_c_ptr = regs_c_as_ptr();
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < sizeof(regs_c) / 4; i++) {
      regs_c_ptr[i] = 0;
    };
  };

  CUDA_INLINE
  void transform_b(uint32_t buffer_id) {
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < WarpShape::N / (MmaShape::M / 4); i++) {
      uint32_t *regs_b_ptr = reinterpret_cast<uint32_t *>(regs_b[buffer_id][i * 64 / MmaShape::M]);
      uint4 zp_vals = arith.prepare_zp_for_dequant(buffer_id, i);
      uint32_t *zp_vals_ptr = reinterpret_cast<uint32_t *>(&zp_vals);
      dequant<ElementB, ElementA, kHasDynamicZeroPoint>(regs_qb[buffer_id], regs_b_ptr, i, zp_vals_ptr);
      arith.may_apply_bs_and_zp_on_b(regs_b_ptr, i, buffer_id);
      uint32_t tmp = regs_b_ptr[1];
      regs_b_ptr[1] = regs_b_ptr[2];
      regs_b_ptr[2] = tmp;
    };
  };

  CUDA_INLINE
  void run(uint32_t stage_id, uint32_t iter_id) {
    static_assert(WarpShape::M == MmaShape::N);
    uint32_t buffer_id = iter_id % 2;

    PRAGMA_UNROLL
    for (uint32_t k = 0; k < kPartMmaShapeK / MmaShape::K; k++) {
      int4 *smem_ptr = smem.a[stage_id] + smem_offset + iter_id * 2 + k;
      uint64_t desc = make_wgmma_smem_desc<kSwizzleBytes>(smem_ptr, iter_id);

      constexpr uint32_t kNumIters = WarpShape::N / (MmaShape::M / 4);

      wgmma_fence();
      PRAGMA_UNROLL
      for (uint32_t j = 0; j < kNumIters; j++) {
        MmaOpClass::fma(regs_b[buffer_id][j][k], desc, regs_c[0][j][0]);
        wgmma_commit();
        wgmma_wait<0>();
        arith.may_apply_as_and_bs_on_wgmma_c(regs_c_as_ptr(), j, k, iter_id);
      }
    }
  };

  template <class T = uint32_t>
  CUDA_INLINE T *regs_qb_as_ptr(uint32_t buffer_id) {
    return reinterpret_cast<T *>(regs_qb[buffer_id]);
  };

  template <class T = uint32_t>
  CUDA_INLINE T *regs_c_as_ptr(uint32_t buffer_id = 0) {
    return reinterpret_cast<T *>(regs_c[buffer_id]);
  };

  template <class T = uint32_t>
  CUDA_INLINE T *final_regs_c_as_ptr() {
    uint32_t index = 0;
    constexpr bool kHasInputScale = QuantParamConfig::kHasInputScale;
    constexpr bool kHasWeightScale = QuantParamConfig::kHasWeightScale;
    constexpr bool kIsGroupInputScale = kHasInputScale && QuantParamConfig::kInputScaleGroupSize > 0;
    constexpr bool kIsGroupWeightScale = kHasWeightScale && QuantParamConfig::kWeightScaleGroupSize > 0;
    if constexpr (ElementA::kBits < 16 && (kIsGroupInputScale || kIsGroupWeightScale)) {
      index = 1;
    }

    return regs_c_as_ptr<T>(index);
  };
};
