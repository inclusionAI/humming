#pragma once

#include <humming/mma/wmma.cuh>
#include <humming/utils/ptx/barrier.cuh>
#include <humming/utils/ptx/tcgen05.cuh>
#include <humming/utils/ptx/tma.cuh>


template <class Ctx, class ArithClass>
struct UMMA : WMMA<Ctx, ArithClass> {
  using Base = WMMA<Ctx, ArithClass>;
  using BlockShape = typename Ctx::BlockShape;
  using WarpShape = typename Ctx::WarpShape;
  using SharedStorage = typename Ctx::SharedStorage;
  using Base::ctx;

  static constexpr bool kUseBlockScale = Ctx::kUseBlockScaledMma;
  static constexpr uint32_t kOperandColumns = Ctx::kWarpIters * 8;
  static constexpr bool kUseFp4 = Ctx::ElementA::kBits == 4;
  static constexpr uint32_t kScaleGroupSize = Ctx::kIsGroupInputScale ? Ctx::kInputScaleGroupSize : Ctx::kWeightScaleGroupSize;
  static constexpr uint32_t kScalesPerIter = kUseBlockScale ? Ctx::kPartMmaShapeK / kScaleGroupSize : 1;
  static constexpr uint32_t kScaleWords = CEIL_DIV(Ctx::kWarpIters * kScalesPerIter, 4);
  static constexpr bool kScaleIsE4M3 = Ctx::MmaOpClass::kSFIsE4M3;
  static constexpr uint32_t kScaleOne = kScaleIsE4M3 ? 0x38383838 : 0x7f7f7f7f;
  static constexpr uint32_t kWeightScaleColumns = kUseBlockScale ? 4 * kScaleWords : 0;
  // Scale operand addresses are 4-column aligned, including partial M tiles.
  static constexpr uint32_t kInputScaleStride = MAX(4u, static_next_power_of_2(CEIL_DIV(BlockShape::M, 32)));
  static constexpr uint32_t kInputScaleColumns = kUseBlockScale ? kInputScaleStride * kScaleWords : 0;
  static constexpr uint32_t kOperandBufferColumns = CEIL_DIV(kOperandColumns + kWeightScaleColumns + kInputScaleColumns, 16) * 16;
  static constexpr uint32_t kOutputGroups = CEIL_DIV(BlockShape::N, 128);
  static constexpr uint32_t kPreferredOperandBuffers = Ctx::kUmmaCtaGroupSize == 2 ? 4 : Ctx::kNumStages;
  static constexpr uint32_t kBufferedTmemColumns = static_next_power_of_2(kOutputGroups * (kPreferredOperandBuffers * kOperandBufferColumns + WarpShape::M));
  static constexpr bool kBufferedOperandsFit = kBufferedTmemColumns * Ctx::kNumCtasPerSm <= 512;
  static constexpr uint32_t kNumOperandBuffers = kBufferedOperandsFit ? kPreferredOperandBuffers : 2;
  static constexpr uint32_t kAccumulatorColumn = kNumOperandBuffers * kOperandBufferColumns;
  // Each logical 128-channel partition has its own operands and accumulator.
  static constexpr uint32_t kGroupColumns = kAccumulatorColumn + WarpShape::M;
  static constexpr uint32_t kTmemColumns = static_next_power_of_2(kOutputGroups * kGroupColumns);

  static constexpr bool kUseBf16 = std::is_same<typename Ctx::ElementA, BFloat16>::value;
  static constexpr bool kUseFp8 = std::is_same<typename Ctx::ElementA, Float8E4M3>::value ||
                                  std::is_same<typename Ctx::ElementA, Float8E5M2>::value ||
                                  std::is_same<typename Ctx::ElementA, Float8E3M4>::value;
  static_assert(kUseFp4 || kUseFp8 || kUseBf16 || std::is_same<typename Ctx::ElementA, Float16>::value);
  static_assert(BlockShape::N == 64 || BlockShape::N == 128 || BlockShape::N == 256 || BlockShape::N == 512);
  static_assert(WarpShape::M >= 8 && WarpShape::M <= 256 && WarpShape::M % (8 * Ctx::kUmmaCtaGroupSize) == 0,
                "UMMA requires warp M in [8, 256], divisible by 8 (one CTA) or 16 (two CTAs)");
  static_assert(WarpShape::N == 32);
  static_assert(WarpShape::K >= 32 && WarpShape::K % 32 == 0,
                "UMMA requires K divisible by 32");
  static_assert(std::is_same<typename Ctx::ElementC, Float16>::value ||
                std::is_same<typename Ctx::ElementC, BFloat16>::value);
  static_assert(kTmemColumns <= 512);

  CUDA_INLINE UMMA(Ctx &ctx, ArithClass &arith) : Base(ctx, arith), tmem_column(ctx.smem.umma_tmem_col) {}

  CUDA_INLINE static void init(SharedStorage &smem) {
    if (threadIdx.x < 32) {
      tcgen05_alloc<kTmemColumns, Ctx::kUmmaCtaGroupSize>(cast_smem_ptr_to_uint(&smem.umma_tmem_col));
    }
    __syncthreads();
  }

  CUDA_INLINE static void dealloc(SharedStorage &smem) {
    if (threadIdx.x < 32) tcgen05_dealloc<kTmemColumns, Ctx::kUmmaCtaGroupSize>(smem.umma_tmem_col);
  }

  CUDA_INLINE void transform_b(uint32_t buffer_id, uint32_t iter_id) {
    if constexpr ((kUseFp8 || kUseFp4) && Ctx::ElementB::kBits == Ctx::ElementA::kBits) {
      // Native mixed FP8/FP4 consumes the original encoding, without conversion
      // to the activation dtype. Equal dtypes are already loaded into regs_b.
      if constexpr (!std::is_same<typename Ctx::ElementA, typename Ctx::ElementB>::value) {
        uint32_t *values = reinterpret_cast<uint32_t *>(this->regs_b[buffer_id]);
        PRAGMA_UNROLL
        for (uint32_t i = 0; i < sizeof(this->regs_b[0]) / sizeof(uint32_t); i++) {
          values[i] = this->regs_qb[buffer_id][i];
        }
      }
    } else {
      Base::transform_b(buffer_id, iter_id);
    }
  }

  template <uint32_t kFragments>
  CUDA_INLINE void store_b(const uint32_t (&values)[kFragments][8], uint32_t iter_id) {
    static_assert(kFragments == 1 || kFragments == 2 || kFragments == 4);
    uint32_t group_base = tmem_column + ctx.math_group * kGroupColumns;
    uint32_t address = group_base + operand_buffer * kOperandBufferColumns + iter_id * 8;
    if constexpr (kFragments == 1) {
      tcgen05_st_16x128b_x2(address, values[0]);
      tcgen05_st_16x128b_x2(address | (16u << 16), values[0] + 4);
    } else if constexpr (kFragments == 2) {
      tcgen05_st_16x128b_x4(address, values[0], values[1]);
      tcgen05_st_16x128b_x4(address | (16u << 16), values[0] + 4, values[1] + 4);
    } else {
      tcgen05_st_16x128b_x8(address, values[0], values[1], values[2], values[3]);
      tcgen05_st_16x128b_x8(address | (16u << 16), values[0] + 4, values[1] + 4, values[2] + 4, values[3] + 4);
    }
  }

  CUDA_INLINE void store_weight_scales(uint32_t stage, uint32_t k_block, uint32_t n_block) {
    if constexpr (kUseBlockScale) {
      if constexpr (kOutputGroups < Ctx::TuningConfig::kUmmaNumDequantWarpgroups) {
        if (ctx.dequant_group_id() != 0) return;
      }
      const uint32_t *scales = reinterpret_cast<const uint32_t *>(ctx.smem.stages[stage].bs);
      uint32_t base = tmem_column + ctx.math_group * kGroupColumns +
                      operand_buffer * kOperandBufferColumns + kOperandColumns;
      uint32_t phase = (k_block * Ctx::kWarpIters * kScalesPerIter) % 4;
      PRAGMA_UNROLL
      for (uint32_t word = 0; word < kScaleWords; word++) {
        uint32_t values[4];
        PRAGMA_UNROLL
        for (uint32_t column = 0; column < 4; column++) {
          uint32_t n = ctx.math_group * 128 + column * 32 + threadIdx.x % 32;
          uint32_t packed = kScaleOne;
          if constexpr (Ctx::kIsGroupWeightScale) {
            if (n < BlockShape::N) {
              uint32_t row = n + (n_block * BlockShape::N) % 128;
              uint32_t index = word * MAX(BlockShape::N, 128) + row / 128 * 128 + row % 32 * 4 + row % 128 / 32;
              packed = scales[index] >> (phase * 8);
            }
          }
          values[column] = packed;
        }
        tcgen05_st_32x32b<4>(base + word * 4, values);
      }
    }
  }

  CUDA_INLINE void store_input_scales(uint32_t stage, uint32_t buffer, uint32_t k_block, uint32_t m_offset) {
    if constexpr (kUseBlockScale) {
      if constexpr (kOutputGroups < Ctx::TuningConfig::kUmmaNumDequantWarpgroups) {
        if (ctx.dequant_group_id() != 0) return;
      }
      const uint32_t *scales = reinterpret_cast<const uint32_t *>(ctx.smem.stages[stage].as);
      uint32_t base = tmem_column + ctx.math_group * kGroupColumns +
                      buffer * kOperandBufferColumns + kOperandColumns + kWeightScaleColumns;
      uint32_t phase = (k_block * Ctx::kWarpIters * kScalesPerIter) % 4;
      PRAGMA_UNROLL
      for (uint32_t word = 0; word < kScaleWords; word++) {
        uint32_t values[kInputScaleStride];
        PRAGMA_UNROLL
        for (uint32_t column = 0; column < kInputScaleStride; column++) {
          uint32_t m = column * 32 + threadIdx.x % 32;
          uint32_t packed = kScaleOne;
          if constexpr (Ctx::kIsGroupInputScale) {
            if (m < BlockShape::M) {
              uint32_t row = m;
              if constexpr (Ctx::kIsGroupedGemm && Ctx::kUseMMajorInputScale) row += m_offset % 4;
              packed = scales[word * SharedStorage::kScaleBlockM + row] >> (phase * 8);
            }
          }
          values[column] = packed;
        }
        tcgen05_st_32x32b<kInputScaleStride>(base + word * kInputScaleStride, values);
      }
    }
  }

  CUDA_INLINE void set_operand_buffer(uint32_t buffer) { operand_buffer = buffer; }

  CUDA_INLINE void load_output_chunk(uint32_t m, uint32_t rows, uint32_t *lower, uint32_t *upper) {
    uint32_t address = tmem_column + ctx.math_group * kGroupColumns +
                       kAccumulatorColumn + m * 32;
    if constexpr (Ctx::kUmmaCtaGroupSize == 2) {
      tcgen05_ld_16x256b_x4(address, lower);
      tcgen05_ld_16x256b_x4(address | (16u << 16), upper);
    } else {
      if (rows == 8) {
        tcgen05_ld_16x128b_x2(address, lower);
        tcgen05_ld_16x128b_x2(address | (16u << 16), upper);
      } else if (rows <= 24) {
        tcgen05_ld_16x128b_x4(address, lower);
        tcgen05_ld_16x128b_x4(address | (16u << 16), upper);
        if (rows == 24) {
          tcgen05_ld_16x128b_x2(address + 16, lower + 8);
          tcgen05_ld_16x128b_x2((address + 16) | (16u << 16), upper + 8);
        }
      } else {
        tcgen05_ld_16x128b_x8(address, lower);
        tcgen05_ld_16x128b_x8(address | (16u << 16), upper);
      }
    }
    tcgen05_wait_ld();
  }

  // Called by one elected lane after operand readiness and proxy fencing.
  CUDA_INLINE void issue(uint32_t stage_id, uint32_t buffer, bool is_first) {
    uint32_t base = tmem_column + ctx.math_group * kGroupColumns;
    uint32_t accumulator = base + kAccumulatorColumn;
    PRAGMA_UNROLL
    for (uint32_t k = 0; k < Ctx::kWarpIters; k++) {
      uint32_t k_offset = k * Ctx::kPartMmaShapeK;
      constexpr uint32_t kElementsPerInt4 = 128 / Ctx::ElementA::kBits;
      constexpr uint32_t kSwizzleK = MIN(BlockShape::K, 8 * kElementsPerInt4);
      uint32_t row = k_offset / kSwizzleK * (BlockShape::M / Ctx::kUmmaCtaGroupSize);
      uint32_t offset = row * (kSwizzleK / kElementsPerInt4) + k_offset % kSwizzleK / kElementsPerInt4;
      uint64_t descriptor = tcgen05_smem_desc<kSwizzleK * Ctx::ElementA::kBits / 8>(&ctx.smem.stages[stage_id].a[offset]);
      using ElementB = typename Ctx::ElementB;
      // f8f6f4 descriptor format 2 selects the undocumented E3M4 format.
      constexpr uint32_t kWeightFormat = std::is_same<ElementB, Float4E2M1>::value   ? 5
                                         : std::is_same<ElementB, Float6E3M2>::value ? 4
                                         : std::is_same<ElementB, Float6E2M3>::value ? 3
                                         : std::is_same<ElementB, Float8E3M4>::value ? 2
                                         : std::is_same<ElementB, Float8E5M2>::value ? 1
                                                                                     : 0;
      constexpr uint32_t kInputFormat = std::is_same<typename Ctx::ElementA, Float8E3M4>::value   ? 2
                                        : std::is_same<typename Ctx::ElementA, Float8E5M2>::value ? 1
                                                                                                  : 0;
      if constexpr (kUseBlockScale) {
        uint32_t scale_base = base + buffer * kOperandBufferColumns + kOperandColumns;
        uint32_t scale_word = k * kScalesPerIter / 4;
        uint32_t scale_id = k * kScalesPerIter % 4;
        uint32_t weight_scale = scale_base + scale_word * 4;
        uint32_t input_scale = scale_base + kWeightScaleColumns + scale_word * kInputScaleStride;
        if constexpr (kUseFp4) {
          tcgen05_mma_mxf4nvf4<WarpShape::M, kScaleGroupSize, kScaleIsE4M3, Ctx::kUmmaCtaGroupSize,
                               std::is_same<ElementB, Float4E0M3>::value,
                               std::is_same<typename Ctx::ElementA, Float4E0M3>::value>(
              accumulator, base + buffer * kOperandBufferColumns + k * 8, descriptor,
              weight_scale, input_scale, scale_id, !is_first || k != 0);
        } else {
          tcgen05_mma_mxf8f6f4<WarpShape::M, kWeightFormat, kInputFormat, Ctx::kUmmaCtaGroupSize>(
              accumulator, base + buffer * kOperandBufferColumns + k * 8, descriptor,
              weight_scale, input_scale, scale_id, !is_first || k != 0);
        }
      } else if constexpr (kUseFp8) {
        tcgen05_mma_f8f6f4<WarpShape::M, kWeightFormat, kInputFormat, Ctx::kUmmaCtaGroupSize>(accumulator,
                                                                                              base + buffer * kOperandBufferColumns + k * 8, descriptor, !is_first || k != 0);
      } else {
        tcgen05_mma_f16<WarpShape::M, kUseBf16, Ctx::kUmmaCtaGroupSize>(accumulator,
                                                                        base + buffer * kOperandBufferColumns + k * 8, descriptor, !is_first || k != 0);
      }
    }
  }

private:
  const uint32_t tmem_column;
  uint32_t operand_buffer = 0;
};
