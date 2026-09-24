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

  static constexpr uint32_t kOperandColumns = Ctx::kWarpIters * 8;
  static constexpr uint32_t kOutputGroups = CEIL_DIV(BlockShape::N, 128);
  static constexpr uint32_t kStageTmemColumns = static_next_power_of_2(kOutputGroups * (Ctx::kNumStages * kOperandColumns + WarpShape::M));
  static constexpr bool kStageOperandsFit = kStageTmemColumns * Ctx::kNumCtasPerSm <= 512;
  static constexpr uint32_t kNumOperandBuffers = kStageOperandsFit ? Ctx::kNumStages : 2;
  static constexpr uint32_t kAccumulatorColumn = kNumOperandBuffers * kOperandColumns;
  // Logical 128-channel partitions share one physical dequantization WG.
  static constexpr uint32_t kGroupColumns = kAccumulatorColumn + WarpShape::M;
  static constexpr uint32_t kTmemColumns = static_next_power_of_2(kOutputGroups * kGroupColumns);

  static constexpr bool kUseBf16 = std::is_same<typename Ctx::ElementA, BFloat16>::value;
  static_assert(kUseBf16 || std::is_same<typename Ctx::ElementA, Float16>::value);
  static_assert(BlockShape::N == 64 || BlockShape::N == 128 || BlockShape::N == 256 || BlockShape::N == 512);
  static_assert(WarpShape::M >= 8 && WarpShape::M <= 256 && WarpShape::M % 8 == 0,
                "UMMA requires warp M in [8, 256], divisible by 8");
  static_assert(WarpShape::N == 32);
  static_assert(WarpShape::K >= 32 && WarpShape::K % 32 == 0,
                "UMMA requires K divisible by 32");
  static_assert(std::is_same<typename Ctx::ElementA, typename Ctx::ElementC>::value,
                "UMMA requires matching activation and output types");
  static_assert(kTmemColumns <= 512);

  CUDA_INLINE UMMA(Ctx &ctx, ArithClass &arith) : Base(ctx, arith), tmem_column(ctx.smem.umma_tmem_col) {}

  CUDA_INLINE static void init(SharedStorage &smem) {
    if (threadIdx.x < 32) {
      tcgen05_alloc<kTmemColumns>(cast_smem_ptr_to_uint(&smem.umma_tmem_col));
    }
    __syncthreads();
  }

  CUDA_INLINE static void dealloc(SharedStorage &smem) {
    if (threadIdx.x < 32) tcgen05_dealloc<kTmemColumns>(smem.umma_tmem_col);
  }

  template <uint32_t kFragments>
  CUDA_INLINE void store_b(const uint32_t (&values)[kFragments][8], uint32_t iter_id) {
    static_assert(kFragments == 2 || kFragments == 4);
    uint32_t group_base = tmem_column + ctx.math_group * kGroupColumns;
    uint32_t address = group_base + operand_buffer * kOperandColumns + iter_id * 8;
    if constexpr (kFragments == 2) {
      tcgen05_st_16x128b_x4(address, values[0], values[1]);
      tcgen05_st_16x128b_x4(address | (16u << 16), values[0] + 4, values[1] + 4);
    } else {
      tcgen05_st_16x128b_x8(address, values[0], values[1], values[2], values[3]);
      tcgen05_st_16x128b_x8(address | (16u << 16), values[0] + 4, values[1] + 4, values[2] + 4, values[3] + 4);
    }
    if (iter_id + kFragments == Ctx::kWarpIters) {
      tcgen05_wait_st();
      tcgen05_fence_before_thread_sync();
    }
  }

  CUDA_INLINE void set_operand_buffer(uint32_t buffer) { operand_buffer = buffer; }

  CUDA_INLINE void load_output_chunk(uint32_t m, uint32_t rows, uint32_t *lower, uint32_t *upper) {
    uint32_t address = tmem_column + ctx.math_group * kGroupColumns +
                       kAccumulatorColumn + m * 32;
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
    tcgen05_wait_ld();
  }

  // The caller publishes operand stores and synchronizes the issuing warp.
  CUDA_INLINE void issue(uint32_t stage_id, uint32_t buffer, bool is_first) {
    if (ctx.math_thread_id() % 128 < 32) {
      uint32_t base = tmem_column + ctx.math_group * kGroupColumns;
      uint32_t accumulator = base + kAccumulatorColumn;
      PRAGMA_UNROLL
      for (uint32_t k = 0; k < Ctx::kWarpIters; k++) {
        uint32_t k_offset = k * 16;
        constexpr uint32_t kSwizzleK = MIN(BlockShape::K, 64);
        uint32_t row = k_offset / kSwizzleK * BlockShape::M;
        uint32_t offset = row * (kSwizzleK / 8) + k_offset % kSwizzleK / 8;
        uint64_t descriptor = tcgen05_smem_desc_f16<kSwizzleK * 2>(&ctx.smem.stages[stage_id].a[offset]);
        tcgen05_mma_f16<WarpShape::M, kUseBf16>(accumulator,
                                       base + buffer * kOperandColumns + k * 8,
                                       descriptor, !is_first || k != 0);
      }
    }
  }

private:
  const uint32_t tmem_column;
  uint32_t operand_buffer = 0;
};
