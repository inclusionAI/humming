#pragma once

#include <humming/mma/wmma.cuh>
#include <humming/utils/ptx/barrier.cuh>
#include <humming/utils/ptx/tcgen05.cuh>
#include <humming/utils/ptx/tma.cuh>


template <uint32_t kRegisterBit, uint32_t kLaneBit>
CUDA_INLINE void umma_transpose_bit(uint32_t (&values)[8]) {
  bool upper = (threadIdx.x & kLaneBit) != 0;
  PRAGMA_UNROLL
  for (uint32_t i = 0; i < 8; i++) {
    if ((i & kRegisterBit) == 0) {
      uint32_t value = upper ? values[i] : values[i | kRegisterBit];
      value = __shfl_xor_sync(0xffffffff, value, kLaneBit);
      if (upper) values[i] = value;
      else values[i | kRegisterBit] = value;
    }
  }
}


template <class Ctx, class ArithClass>
struct UMMA : WMMA<Ctx, ArithClass> {
  using Base = WMMA<Ctx, ArithClass>;
  using BlockShape = typename Ctx::BlockShape;
  using WarpShape = typename Ctx::WarpShape;
  using SharedStorage = typename Ctx::SharedStorage;
  using Base::ctx;

  static constexpr uint32_t kOperandColumns = Ctx::kWarpIters * 8;
  static constexpr uint32_t kAccumulatorColumn = 2 * kOperandColumns;
  static constexpr uint32_t kTmemColumns = kAccumulatorColumn + BlockShape::M <= 128 ? 128 : 256;

  static_assert(std::is_same<typename Ctx::ElementA, BFloat16>::value);
  static_assert(BlockShape::N == 128 && BlockShape::K == 64);
  static_assert(BlockShape::M == 64 || BlockShape::M == 128);
  static_assert(WarpShape::M == BlockShape::M && WarpShape::N == 32 && WarpShape::K == BlockShape::K);
  static_assert(Ctx::kNumMathThreads == 128);

  CUDA_INLINE UMMA(Ctx &ctx, ArithClass &arith) : Base(ctx, arith) {}

  CUDA_INLINE static void init(SharedStorage &smem) {
    if (threadIdx.x < 32) {
      tcgen05_alloc<kTmemColumns>(cast_smem_ptr_to_uint(&smem.umma_tmem_col));
    }
    if (threadIdx.x == 0) __mbarrier_init(&smem.umma_mbar, 1);
    __syncthreads();
  }

  CUDA_INLINE static void dealloc(SharedStorage &smem) {
    if (threadIdx.x < 32) tcgen05_dealloc<kTmemColumns>(smem.umma_tmem_col);
  }

  CUDA_INLINE void zero_accum() {
    first_issue = true;
    operand_buffer = 0;
  }

  CUDA_INLINE void transform_b(uint32_t buffer_id, uint32_t iter_id) {
    Base::transform_b(buffer_id, iter_id);
    const uint32_t *values = reinterpret_cast<const uint32_t *>(this->regs_b[buffer_id]);
    uint32_t address = ctx.smem.umma_tmem_col + operand_buffer * kOperandColumns + iter_id * 8;
    // The standard MMA B fragment maps directly to two 16-row TMEM stores.
    tcgen05_st_16x128b_x2(address, values);
    tcgen05_st_16x128b_x2(address | (16u << 16), values + 4);
    if (iter_id == Ctx::kWarpIters - 1) {
      tcgen05_wait_st();
      tcgen05_fence_before_thread_sync();
    }
  }

  CUDA_INLINE void run(uint32_t stage_id, uint32_t iter_id) {
    if (iter_id != Ctx::kWarpIters - 1) return;
    // Publish completed cp.async activation loads to the MMA shared-memory proxy.
    if constexpr (!Ctx::kUseTmaA) {
      if (threadIdx.x == 0) tma_fence_async_shared();
    }
    ctx.sync_math_threads();
    tcgen05_fence_after_thread_sync();
    if (threadIdx.x == 0) {
      uint32_t base = ctx.smem.umma_tmem_col;
      PRAGMA_UNROLL
      for (uint32_t k = 0; k < Ctx::kWarpIters; k++) {
        uint64_t descriptor = tcgen05_smem_desc_bf16(&ctx.smem.stages[stage_id].a[k * 2]);
        tcgen05_mma_bf16<BlockShape::M>(base + kAccumulatorColumn,
                                      base + operand_buffer * kOperandColumns + k * 8,
                                      descriptor, !first_issue || k != 0);
      }
      tcgen05_commit(cast_smem_ptr_to_uint(&ctx.smem.umma_mbar));
    }
    first_issue = false;
    operand_buffer ^= 1;
  }

  // Retire every reader before releasing the activation stage or overwriting B.
  CUDA_INLINE void wait_stage() {
    mbarrier_wait(&ctx.smem.umma_mbar, phase);
    phase ^= 1;
    tcgen05_fence_after_thread_sync();
  }

  template <class T = uint32_t>
  CUDA_INLINE T *final_regs_c_as_ptr() {
    uint32_t lane = ctx.lane_id();
    uint32_t source_lane = (lane & 24u) | ((lane & 3u) << 1) | ((lane >> 2) & 1u);
    PRAGMA_UNROLL
    for (uint32_t m = 0; m < BlockShape::M / 8; m++) {
      uint32_t values[8];
      tcgen05_ld_32x32b_x8(ctx.smem.umma_tmem_col + kAccumulatorColumn + m * 8, values);
      tcgen05_wait_ld();
      // TMEM is [N, M]. Restore the standard m16n8 accumulator register layout.
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < 8; i++) {
        values[i] = __shfl_sync(0xffffffff, values[i], source_lane);
      }
      umma_transpose_bit<1, 4>(values);
      umma_transpose_bit<2, 8>(values);
      umma_transpose_bit<4, 16>(values);
      PRAGMA_UNROLL
      for (uint32_t n = 0; n < 4; n++) {
        this->regs_c[0][m / 2][n][m % 2 * 2] = __uint_as_float(values[n * 2]);
        this->regs_c[0][m / 2][n][m % 2 * 2 + 1] = __uint_as_float(values[n * 2 + 1]);
      }
    }
    return Base::template regs_c_as_ptr<T>();
  }

private:
  bool first_issue = true;
  uint32_t operand_buffer = 0;
  uint32_t phase = 0;
};
