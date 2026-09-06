#pragma once

#include <humming/mma/mxmma.cuh>
#include <humming/mma/umma.cuh>


CUDA_INLINE void mxumma_store_scales(uint32_t address, const uint32_t *values) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1, %2, %3, %4};"
               :: "r"(address), "r"(values[0]), "r"(values[1]),
                  "r"(values[2]), "r"(values[3]) : "memory");
}


template <uint32_t kN, uint32_t kScaleVec, bool kE4M3>
CUDA_INLINE void mxumma_mma(uint32_t d, uint32_t a, uint64_t b,
                            uint32_t sfa, uint32_t sfb, uint32_t scale_id,
                            bool accumulate) {
  constexpr uint32_t descriptor = (1u << 7) | (1u << 10)
      | ((kN / 8) << 17) | (uint32_t(!kE4M3) << 23) | (8u << 24);
  uint32_t idesc = descriptor | (scale_id << 29) | (scale_id << 4);
  if constexpr (kScaleVec == 2) {
    asm volatile(
        "{ .reg .pred p; setp.ne.b32 p, %6, 0;\n"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block32 "
        "[%0], [%1], %2, %3, [%4], [%5], p; }"
        :: "r"(d), "r"(a), "l"(b), "r"(idesc), "r"(sfa), "r"(sfb),
           "r"(uint32_t(accumulate)) : "memory");
  } else {
    asm volatile(
        "{ .reg .pred p; setp.ne.b32 p, %6, 0;\n"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 "
        "[%0], [%1], %2, %3, [%4], [%5], p; }"
        :: "r"(d), "r"(a), "l"(b), "r"(idesc), "r"(sfa), "r"(sfb),
           "r"(uint32_t(accumulate)) : "memory");
  }
}


template <class Ctx, class ArithClass>
struct MXUMMA : MXMMA<Ctx, ArithClass> {
  using Base = MXMMA<Ctx, ArithClass>;
  using BlockShape = typename Ctx::BlockShape;
  using SharedStorage = typename Ctx::SharedStorage;
  using MmaOpClass = typename Ctx::MmaOpClass;
  using Base::ctx;
  static constexpr uint32_t kScaleVec = MmaOpClass::kScaleVec;
  static constexpr uint32_t kScaleWords = Ctx::kWarpIters * kScaleVec / 4;
  static constexpr uint32_t kOperandColumns = Ctx::kWarpIters * 8;
  static constexpr uint32_t kScaleAColumn = 2 * kOperandColumns;
  static constexpr uint32_t kScaleBColumn = kScaleAColumn + kScaleWords * 4;
  static constexpr uint32_t kAccumulatorColumn = kScaleBColumn + kScaleWords * 4;
  static constexpr uint32_t kTmemColumns =
      kAccumulatorColumn + BlockShape::M <= 128 ? 128 : 256;

  static_assert(std::is_same<typename Ctx::ElementA, Float4E2M1>::value);
  static_assert(BlockShape::N == 128 && (BlockShape::K == 128 || BlockShape::K == 256));
  static_assert(Ctx::kNumMathThreads == 128);

  CUDA_INLINE MXUMMA(Ctx &ctx, ArithClass &arith) : Base(ctx, arith) {}

  CUDA_INLINE static void init(SharedStorage &smem) {
    if (threadIdx.x < 32) tcgen05_alloc<kTmemColumns>(cast_smem_ptr_to_uint(&smem.umma_tmem_col));
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
    tcgen05_st_16x128b_x2(address, values);
    tcgen05_st_16x128b_x2(address | (16u << 16), values + 4);
    if (iter_id == Ctx::kWarpIters - 1) {
      tcgen05_wait_st();
      tcgen05_fence_before_thread_sync();
    }
  }

  CUDA_INLINE void run(uint32_t stage_id, uint32_t iter_id) {
    if (iter_id != Ctx::kWarpIters - 1) return;
    uint32_t base = ctx.smem.umma_tmem_col;
    // Replicate each scale matrix across all four 32-lane TMEM partitions.
    PRAGMA_UNROLL
    for (uint32_t word = 0; word < kScaleWords; word++) {
      uint32_t weights[4], inputs[4];
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < 4; i++) {
        uint32_t row = ctx.lane_id() + i * 32;
        weights[i] = Base::kSFOneWord;
        inputs[i] = Base::kSFOneWord;
        if constexpr (Base::kIsGroupOrBlockWeightScale)
          weights[i] = reinterpret_cast<const uint32_t *>(ctx.smem.stages[stage_id].bs)[word * BlockShape::N + row];
        if constexpr (Base::kIsGroupInputScale) {
          if (row < BlockShape::M)
            inputs[i] = reinterpret_cast<const uint32_t *>(ctx.smem.stages[stage_id].as)[word * BlockShape::M + row];
        }
      }
      mxumma_store_scales(base + kScaleAColumn + word * 4, weights);
      mxumma_store_scales(base + kScaleBColumn + word * 4, inputs);
    }
    tcgen05_wait_st();
    tcgen05_fence_before_thread_sync();
    if constexpr (!Ctx::kUseTmaA) {
      if (threadIdx.x == 0) tma_fence_async_shared();
    }
    ctx.sync_math_threads();
    tcgen05_fence_after_thread_sync();
    if (threadIdx.x == 0) {
      PRAGMA_UNROLL
      for (uint32_t k = 0; k < Ctx::kWarpIters; k++) {
        const void *ptr = &ctx.smem.stages[stage_id].a[k * 2];
        uint64_t descriptor = ((uint64_t(cast_smem_ptr_to_uint(ptr)) >> 4) & 0x3fff)
            | (uint64_t(1) << 16) | (uint64_t(BlockShape::K / 4) << 32)
            | (uint64_t(1) << 46) | (uint64_t(BlockShape::K == 128 ? 4 : 2) << 61);
        uint32_t word = (k * kScaleVec / 4) * 4;
        mxumma_mma<BlockShape::M, kScaleVec, MmaOpClass::kSFIsE4M3>(
            base + kAccumulatorColumn,
            base + operand_buffer * kOperandColumns + k * 8, descriptor,
            base + kScaleAColumn + word, base + kScaleBColumn + word,
            (k * kScaleVec) % 4, !first_issue || k != 0);
      }
      tcgen05_commit(cast_smem_ptr_to_uint(&ctx.smem.umma_mbar));
    }
    first_issue = false;
    operand_buffer ^= 1;
  }

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
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < 8; i++) values[i] = __shfl_sync(0xffffffff, values[i], source_lane);
      umma_transpose_bit<1, 4>(values);
      umma_transpose_bit<2, 8>(values);
      umma_transpose_bit<4, 16>(values);
      PRAGMA_UNROLL
      for (uint32_t n = 0; n < 4; n++) {
        this->regs_c[m / 2][n][m % 2 * 2] = __uint_as_float(values[n * 2]);
        this->regs_c[m / 2][n][m % 2 * 2 + 1] = __uint_as_float(values[n * 2 + 1]);
      }
    }
    return Base::template regs_c_as_ptr<T>();
  }

private:
  bool first_issue = true;
  uint32_t operand_buffer = 0;
  uint32_t phase = 0;
};
