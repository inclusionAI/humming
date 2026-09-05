#pragma once

#include <humming/mma/mxmma.cuh>
#include <humming/mma/umma.cuh>


template <class Ctx, class ArithClass>
struct MXUMMA : MXMMA<Ctx, ArithClass> {
  using Base = MXMMA<Ctx, ArithClass>;
  using BlockShape = typename Ctx::BlockShape;
  using SharedStorage = typename Ctx::SharedStorage;
  using MmaOpClass = typename Ctx::MmaOpClass;
  using ElementA = typename Ctx::ElementA;
  using ElementB = typename Ctx::ElementB;
  using Base::ctx;
  static constexpr uint32_t kOperandColumns = 32;
  static constexpr uint32_t kScaleAColumn = 64;
  static constexpr uint32_t kScaleBColumn = 68;
  static constexpr uint32_t kAccumulatorColumn = 72;
  static constexpr uint32_t kTmemColumns = 256;
  static constexpr uint32_t kActivationType = std::is_same<ElementA, Float8E5M2>::value ? 1 : 0;
  static constexpr uint32_t kWeightType = std::is_same<ElementB, Float4E2M1>::value ? 5 : kActivationType;

  static_assert(std::is_same<ElementA, Float8E4M3>::value ||
                std::is_same<ElementA, Float8E5M2>::value);
  static_assert(MmaOpClass::kScaleVec == 1);
  static_assert(BlockShape::N == 128 && BlockShape::K == 128);
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
    // Replicate scale rows across all four 32-lane TMEM partitions.
    uint32_t weights[4], inputs[4];
    const uint32_t *weight_scales = reinterpret_cast<const uint32_t *>(ctx.smem.stages[stage_id].bs);
    const uint32_t *input_scales = reinterpret_cast<const uint32_t *>(ctx.smem.stages[stage_id].as);
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < 4; i++) {
      uint32_t row = ctx.lane_id() + i * 32;
      // Canonical MXMMA words interleave two K groups and two N rows.
      uint32_t offset = row / 16 * 8 + row % 8;
      uint32_t shift = (row % 16 / 8) * 16;
      weights[i] = ((weight_scales[offset] >> shift) & 0xffffu)
          | ((weight_scales[BlockShape::N / 2 + offset] >> shift) << 16);
      inputs[i] = row < BlockShape::M ? input_scales[row] : Base::kSFOneWord;
    }
    tcgen05_st_32x32b_x4(base + kScaleAColumn, weights);
    tcgen05_st_32x32b_x4(base + kScaleBColumn, inputs);
    tcgen05_wait_st();
    tcgen05_fence_before_thread_sync();
    if constexpr (!Ctx::kUseTmaA) {
      if (threadIdx.x == 0) tma_fence_async_shared();
    }
    ctx.sync_math_threads();
    tcgen05_fence_after_thread_sync();
    if (threadIdx.x == 0) {
      PRAGMA_UNROLL
      for (uint32_t k = 0; k < 4; k++) {
        const void *ptr = &ctx.smem.stages[stage_id].a[k * 2];
        uint64_t descriptor = ((uint64_t(cast_smem_ptr_to_uint(ptr)) >> 4) & 0x3fff)
            | (uint64_t(1) << 16) | (uint64_t(64) << 32)
            | (uint64_t(1) << 46) | (uint64_t(2) << 61);
        tcgen05_mma_mxfp8<BlockShape::M, kWeightType, kActivationType>(
            base + kAccumulatorColumn,
            base + operand_buffer * kOperandColumns + k * 8, descriptor,
            base + kScaleAColumn, base + kScaleBColumn,
            k, !first_issue || k != 0);
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

  template <class OutputArithmetic>
  CUDA_INLINE void write_native_output(OutputArithmetic &arith) {
    static_assert(!Ctx::kUseStreamK && BlockShape::K == Ctx::WarpShape::K);
    static_assert(Ctx::kNumWriteSplits == 1);
    __nv_bfloat16 *out = reinterpret_cast<__nv_bfloat16 *>(ctx.smem.reduce);
    const uint32_t swizzle = offsetof(SharedStorage, reduce) / 128 % 8;
    const uint32_t n = threadIdx.x;
    PRAGMA_UNROLL
    for (uint32_t m = 0; m < BlockShape::M; m += 8) {
      uint32_t values[8];
      tcgen05_ld_32x32b_x8(ctx.smem.umma_tmem_col + kAccumulatorColumn + m, values);
      tcgen05_wait_ld();
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < 8; i++) {
        uint32_t row = (n / 64) * BlockShape::M + m + i;
        uint32_t col = ((n % 64 / 8) ^ ((m + i + swizzle) % 8)) * 8 + n % 8;
        out[row * 64 + col] = arith.apply_native_output(__uint_as_float(values[i]));
      }
    }
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
