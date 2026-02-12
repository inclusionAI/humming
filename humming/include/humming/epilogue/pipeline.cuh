#pragma once

#include <humming/utils/all.cuh>

#include <humming/epilogue/gmem_writer.cuh>
#include <humming/epilogue/smem_reducer.cuh>
#include <humming/epilogue/smem_writer.cuh>


template <
    class MmaOpClass, class SharedStorage, class ArithClass,
    class ProblemShape, class BlockShape, class WarpShape,
    class ElementA, class ElementC,
    class SchedulerConfig, class PipelineConfig, class EpilogueConfig,
    class QuantParamConfig, class MoEConfig>
class EpiloguePipeline {
private:
  using SmemReducer = EpilogueSmemReducer<MmaOpClass, BlockShape, WarpShape, ElementC, PipelineConfig>;
  using SmemWriter = EpilogueSmemWriter<MmaOpClass, ArithClass, BlockShape, WarpShape, ElementA, ElementC, PipelineConfig, QuantParamConfig>;
  using GmemWriter = EpilogueGmemWriter<ArithClass, ProblemShape, BlockShape, ElementC, SchedulerConfig, PipelineConfig, EpilogueConfig, MoEConfig>;
  using OutputPtrType = std::conditional_t<PipelineConfig::kUseTmaC, const void *, void *>;

public:
  SmemReducer smem_reducer;
  SmemWriter smem_writer;
  GmemWriter gmem_writer;
  ArithClass &arith;
  const uint32_t *GS;
  int32_t *locks;

  uint32_t slice_count;
  uint32_t slice_id;
  uint32_t locks_offset;

  CUDA_INLINE
  EpiloguePipeline(
      SharedStorage &smem, OutputPtrType output_ptr, ArithClass &arith,
      const uint32_t *GS, int32_t *locks, uint32_t output_shape_m)
      : GS(GS), locks(locks), arith(arith),
        smem_reducer(smem.reduce), smem_writer(smem.reduce, arith),
        gmem_writer(arith, smem.reduce, output_ptr, smem.wr_row_index, output_shape_m) {
  }

  CUDA_INLINE
  void call(uint32_t *regs_c_ptr) {
    if constexpr (BlockShape::K > WarpShape::K) smem_reducer.reduce(regs_c_ptr);
    smem_writer.write(regs_c_ptr, slice_count);
    sync_math_threads();
    if (slice_count > 1) acquire_gmem_barrier();
    gmem_writer.write(slice_id, slice_count);
    sync_math_threads();
    if (slice_count > 1) release_gmem_barrier();
  }

  CUDA_INLINE
  void sync_math_threads() {
    sync_part_threads<PipelineConfig::kNumMathThreads, PipelineConfig::kNumThreads>();
  }

  CUDA_INLINE
  void acquire_gmem_barrier() {
    barrier_acquire<PipelineConfig::kNumMathThreads, PipelineConfig::kNumThreads>(&locks[locks_offset], slice_id);
  }

  CUDA_INLINE
  void release_gmem_barrier() {
    barrier_release<PipelineConfig::kNumMathThreads, PipelineConfig::kNumThreads>(&locks[locks_offset], slice_id == slice_count - 1);
  }

  CUDA_INLINE
  void seek(uint32_t expert_id, uint32_t m_block_id, uint32_t n_block_id) {
    gmem_writer.seek(m_block_id, n_block_id);
    if constexpr (QuantParamConfig::kHasGlobalScale) {
      arith.gs = GS[MoEConfig::kIsMoE ? expert_id : 0];
    }
  };

  CUDA_INLINE
  void set_streamk_state(uint32_t slice_count_, uint32_t slice_id_, uint32_t locks_offset_) {
    slice_count = slice_count_;
    slice_id = slice_id_;
    locks_offset = locks_offset_;
  };
};
