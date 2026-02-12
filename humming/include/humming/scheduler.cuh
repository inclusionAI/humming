#pragma once

#include <humming/utils/all.cuh>

template <
    class SharedStorage,
    class ProblemShape, class BlockShape,
    class SchedulerConfig, class PipelineConfig, class QuantParamConfig, class MoEConfig>
class Scheduler {
private:
  static constexpr bool kUseCpAsync = PipelineConfig::kUseCpAsync;
  static constexpr bool kUseWarpSpec = PipelineConfig::kUseWarpSpec;
  static constexpr bool kUseStreamK = SchedulerConfig::kUseStreamK;
  static constexpr bool kIsMoE = MoEConfig::kIsMoE;
  static constexpr bool kIsMoEDown = MoEConfig::kIsMoEDown;
  static constexpr uint32_t kTopK = MoEConfig::kTopK;
  static constexpr uint32_t kNumThreads = PipelineConfig::kNumThreads;

  static constexpr bool kIsGroupInputScale = QuantParamConfig::kHasInputScale && QuantParamConfig::kInputScaleGroupSize > 0;
  static constexpr bool kIsGroupWeightScale = QuantParamConfig::kHasWeightScale && QuantParamConfig::kWeightScaleGroupSize > 0;
  static constexpr uint32_t kInputScaleGroupSize = kIsGroupInputScale ? QuantParamConfig::kInputScaleGroupSize : 1;
  static constexpr uint32_t kWeightScaleGroupSize = kIsGroupWeightScale ? QuantParamConfig::kWeightScaleGroupSize : 1;
  static constexpr uint32_t kMaxGroupSize = MAX(kInputScaleGroupSize, kWeightScaleGroupSize);

  static constexpr uint32_t N_BLOCKS = ProblemShape::N / BlockShape::N;
  static constexpr uint32_t K_BLOCKS = ProblemShape::K / BlockShape::K;

  uint32_t m_blocks;
  uint32_t mn_blocks;
  uint32_t mnk_blocks;

  uint32_t streamk_mnk_total_iters;
  uint32_t streamk_mnk_iters;
  uint32_t streamk_mnk_next_index;
  uint32_t dp_mn_total_iters;
  uint32_t dp_mn_iters;
  uint32_t dp_mn_next_index;

public:
  SharedStorage &smem;
  const uint32_t *row_index_blocks;
  const uint32_t *expert_ids;
  uint32_t expert_id;

  uint32_t m_block_id;
  uint32_t n_block_id;
  uint32_t k_block_id;

  uint32_t slice_iters;
  uint32_t slice_count;
  uint32_t slice_id;
  uint32_t locks_offset;

  CUDA_INLINE
  Scheduler(SharedStorage &smem, const uint32_t *row_index_blocks, const uint32_t *expert_ids, uint32_t shape_m)
      : smem(smem), row_index_blocks(row_index_blocks), expert_ids(expert_ids) {
    m_blocks = CEIL_DIV(shape_m, BlockShape::M);
    mn_blocks = m_blocks * N_BLOCKS;
    mnk_blocks = mn_blocks * K_BLOCKS;

    if constexpr (kUseStreamK) {

      uint32_t streamk_mn_blocks = mn_blocks;
      if (mn_blocks > gridDim.x) {
        streamk_mn_blocks = mn_blocks % gridDim.x;
        if (streamk_mn_blocks && streamk_mn_blocks * 3 <= gridDim.x) streamk_mn_blocks += gridDim.x;
      }

      dp_mn_iters = (mn_blocks - streamk_mn_blocks) / gridDim.x;

      uint32_t streamk_mnk_blocks = streamk_mn_blocks * K_BLOCKS;

      streamk_mnk_total_iters = CEIL_DIV(streamk_mnk_blocks, gridDim.x);

      constexpr int32_t blocks_per_group = kMaxGroupSize / BlockShape::K;

      if constexpr (blocks_per_group > 1) {
        streamk_mnk_total_iters = blocks_per_group * CEIL_DIV(streamk_mnk_total_iters, blocks_per_group);
      };

      streamk_mnk_next_index = gridDim.x * dp_mn_iters * K_BLOCKS + streamk_mnk_total_iters * blockIdx.x;

      if (streamk_mnk_next_index >= mnk_blocks) {
        streamk_mnk_iters = 0;
      } else {
        streamk_mnk_iters = mnk_blocks - streamk_mnk_next_index;
        if (streamk_mnk_iters > streamk_mnk_total_iters) streamk_mnk_iters = streamk_mnk_total_iters;
      };
    } else {
      dp_mn_iters = mn_blocks / gridDim.x;
      if (blockIdx.x < mn_blocks % gridDim.x) { dp_mn_iters += 1; };
    }

    dp_mn_total_iters = dp_mn_iters;
    slice_count = 1;
    slice_id = 0;

    if (dp_mn_iters) { dp_mn_next_index = blockIdx.x; };
  };


  CUDA_INLINE
  bool get_next_block() {
    bool has_next_block = false;
    if (dp_mn_iters) {
      slice_iters = K_BLOCKS;
      m_block_id = dp_mn_next_index / N_BLOCKS;
      n_block_id = dp_mn_next_index % N_BLOCKS;
      k_block_id = 0;

      dp_mn_next_index += gridDim.x;
      dp_mn_iters--;
      has_next_block = true;
    } else if constexpr (kUseStreamK) {
      has_next_block = get_streamk_next_block();
    }

    if (kIsMoE && has_next_block) { fetch_moe_block(); }

    return has_next_block;
  };

  CUDA_INLINE
  bool get_streamk_next_block() {
    if (!streamk_mnk_iters) return false;
    uint32_t streamk_mn_index = streamk_mnk_next_index / K_BLOCKS;
    m_block_id = streamk_mn_index / N_BLOCKS;
    n_block_id = streamk_mn_index % N_BLOCKS;
    k_block_id = streamk_mnk_next_index - streamk_mn_index * K_BLOCKS;

    slice_iters = K_BLOCKS - k_block_id;
    slice_iters = slice_iters > streamk_mnk_iters ? streamk_mnk_iters : slice_iters;

    streamk_mnk_iters -= slice_iters;
    streamk_mnk_next_index += slice_iters;

    if (k_block_id == 0) {
      slice_id = 0;
      slice_count = CEIL_DIV(K_BLOCKS - slice_iters, streamk_mnk_total_iters) + 1;
    } else {
      slice_id = k_block_id / streamk_mnk_total_iters;
      uint32_t slice_first_block_iters = k_block_id - slice_id * streamk_mnk_total_iters;
      slice_count = CEIL_DIV(K_BLOCKS - slice_first_block_iters, streamk_mnk_total_iters);
      if (slice_first_block_iters) {
        slice_id++;
        slice_count++;
      }
    }

    slice_id = slice_count - 1 - slice_id;

    locks_offset = streamk_mn_index - dp_mn_total_iters * gridDim.x;

    return true;
  };

  CUDA_INLINE
  void fetch_moe_block() {
    if (kUseWarpSpec && threadIdx.x < kNumThreads) return;

    expert_id = expert_ids[m_block_id];

    const uint32_t *gmem_ptr = row_index_blocks + m_block_id * BlockShape::M;
    const int4 *gmem_ptr_load = reinterpret_cast<const int4 *>(gmem_ptr);
    int4 *smem_ptr_load = reinterpret_cast<int4 *>(smem.wr_row_index);

    legacy_load_1d<kUseCpAsync, BlockShape::M / 4, kNumThreads>(gmem_ptr_load, smem_ptr_load);
    if constexpr (kUseCpAsync) cp_async_commit_group();
    if constexpr (kUseCpAsync) cp_async_wait_group<0>();

    if constexpr (kUseWarpSpec) __syncwarp();
    else __syncthreads();

    static_assert(kNumThreads >= BlockShape::M);
    if (threadIdx.x < BlockShape::M) {
      uint32_t idx = smem.wr_row_index[threadIdx.x];
      if constexpr (!kIsMoEDown) { smem.rd_row_index[threadIdx.x] = idx / kTopK; };
    };

    if constexpr (kUseWarpSpec) __syncwarp();
    else __syncthreads();
  };
};
