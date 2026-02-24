#pragma once

#include <cuda_awbarrier_primitives.h>
#include <humming/memory/g2s_loader/loader_a.cuh>
#include <humming/memory/g2s_loader/loader_as.cuh>
#include <humming/memory/g2s_loader/loader_b.cuh>
#include <humming/memory/g2s_loader/loader_bias.cuh>
#include <humming/memory/g2s_loader/loader_bs.cuh>
#include <humming/memory/g2s_loader/loader_bzp.cuh>
#include <humming/memory/g2s_loader/loader_topk_weights.cuh>


template <
    class SharedStorage,
    class ProblemShape, class BlockShape, class PadShape,
    class ElementA, class ElementB, class ElementBS,
    class PipelineConfig, class EpilogueConfig,
    class QuantParamConfig, class MoEConfig>
class ProducerPipeline {
private:
  static constexpr uint32_t kNumThreads = PipelineConfig::kNumThreads;
  static constexpr uint32_t kNumLoadThreads = PipelineConfig::kNumLoadThreads;
  static constexpr uint32_t kNumMathThreads = PipelineConfig::kNumMathThreads;
  static constexpr uint32_t kLoadThreadOffset = kNumThreads - kNumLoadThreads;

  static constexpr bool kUseMBarrier = PipelineConfig::kUseMBarrier;
  static constexpr bool kUseCpAsync = PipelineConfig::kUseCpAsync;
  static constexpr bool kUseTmaA = PipelineConfig::kUseTmaA;
  static constexpr bool kUseTmaB = PipelineConfig::kUseTmaB;
  static constexpr bool kUseTmaBS = PipelineConfig::kUseTmaBS;
  static constexpr bool kUseTmaBZP = PipelineConfig::kUseTmaBZP;
  static constexpr bool kUseTmaBias = PipelineConfig::kUseTmaBias;

  static constexpr bool kHasInputScale = QuantParamConfig::kHasInputScale;
  static constexpr bool kHasWeightScale = QuantParamConfig::kHasWeightScale;
  static constexpr bool kIsChannelInputScale = kHasInputScale && QuantParamConfig::kInputScaleGroupSize == 0;
  static constexpr bool kIsGroupInputScale = kHasInputScale && QuantParamConfig::kInputScaleGroupSize > 0;
  static constexpr bool kIsChannelWeightScale = kHasWeightScale && QuantParamConfig::kWeightScaleGroupSize == 0;
  static constexpr bool kIsGroupWeightScale = kHasWeightScale && QuantParamConfig::kWeightScaleGroupSize > 0;
  static constexpr bool kHasDynamicZeroPoint = QuantParamConfig::kHasDynamicZeroPoint;
  static constexpr bool kHasBias = EpilogueConfig::kHasBias;

  static constexpr bool kIsMoE = MoEConfig::kIsMoE;
  static constexpr bool kIsMoEDown = MoEConfig::kIsMoEDown;
  static constexpr uint32_t kNumStages = PipelineConfig::kNumStages;

  template <bool kIsFirst = false>
  static constexpr uint2 get_stage_load_bytes() {
    uint32_t tma_load_bytes = 0;
    uint32_t legacy_load_bytes = 0;

    if constexpr (kUseTmaA) tma_load_bytes += sizeof(smem.a[0]);
    else legacy_load_bytes += sizeof(smem.a[0]);

    if constexpr (kUseTmaB) tma_load_bytes += sizeof(smem.b[0]);
    else legacy_load_bytes += sizeof(smem.b[0]);

    if constexpr (kIsGroupInputScale) {
      legacy_load_bytes += sizeof(smem.as[0]);
    }

    if constexpr (kIsGroupWeightScale) {
      if constexpr (kUseTmaBS) tma_load_bytes += sizeof(smem.bs[0]);
      else legacy_load_bytes += sizeof(smem.bs[0]);
    }

    if constexpr (kHasDynamicZeroPoint && (kIsGroupWeightScale || kIsFirst)) {
      if constexpr (kUseTmaBZP) tma_load_bytes += sizeof(smem.bzp[0]);
      else legacy_load_bytes += sizeof(smem.bzp[0]);
    }

    return {tma_load_bytes, legacy_load_bytes};
  }

  static constexpr uint2 get_channel_load_bytes() {
    uint32_t tma_load_bytes = 0;
    uint32_t legacy_load_bytes = 0;

    if constexpr (kIsChannelInputScale) {
      legacy_load_bytes += sizeof(smem.as_c);
    }

    if constexpr (kIsChannelWeightScale) {
      if constexpr (kUseTmaBS) tma_load_bytes += sizeof(smem.bs_c);
      else legacy_load_bytes += sizeof(smem.bs_c);
    }

    if constexpr (kHasBias) {
      if constexpr (kUseTmaBias) tma_load_bytes += sizeof(smem.bias);
      else legacy_load_bytes += sizeof(smem.bias);
    }

    if constexpr (kIsMoEDown) legacy_load_bytes += sizeof(smem.topk_weights);

    return {tma_load_bytes, legacy_load_bytes};
  }

public:
  static constexpr bool kHasFirstStageTmaMBarrier = get_stage_load_bytes<true>().x > 0;
  static constexpr bool kHasFirstStageCpAsyncMBarrier = get_stage_load_bytes<true>().y > 0;
  static constexpr bool kHasStageTmaMBarrier = get_stage_load_bytes().x > 0;
  static constexpr bool kHasStageCpAsyncMBarrier = get_stage_load_bytes().y > 0;
  static constexpr bool kHasChannelTmaMBarrier = get_channel_load_bytes().x > 0;
  static constexpr bool kHasChannelCpAsyncMBarrier = get_channel_load_bytes().y > 0;

  using LoaderA = G2SMemoryLoaderA<ProblemShape, BlockShape, PadShape, ElementA, PipelineConfig, MoEConfig>;
  using LoaderB = G2SMemoryLoaderB<ProblemShape, BlockShape, ElementA, ElementB, PipelineConfig, MoEConfig>;
  using LoaderAS = G2SMemoryLoaderAS<ProblemShape, BlockShape, PadShape, PipelineConfig, QuantParamConfig, MoEConfig>;
  using LoaderBS = G2SMemoryLoaderBS<ProblemShape, BlockShape, ElementBS, PipelineConfig, QuantParamConfig, MoEConfig>;
  using LoaderBZP = G2SMemoryLoaderBZP<ProblemShape, BlockShape, ElementB, PipelineConfig, QuantParamConfig, MoEConfig>;
  using LoaderBias = G2SMemoryLoaderBias<ProblemShape, BlockShape, PipelineConfig, MoEConfig>;
  using LoaderTopkWeights = G2SMemoryLoaderTopKWeights<BlockShape, MoEConfig>;

  SharedStorage &smem;
  LoaderA loader_a;
  LoaderB loader_b;
  LoaderAS loader_as;
  LoaderBS loader_bs;
  LoaderBZP loader_bzp;
  LoaderBias loader_bias;
  LoaderTopkWeights loader_topk_weights;
  uint32_t phase = 0;
  const uint32_t thread_id = threadIdx.x - kLoadThreadOffset;

  CUDA_INLINE
  ProducerPipeline(
      SharedStorage &smem,
      const void *void_ptr_a,
      const void *void_ptr_b,
      const void *void_ptr_as,
      const void *void_ptr_bs,
      const void *void_ptr_bzp,
      const void *void_ptr_bias,
      const void *void_ptr_topk_weights,
      uint32_t shape_m)
      : smem(smem),
        loader_a(void_ptr_a, smem.rd_row_index, shape_m),
        loader_b(void_ptr_b),
        loader_as(void_ptr_as, smem.rd_row_index, shape_m),
        loader_bs(void_ptr_bs),
        loader_bzp(void_ptr_bzp),
        loader_bias(void_ptr_bias),
        loader_topk_weights(void_ptr_topk_weights, smem.rd_row_index, shape_m) {

    if (thread_id == 0) {
      if constexpr (kUseTmaA) prefetch_tensor_map(void_ptr_a);
      if constexpr (kUseTmaB) prefetch_tensor_map(void_ptr_b);
      if constexpr (kUseTmaBS) prefetch_tensor_map(void_ptr_bs);
      if constexpr (kUseTmaBZP) prefetch_tensor_map(void_ptr_bzp);
      if constexpr (kUseTmaBias) prefetch_tensor_map(void_ptr_bias);
    }
    __syncwarp();
  }

  CUDA_INLINE void init_mbarrir() {
    if constexpr (kUseMBarrier) {
      phase = 0;

      uint32_t count;
      if (thread_id < kNumStages) {
        constexpr uint32_t cp_async_thread_count = kHasStageCpAsyncMBarrier ? kNumLoadThreads : 0;
        constexpr uint32_t tma_thread_count = kHasStageTmaMBarrier ? 1 : 0;
        count = cp_async_thread_count + tma_thread_count;
      } else if (thread_id == kNumStages) {
        constexpr uint32_t cp_async_thread_count = kHasFirstStageCpAsyncMBarrier ? kNumLoadThreads : 0;
        constexpr uint32_t tma_thread_count = kHasFirstStageTmaMBarrier ? 1 : 0;
        count = cp_async_thread_count + tma_thread_count;
      } else if (thread_id == kNumStages + 1) {
        constexpr uint32_t cp_async_thread_count = kHasChannelCpAsyncMBarrier ? kNumLoadThreads : 0;
        constexpr uint32_t tma_thread_count = kHasChannelTmaMBarrier ? 1 : 0;
        count = cp_async_thread_count + tma_thread_count;
      }

      if (thread_id < kNumStages + 2) __mbarrier_init(&smem.load_mbar[thread_id], count);

      if (thread_id < kNumStages) __mbarrier_init(&smem.math_mbar[thread_id], PipelineConfig::kNumMathThreads);
    }
  }

  template <bool kShouldAdvance = true, bool kIsFirst = false>
  CUDA_INLINE void load_stage(uint32_t stage_id, bool pred = true) {
    stage_id = stage_id % kNumStages;

    uint32_t mbar_index = kIsFirst ? kNumStages : stage_id;

    uint2 load_bytes;
    if (pred) {
      loader_a.load<kShouldAdvance>(smem.a[stage_id], &smem.load_mbar[mbar_index]);
      loader_b.load<kShouldAdvance>(smem.b[stage_id], &smem.load_mbar[mbar_index]);
      if constexpr (kIsGroupInputScale) {
        loader_as.load<kShouldAdvance>(smem.as[stage_id], &smem.load_mbar[mbar_index]);
      };
      if constexpr (kIsGroupWeightScale) {
        loader_bs.load<kShouldAdvance>(smem.bs[stage_id], &smem.load_mbar[mbar_index]);
      };
      if constexpr (kHasDynamicZeroPoint && (kIsGroupWeightScale || kIsFirst)) {
        loader_bzp.load<kShouldAdvance>(smem.bzp[stage_id], &smem.load_mbar[mbar_index]);
      }

      load_bytes = get_stage_load_bytes<kIsFirst>();
    } else {
      load_bytes = {0, 0};
    }

    if constexpr (kIsFirst) {
      commit_load<kHasFirstStageCpAsyncMBarrier, kHasFirstStageTmaMBarrier>(mbar_index, load_bytes);
    } else {
      commit_load<kHasStageCpAsyncMBarrier, kHasStageTmaMBarrier>(mbar_index, load_bytes);
    }
  }

  CUDA_INLINE void load_channel() {
    if constexpr (kIsChannelInputScale) loader_as.load(smem.as_c, &smem.load_mbar[kNumStages + 1]);
    if constexpr (kIsChannelWeightScale) loader_bs.load(smem.bs_c, &smem.load_mbar[kNumStages + 1]);
    if constexpr (kHasBias) loader_bias.load(smem.bias, &smem.load_mbar[kNumStages + 1]);
    if constexpr (kIsMoEDown) loader_topk_weights.load(smem.topk_weights);

    constexpr uint2 load_bytes = get_channel_load_bytes();
    if constexpr (load_bytes.x > 0 || load_bytes.y > 0) {
      commit_load<kHasChannelCpAsyncMBarrier, kHasChannelTmaMBarrier>(kNumStages + 1, load_bytes);
    }
  }

  template <bool kHasCpAsyncMBarrier, bool kHasTmaMBarrier>
  CUDA_INLINE void commit_load(uint32_t stage_id, uint2 load_bytes) {
    if constexpr (kUseMBarrier) {
      if constexpr (kHasCpAsyncMBarrier) {
        cp_async_commit_mbarrier(&smem.load_mbar[stage_id]);
      }
      if constexpr (kHasTmaMBarrier) {
        if (thread_id == 0) tma_commit_mbarrier(&smem.load_mbar[stage_id], load_bytes.x);
      }
    } else if constexpr (kUseCpAsync) {
      cp_async_commit_group();
    }
  }

  CUDA_INLINE void wait_stage(uint32_t stage_id) {
    mbarrier_wait(&smem.math_mbar[stage_id], phase);
    if (stage_id == kNumStages - 1) phase ^= 1;
  }

  CUDA_INLINE void seek(uint32_t expert_id, uint32_t m_block_id, uint32_t n_block_id, uint32_t k_block_id) {
    loader_a.seek(m_block_id, k_block_id);
    loader_b.seek(expert_id, n_block_id, k_block_id);
    loader_as.seek(m_block_id, k_block_id);
    loader_bs.seek(expert_id, n_block_id, k_block_id);
    loader_bzp.seek(expert_id, n_block_id, k_block_id);
    loader_bias.seek(expert_id, n_block_id);
    loader_topk_weights.seek();
  }
};


template <
    class SharedStorage,
    class PipelineConfig, class EpilogueConfig,
    class QuantParamConfig, class MoEConfig>
class ConsumerPipeline {
private:
  static constexpr uint32_t kNumThreads = PipelineConfig::kNumThreads;
  static constexpr uint32_t kNumMathThreads = PipelineConfig::kNumMathThreads;

  static constexpr bool kUseMBarrier = PipelineConfig::kUseMBarrier;
  static constexpr bool kUseCpAsync = PipelineConfig::kUseCpAsync;

  static constexpr bool kHasInputScale = QuantParamConfig::kHasInputScale;
  static constexpr bool kHasWeightScale = QuantParamConfig::kHasWeightScale;
  static constexpr bool kIsChannelInputScale = kHasInputScale && QuantParamConfig::kInputScaleGroupSize == 0;
  static constexpr bool kIsChannelWeightScale = kHasWeightScale && QuantParamConfig::kWeightScaleGroupSize == 0;
  static constexpr bool kHasBias = EpilogueConfig::kHasBias;
  static constexpr bool kIsMoEDown = MoEConfig::kIsMoEDown;
  static constexpr bool kHasChannelData = kIsChannelInputScale || kIsChannelWeightScale || kHasBias || kIsMoEDown;

  static constexpr uint32_t kNumStages = PipelineConfig::kNumStages;

public:
  SharedStorage &smem;
  uint32_t phase = 0;

  CUDA_INLINE
  ConsumerPipeline(SharedStorage &smem)
      : smem(smem) {
  }

  CUDA_INLINE void init_mbarrir() {
    if constexpr (kUseMBarrier) {
      phase = 0;
    }
  }

  template <bool kIsFirst = false>
  CUDA_INLINE void wait_stage(uint32_t stage_id) {
    stage_id = kIsFirst ? kNumStages : (stage_id % kNumStages);
    if constexpr (kUseMBarrier) {
      mbarrier_wait(&smem.load_mbar[stage_id], phase);
      __syncwarp();
      if (!kIsFirst && stage_id == 0) phase ^= 1;
    } else if constexpr (kUseCpAsync) {
      cp_async_wait_group<kNumStages - 2>();
      __syncthreads();
    } else {
      __syncthreads();
    }
  }

  CUDA_INLINE void wait_channel() {
    if constexpr (kHasChannelData) {
      if constexpr (kUseMBarrier) {
        mbarrier_wait(&smem.load_mbar[kNumStages + 1], 0);
        __syncwarp();
      } else if constexpr (kUseCpAsync) {
        cp_async_wait_group<0>();
        __syncthreads();
      } else {
        __syncthreads();
      }
    }
  }

  CUDA_INLINE void arrive(uint32_t stage_id) {
    __mbarrier_arrive(&smem.math_mbar[stage_id]);
  }
};
