#pragma once

#include <humming/utils/all.cuh>


template <
    class ProblemShape, class BlockShape,
    class PipelineConfig, class QuantParamConfig, class MoEConfig>
class G2SMemoryLoaderAS {
private:
  static constexpr bool kUseWarpSpec = PipelineConfig::kUseWarpSpec;
  static constexpr bool kUseCpAsync = PipelineConfig::kUseCpAsync;
  static constexpr bool kIsMoE = MoEConfig::kIsMoE;
  static constexpr uint32_t kNumLoadThreads = PipelineConfig::kNumLoadThreads;
  static constexpr uint32_t kLoadThreadOffset = PipelineConfig::kNumThreads - kNumLoadThreads;

  static constexpr bool kHasInputScale = QuantParamConfig::kHasInputScale;
  static constexpr bool kIsChannelScale = kHasInputScale && QuantParamConfig::kInputScaleGroupSize == 0;
  static constexpr bool kIsGroupScale = kHasInputScale && QuantParamConfig::kInputScaleGroupSize > 0;
  static constexpr uint32_t kGroupSize = kIsGroupScale ? QuantParamConfig::kInputScaleGroupSize : ProblemShape::K;

  static constexpr uint32_t kProblemNumGroups = CEIL_DIV(ProblemShape::K, kGroupSize);
  static constexpr uint32_t kNumGroups = CEIL_DIV(BlockShape::K, kGroupSize);
  static constexpr uint32_t kLoadsPerGroup = CEIL_DIV(kGroupSize, BlockShape::K);

  using LoadType = typename LoadTypeChooser<kNumGroups * 4>::Type;

public:
  const uint32_t thread_id = threadIdx.x - kLoadThreadOffset;
  const CUtensorMap *tensor_map_ptr;
  const uint32_t *gmem_ptr_raw;
  const uint32_t *gmem_ptr;

  const uint32_t *row_index_ptr;
  const uint32_t shape_m;
  uint32_t block_shape_m;
  uint32_t row_offset;
  uint32_t load_row_index;
  uint32_t col_offset = 0;
  uint32_t counter = 0;

  CUDA_INLINE
  G2SMemoryLoaderAS(const void *ptr, const uint32_t *row_index_ptr, uint32_t shape_m)
      : row_index_ptr(row_index_ptr), shape_m(shape_m) {
    gmem_ptr_raw = reinterpret_cast<const uint32_t *>(ptr);
  }

  template <bool kShouldAdvance = true>
  CUDA_INLINE void load(void *smem_ptr, void *mbar_ptr) {
    counter = kLoadsPerGroup != 1 ? (counter + 1) % kLoadsPerGroup : 0;
    load_legacy(smem_ptr);
    if constexpr (kShouldAdvance) advance();
  }

  CUDA_INLINE void load_legacy(void *smem_ptr) {
    if constexpr (!kIsMoE && kIsChannelScale) {
      uint32_t *smem_ptr_load = reinterpret_cast<uint32_t *>(smem_ptr);
      legacy_load_pred<kUseCpAsync>(gmem_ptr + thread_id, smem_ptr_load + thread_id, thread_id < BlockShape::M);
    } else {
      constexpr uint32_t kSmemStride = kNumGroups / (sizeof(LoadType) / 4);
      constexpr uint32_t kGmemStride = kProblemNumGroups / (sizeof(LoadType) / 4);

      PRAGMA_UNROLL
      for (uint32_t i = 0; i < CEIL_DIV(BlockShape::M, kNumLoadThreads); i++) {
        PRAGMA_UNROLL
        for (uint32_t j = 0; j < kSmemStride; j++) {
          uint32_t smem_offset = (i * BlockShape::M + thread_id) * kSmemStride + j;
          uint32_t smem_row = smem_offset / kSmemStride;
          uint32_t smem_col = smem_offset % kSmemStride;

          uint32_t gmem_row = kIsMoE ? load_row_index : smem_row;
          uint32_t gmem_offset = gmem_row * kGmemStride + smem_col;

          const LoadType *gmem_ptr_load = reinterpret_cast<const LoadType *>(gmem_ptr);
          LoadType *smem_ptr_load = reinterpret_cast<LoadType *>(smem_ptr);
          bool pred = kIsMoE ? (gmem_row < shape_m) : (smem_row < block_shape_m);
          legacy_load_pred<kUseCpAsync>(gmem_ptr_load + gmem_offset, smem_ptr_load + smem_offset, pred);
        }
      }
    }
  }

  CUDA_INLINE
  void advance() {
    if (kIsGroupScale && (kLoadsPerGroup == 1 || counter == 0)) {
      col_offset += kNumGroups;
      gmem_ptr += kNumGroups;
    }
  }

  CUDA_INLINE
  void seek(uint32_t m_block_id, uint32_t k_block_id) {
    if constexpr (kIsGroupScale) {
      if constexpr (BlockShape::K >= kGroupSize) {
        col_offset = k_block_id * kNumGroups;
      } else {
        col_offset = (k_block_id * BlockShape::K) / kGroupSize;
      }
    } else {
      col_offset = 0;
    }

    row_offset = m_block_id * BlockShape::M;
    block_shape_m = MIN((shape_m - row_offset), BlockShape::M);
    if constexpr (!kIsMoE) {
      gmem_ptr = gmem_ptr_raw + (m_block_id * (BlockShape::M * kProblemNumGroups) + col_offset);
    } else {
      gmem_ptr = gmem_ptr_raw;
    }

    if constexpr (kIsMoE) {
      constexpr uint32_t kSmemStride = kNumGroups / (sizeof(LoadType) / 4);
      uint32_t smem_row = threadIdx.x / kSmemStride;

      if (smem_row < BlockShape::M) {
        load_row_index = row_index_ptr[smem_row];
      } else {
        load_row_index = shape_m;
      }
    }
  }
};
