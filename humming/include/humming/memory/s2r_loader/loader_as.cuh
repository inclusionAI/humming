#pragma once

#include <humming/utils/all.cuh>


template <
    class MmaOpClass,
    class BlockShape, class WarpShape,
    class ElementA,
    class PipelineConfig, class QuantParamConfig>
class S2RMemoryLoaderAS {
private:
  static constexpr bool kUseWgmma = MmaOpClass::kMmaType == MmaType::WGMMA;

  static constexpr uint32_t kNumThreads = PipelineConfig::kNumThreads;
  static constexpr bool kHasInputScale = QuantParamConfig::kHasInputScale;
  static constexpr bool kIsChannelScale = kHasInputScale && QuantParamConfig::kInputScaleGroupSize == 0;
  static constexpr bool kIsGroupScale = kHasInputScale && QuantParamConfig::kInputScaleGroupSize > 0;
  static constexpr uint32_t kGroupSize = kIsGroupScale ? QuantParamConfig::kInputScaleGroupSize : BlockShape::K;
  static constexpr uint32_t kPartMmaShapeK = 256 / ElementA::kBits;

  static constexpr uint32_t kBlockNumGroups = CEIL_DIV(BlockShape::K, kGroupSize);
  static constexpr uint32_t kNumGroups = CEIL_DIV(kPartMmaShapeK, kGroupSize);

  static constexpr uint32_t M_WARPS = BlockShape::M / WarpShape::M;
  static constexpr uint32_t N_WARPS = BlockShape::N / WarpShape::N;

  static constexpr uint32_t kLoadBytes = kNumGroups * 4;
  using LoadType = typename LoadTypeChooser<kLoadBytes>::Type;
  static constexpr uint32_t kLoadItersPerLine = kLoadBytes / sizeof(LoadType);
  static constexpr uint32_t kNumLinesPerBlock = kUseWgmma && kIsGroupScale ? 2 : 1;
  static constexpr uint32_t kSmemStride = kBlockNumGroups * 4 / sizeof(LoadType);

public:
  CUDA_INLINE
  void load(const int4 *smem_ptr, uint32_t *regs_ptr, int32_t iter_id) {
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t sub_row;

    if constexpr (kUseWgmma && kIsChannelScale) {
      sub_row = (lane_id % 4) * 2 + (lane_id % 8) / 4;
    } else if constexpr (kUseWgmma && kIsGroupScale) {
      sub_row = (lane_id % 4) * 2;
    } else if constexpr (!kUseWgmma) {
      sub_row = lane_id / 4;
    }

    uint32_t offset = 0;
    if constexpr (M_WARPS > 1) {
      offset += (warp_id / N_WARPS % M_WARPS) * (WarpShape::M * kSmemStride);
    }

    if constexpr (kGroupSize < BlockShape::K) {
      uint32_t k_index = (warp_id / (M_WARPS * N_WARPS)) * WarpShape::K + iter_id * kPartMmaShapeK;
      offset += (k_index / kGroupSize) * 4 / sizeof(LoadType);
    };

    LoadType *reg_ptr_load = reinterpret_cast<LoadType *>(regs_ptr);
    const LoadType *smem_ptr_load = reinterpret_cast<const LoadType *>(smem_ptr);

    PRAGMA_UNROLL
    for (uint32_t i = 0; i < WarpShape::M / 8; i++) {
      PRAGMA_UNROLL
      for (uint32_t j = 0; j < kNumLinesPerBlock; j++) {
        PRAGMA_UNROLL
        for (uint32_t k = 0; k < kLoadItersPerLine; k++) {
          uint32_t smem_idx = offset + (i * 8 + sub_row + j) * kSmemStride + k;
          uint32_t reg_idx = i * (kLoadItersPerLine * kNumLinesPerBlock) + j * kLoadItersPerLine + k;
          reg_ptr_load[reg_idx] = smem_ptr_load[smem_idx];
        }
      }
    }
  }
};
