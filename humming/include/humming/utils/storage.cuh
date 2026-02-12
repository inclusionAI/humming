#pragma once

#include <humming/utils/base.cuh>


template <
    class MmaOpClass,
    class BlockShape, class WarpShape,
    class ElementA, class ElementB, class ElementBS,
    class PipelineConfig, class EpilogueConfig,
    class QuantParamConfig, class MoEConfig>
struct SharedStorage {
private:
  static constexpr bool kHasInputScale = QuantParamConfig::kHasInputScale;
  static constexpr bool kHasWeightScale = QuantParamConfig::kHasWeightScale;
  static constexpr bool kHasChannelInputScale = kHasInputScale && QuantParamConfig::kInputScaleGroupSize == 0;
  static constexpr bool kHasGroupInputScale = kHasInputScale && QuantParamConfig::kInputScaleGroupSize > 0;
  static constexpr bool kHasChannelWeightScale = kHasWeightScale && QuantParamConfig::kWeightScaleGroupSize == 0;
  static constexpr bool kHasGroupWeightScale = kHasWeightScale && QuantParamConfig::kWeightScaleGroupSize > 0;
  static constexpr bool kHasDynamicZeroPoint = QuantParamConfig::kHasDynamicZeroPoint;

public:
  static constexpr uint32_t kNumStages = PipelineConfig::kNumStages;
  static constexpr uint32_t kPartMmaShapeK = 256 / ElementA::kBits;
  static constexpr uint32_t kNumWarpsDimK = BlockShape::K / WarpShape::K;
  static constexpr uint32_t kMmaCTypeBits = MmaOpClass::kCTypeBits;
  static constexpr uint32_t M_WARPS = (BlockShape::M / WarpShape::M);
  static constexpr uint32_t kWarpReduceSize = M_WARPS * 16 * BlockShape::N * kMmaCTypeBits / 128 * (kNumWarpsDimK / 2);
  static constexpr uint32_t kBlockOutputSize = BlockShape::M * BlockShape::N / 2 / 4;

  static constexpr uint32_t kSmemStrideA = BlockShape::K * ElementA::kBits / 32 / 4;
  static constexpr uint32_t kSmemStrideB = BlockShape::N * kPartMmaShapeK * ElementB::kBits / 32 / 4;
  static constexpr uint32_t kSmemStrideBS = BlockShape::N * ElementBS::kBits / 32 / 4;
  static constexpr uint32_t kSmemStrideBZP = BlockShape::N * MAX(4, static_next_power_of_2(ElementB::kBits)) / 32 / 4;
  static constexpr uint32_t kSmemStrideBias = BlockShape::N * 16 / 32 / 4;

  static constexpr uint32_t kGroupSzieA = QuantParamConfig::kInputScaleGroupSize;
  static constexpr uint32_t kGroupSzieB = QuantParamConfig::kWeightScaleGroupSize;
  static constexpr uint32_t kNumGroupsA = kHasGroupInputScale ? CEIL_DIV(BlockShape::K, kGroupSzieA) : 0;
  static constexpr uint32_t kNumGroupsB = kHasGroupWeightScale ? CEIL_DIV(BlockShape::K, kGroupSzieB) : 0;

  static constexpr uint32_t kStageSizeA = BlockShape::M * kSmemStrideA;
  static constexpr uint32_t kStageSizeB = BlockShape::K / kPartMmaShapeK * kSmemStrideB;
  static constexpr uint32_t kStageSizeAS = kNumGroupsA * BlockShape::M / 4;
  static constexpr uint32_t kStageSizeBS = kNumGroupsB * kSmemStrideBS;
  static constexpr uint32_t kStageSizeBZP = kNumGroupsB * kSmemStrideBZP;

  static constexpr uint32_t kChannelSizeAS = kHasChannelInputScale ? BlockShape::M / 4 : 0;
  static constexpr uint32_t kChannelSizeBS = kHasChannelWeightScale ? kSmemStrideBS : 0;
  static constexpr uint32_t kChannelSizeBZP = (kHasChannelWeightScale && kHasDynamicZeroPoint) ? kSmemStrideBZP : 0;
  static constexpr uint32_t kBiasSize = EpilogueConfig::kHasBias ? kSmemStrideBias : 0;

  static constexpr bool kUseWarpSpec = PipelineConfig::kUseWarpSpec;
  static constexpr bool kUseMBarrier = PipelineConfig::kUseMBarrier;
  static constexpr bool kIsMoE = MoEConfig::kIsMoE;
  static constexpr bool kIsMoEDown = MoEConfig::kIsMoEDown;

  union alignas(128) {
    struct {
      int4 a[kNumStages][kStageSizeA];
      int4 b[kNumStages][kStageSizeB];
      int4 as[kNumStages][kStageSizeAS];
      int4 bs[kNumStages][kStageSizeBS];
      int4 bzp[kHasChannelWeightScale ? 1 : kNumStages][kHasChannelWeightScale ? kChannelSizeBZP : kStageSizeBZP];
      int4 bs_c[kChannelSizeBS];
      int4 bias[kBiasSize];
      int4 as_c[kChannelSizeAS];
    };
    int4 reduce[MAX(kWarpReduceSize, kBlockOutputSize)];
  };

  uint32_t rd_row_index[(kIsMoE && !kIsMoEDown) ? BlockShape::M : 0];
  uint32_t wr_row_index[kIsMoE ? BlockShape::M : 0];
  uint32_t topk_weights[kIsMoE ? BlockShape::M : 0];

  uint64_t load_mbar[kUseMBarrier ? (kNumStages + 2) : 0];
  uint64_t math_mbar[kUseWarpSpec ? kNumStages : 0];
};
