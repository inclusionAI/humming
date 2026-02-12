#pragma once

#include <humming/memory/s2r_loader/loader_a.cuh>
#include <humming/memory/s2r_loader/loader_as.cuh>
#include <humming/memory/s2r_loader/loader_b.cuh>
#include <humming/memory/s2r_loader/loader_bias.cuh>
#include <humming/memory/s2r_loader/loader_bs.cuh>
#include <humming/memory/s2r_loader/loader_bzp.cuh>
#include <humming/memory/s2r_loader/loader_topk_weights.cuh>

template <
    class SharedStorage, class MMA, class Epilogue,
    class BlockShape, class WarpShape,
    class ElementA, class ElementB, class ElementBS,
    class PipelineConfig, class EpilogueConfig,
    class QuantParamConfig, class MoEConfig>
class S2RMemoryPipeline {
private:
  static constexpr bool kUseWgmma = MMA::MmaOpClass::kMmaType == MmaType::WGMMA;
  static constexpr uint32_t kPartMmaShapeK = 256 / ElementA::kBits;
  static constexpr uint32_t kWarpItersK = WarpShape::K / kPartMmaShapeK;
  static constexpr uint32_t kNumStages = PipelineConfig::kNumStages;

  static constexpr bool kIsMoE = MoEConfig::kIsMoE;
  static constexpr bool kIsMoEDown = MoEConfig::kIsMoEDown;

  static constexpr bool kHasInputScale = QuantParamConfig::kHasInputScale;
  static constexpr bool kHasWeightScale = QuantParamConfig::kHasWeightScale;
  static constexpr bool kIsChannelInputScale = kHasInputScale && QuantParamConfig::kInputScaleGroupSize == 0;
  static constexpr bool kIsGroupInputScale = kHasInputScale && QuantParamConfig::kInputScaleGroupSize > 0;
  static constexpr bool kIsChannelWeightScale = kHasWeightScale && QuantParamConfig::kWeightScaleGroupSize == 0;
  static constexpr bool kIsGroupWeightScale = kHasWeightScale && QuantParamConfig::kWeightScaleGroupSize > 0;
  static constexpr bool kHasDynamicZeroPoint = QuantParamConfig::kHasDynamicZeroPoint;
  static constexpr bool kHasBias = EpilogueConfig::kHasBias;

  using MmaOpClass = class MMA::MmaOpClass;
  using LoaderA = S2RMemoryLoaderA<BlockShape, WarpShape, ElementA, PipelineConfig>;
  using LoaderB = S2RMemoryLoaderB<BlockShape, WarpShape, ElementA, ElementB, PipelineConfig>;
  using LoaderAS = S2RMemoryLoaderAS<MmaOpClass, BlockShape, WarpShape, ElementA, PipelineConfig, QuantParamConfig>;
  using LoaderBS = S2RMemoryLoaderBS<MmaOpClass, BlockShape, WarpShape, ElementA, ElementBS, PipelineConfig, QuantParamConfig>;
  using LoaderBZP = S2RMemoryLoaderBZP<BlockShape, WarpShape, ElementA, ElementB, PipelineConfig, QuantParamConfig>;
  using LoaderBias = S2RMemoryLoaderBias<MmaOpClass, BlockShape, WarpShape, PipelineConfig>;
  using LoaderTopkWeights = S2RMemoryLoaderTopKWeights<BlockShape, WarpShape>;

public:
  SharedStorage &smem;
  MMA &mma;
  Epilogue &epilogue;
  LoaderA loader_a;
  LoaderB loader_b;
  LoaderAS loader_as;
  LoaderBS loader_bs;
  LoaderBZP loader_bzp;
  LoaderBias loader_bias;
  LoaderTopkWeights loader_topk_weights;

  CUDA_INLINE
  S2RMemoryPipeline(SharedStorage &smem, MMA &mma, Epilogue &epilogue)
      : smem(smem), mma(mma), epilogue(epilogue) {
  }

  template <bool kIsFirst = false>
  CUDA_INLINE void load_stage_iter(uint32_t stage_id, uint32_t iter_id) {
    stage_id = (stage_id + iter_id / kWarpItersK) % kNumStages;
    iter_id = iter_id % kWarpItersK;
    uint32_t buffer_id = iter_id % 2;

    if constexpr (!kUseWgmma)
      loader_a.load(smem.a[stage_id], mma.regs_a_as_ptr(buffer_id), iter_id);
    loader_b.load(smem.b[stage_id], mma.regs_qb_as_ptr(buffer_id), iter_id);
    if constexpr (kIsGroupInputScale)
      loader_as.load(smem.as[stage_id], mma.arith.regs_as_as_ptr(buffer_id), iter_id);
    if constexpr (kIsGroupWeightScale)
      loader_bs.load(smem.bs[stage_id], mma.arith.regs_bs_as_ptr(buffer_id), iter_id);
    if constexpr (kHasDynamicZeroPoint && (kIsGroupWeightScale || kIsFirst))
      loader_bzp.load(smem.bzp[stage_id], mma.arith.regs_zp_as_ptr(buffer_id), iter_id);
  }

  CUDA_INLINE void load_channel(uint32_t slice_id) {
    if constexpr (kIsChannelInputScale) loader_as.load(smem.as_c, epilogue.arith.regs_as_as_ptr(), -1);
    if constexpr (kIsChannelWeightScale) loader_bs.load(smem.bs_c, epilogue.arith.regs_bs_as_ptr(), -1);
    if constexpr (kHasBias) loader_bias.load(smem.bias, epilogue.arith.regs_bias_as_ptr(), slice_id == 0);
    if constexpr (kIsMoEDown) loader_topk_weights.load(smem.topk_weights, epilogue.arith.regs_topk_weights_as_ptr());
  }
};
