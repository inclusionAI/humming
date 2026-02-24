#pragma once

#include <humming/scheduler.cuh>
#include <humming/utils/all.cuh>

#include <humming/arith/epilogue_arith.cuh>
#include <humming/arith/mainloop_arith.cuh>

#include <humming/epilogue/pipeline.cuh>
#include <humming/memory/g2s_pipeline.cuh>
#include <humming/memory/s2r_pipeline.cuh>
#include <humming/mma/wgmma.cuh>
#include <humming/mma/wmma.cuh>

#include <humming/datatype/dequant.cuh>


template <bool kUseTma>
class KernelTensorParamType {
public:
  using Type = std::conditional_t<kUseTma, CUtensorMap const, void *const>;
};


template <
    class MmaOpClass,
    class ProblemShape, class BlockShape, class WarpShape, class PadShape,
    class ElementA, class ElementB, class ElementC, class ElementBS,
    class SchedulerConfig, class PipelineConfig, class EpilogueConfig,
    class QuantParamConfig, class MoEConfig>
__global__ __launch_bounds__(PipelineConfig::kNumThreads, PipelineConfig::kNumCtasPerSm) void humming(
    const __grid_constant__ typename KernelTensorParamType<PipelineConfig::kUseTmaA>::Type A,
    const __grid_constant__ typename KernelTensorParamType<PipelineConfig::kUseTmaB>::Type B,
    const __grid_constant__ typename KernelTensorParamType<PipelineConfig::kUseTmaC>::Type C,
    const uint32_t *AS,
    const __grid_constant__ typename KernelTensorParamType<PipelineConfig::kUseTmaBS>::Type BS,
    const __grid_constant__ typename KernelTensorParamType<PipelineConfig::kUseTmaBZP>::Type BZP,
    const __grid_constant__ typename KernelTensorParamType<PipelineConfig::kUseTmaBias>::Type Bias,
    const uint32_t *GS,
    const uint32_t *topk_weights_ptr,
    const uint32_t *sorted_token_ids_ptr,
    const uint32_t *expert_ids_ptr,
    const uint32_t *num_tokens_post_padded_ptr,
    int32_t *locks,
    uint32_t shape_m) {

  constexpr uint32_t kNumThreads = PipelineConfig::kNumThreads;
  constexpr uint32_t kNumStages = PipelineConfig::kNumStages;

  using SharedStorage = SharedStorage<
      MmaOpClass, BlockShape, WarpShape, ElementA, ElementB, ElementBS,
      PipelineConfig, EpilogueConfig, QuantParamConfig, MoEConfig>;
  using Scheduler = Scheduler<
      SharedStorage, ProblemShape, BlockShape,
      SchedulerConfig, PipelineConfig, QuantParamConfig, MoEConfig>;
  using ProducerPipeline = ProducerPipeline<
      SharedStorage, ProblemShape, BlockShape, PadShape, ElementA, ElementB, ElementBS,
      PipelineConfig, EpilogueConfig, QuantParamConfig, MoEConfig>;
  using ConsumerPipeline = ConsumerPipeline<SharedStorage, PipelineConfig, EpilogueConfig, QuantParamConfig, MoEConfig>;
  using MainloopArithmetic = MainloopArithmetic<
      MmaOpClass, BlockShape, WarpShape,
      ElementA, ElementB, ElementC, ElementBS, QuantParamConfig>;
  using EpilogueArithmetic = EpilogueArithmetic<
      MmaOpClass, BlockShape, WarpShape,
      ElementA, ElementB, ElementC, ElementBS, SchedulerConfig, EpilogueConfig, QuantParamConfig, MoEConfig>;
  using WMMA = WMMA<MmaOpClass, SharedStorage, MainloopArithmetic, WarpShape, ElementA, ElementB, QuantParamConfig>;
  using WGMMA = WGMMA<MmaOpClass, SharedStorage, MainloopArithmetic, BlockShape, WarpShape, ElementA, ElementB, QuantParamConfig>;
  using MMA = std::conditional_t<MmaOpClass::kMmaType == MmaType::WGMMA, WGMMA, WMMA>;
  using Epilogue = EpiloguePipeline<
      MmaOpClass, SharedStorage, EpilogueArithmetic, ProblemShape, BlockShape, WarpShape, PadShape,
      ElementA, ElementC, SchedulerConfig, PipelineConfig, EpilogueConfig, QuantParamConfig, MoEConfig>;
  using S2RMemoryPipeline = S2RMemoryPipeline<
      SharedStorage, MMA, Epilogue, BlockShape, WarpShape, ElementA, ElementB, ElementBS,
      PipelineConfig, EpilogueConfig, QuantParamConfig, MoEConfig>;

  extern __shared__ int4 shared_memory[];
  auto &smem = *reinterpret_cast<SharedStorage *>(shared_memory);

  auto pa = [&]() {if constexpr (PipelineConfig::kUseTmaA) return &A; else return A; };
  auto pb = [&]() {if constexpr (PipelineConfig::kUseTmaB) return &B; else return B; };
  auto pc = [&]() {if constexpr (PipelineConfig::kUseTmaC) return &C; else return C; };
  auto pas = [&]() { return AS; };
  auto pbs = [&]() {if constexpr (PipelineConfig::kUseTmaBS) return &BS; else return BS; };
  auto pbzp = [&]() {if constexpr (PipelineConfig::kUseTmaBZP) return &BZP; else return BZP; };
  auto pbias = [&]() {if constexpr (PipelineConfig::kUseTmaBias) return &Bias; else return Bias; };
  uint32_t block_padded_shape_m = MoEConfig::kIsMoE ? num_tokens_post_padded_ptr[0] : shape_m;
  auto scheduler = Scheduler(smem, sorted_token_ids_ptr, expert_ids_ptr, block_padded_shape_m);
  auto mainloop_arith = MainloopArithmetic();
  auto epilogue_arith = EpilogueArithmetic();
  auto mma = MMA(smem, mainloop_arith);
  auto epilogue = Epilogue(smem, pc(), epilogue_arith, GS, locks, shape_m);
  auto producer = ProducerPipeline(smem, pa(), pb(), pas(), pbs(), pbzp(), pbias(), topk_weights_ptr, shape_m);
  auto consumer = ConsumerPipeline(smem);
  auto s2r_pipe = S2RMemoryPipeline(smem, mma, epilogue);

  while (scheduler.get_next_block()) {
    mma.zero_accum();
    __syncthreads();

    uint32_t &slice_iters = scheduler.slice_iters;
    producer.init_mbarrir();
    consumer.init_mbarrir();
    __syncthreads();
    producer.seek(scheduler.expert_id, scheduler.m_block_id, scheduler.n_block_id, scheduler.k_block_id);
    epilogue.seek(scheduler.expert_id, scheduler.m_block_id, scheduler.n_block_id);
    epilogue.set_streamk_state(scheduler.slice_count, scheduler.slice_id, scheduler.locks_offset);

    if constexpr (PipelineConfig::kUseTmaC) tma_wait_store_group<0, true>();
    producer.load_stage<true, true>(0);
    PRAGMA_UNROLL
    for (uint32_t stage_id = 1; stage_id < kNumStages - 1; stage_id++) {
      producer.load_stage(stage_id, stage_id < slice_iters);
    };

    consumer.wait_stage<true>(kNumStages);
    s2r_pipe.load_stage_iter<true>(0, 0);
    mma.transform_b(0);

    while (slice_iters) {
      PRAGMA_UNROLL
      for (uint32_t stage_id = 0; stage_id < kNumStages; stage_id++) {
        if (slice_iters == 1) producer.load_channel();
        constexpr uint32_t kPartMmaShapeK = 256 / ElementA::kBits;
        constexpr uint32_t warp_k_iters = WarpShape::K / kPartMmaShapeK;

        PRAGMA_UNROLL
        for (uint32_t warp_k_iter_id = 0; warp_k_iter_id < warp_k_iters; warp_k_iter_id++) {
          s2r_pipe.load_stage_iter(stage_id, warp_k_iter_id + 1);
          mma.run(stage_id, warp_k_iter_id);
          if (warp_k_iter_id == warp_k_iters - 2) {
            producer.load_stage(stage_id + kNumStages - 1, slice_iters >= kNumStages);
            consumer.wait_stage((stage_id + 1) % kNumStages);
            __syncthreads();
          }

          mma.transform_b((warp_k_iter_id + 1) % 2);
        }

        slice_iters--;
        if (!slice_iters) break;
      };
    };

    consumer.wait_channel();
    s2r_pipe.load_channel(scheduler.slice_id);
    epilogue.call(mma.final_regs_c_as_ptr());
  }
};
