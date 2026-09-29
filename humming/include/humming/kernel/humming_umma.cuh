#pragma once

#include <humming/scheduler.cuh>
#include <humming/utils/all.cuh>

#include <humming/arith/epilogue_arith.cuh>
#include <humming/arith/mainloop_arith.cuh>

#include <humming/epilogue/pipeline.cuh>
#include <humming/memory/g2s_pipeline.cuh>
#include <humming/memory/s2r_pipeline.cuh>
#include <humming/mma/all.cuh>

#include <humming/datatype/dequant.cuh>


template <bool kUseTma>
class KernelTensorParamType {
public:
  using Type = std::conditional_t<kUseTma, CUtensorMap const, void *const>;
};

CUDA_INLINE const void *param_to_ptr(const CUtensorMap &x) { return &x; }
CUDA_INLINE const void *param_to_ptr(void *const &x) { return x; }

template <
    class MmaOpClass,
    class ProblemShape, class BlockShape, class WarpShape, class PadShape,
    class ElementA, class ElementB, class ElementC, class ElementBS,
    class LayerConfig, class ComputeConfig, class TuningConfig>
__global__ __launch_bounds__(TuningConfig::kNumThreads, TuningConfig::kNumCtasPerSm) void humming(
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaA>::Type A,
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaB>::Type B,
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaC>::Type C,
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaAS>::Type AS,
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaAS2>::Type AS2,
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaBS>::Type BS,
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaBZP>::Type BZP,
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaBias>::Type Bias,
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaBS2>::Type BS2,
    const uint32_t *sorted_ids_ptr,
    const uint32_t *expert_ids_ptr,
    const uint32_t *num_tokens_padded_ptr,
    const uint32_t *expert_layout_ptr,
    CUtensorMap *tensor_map_buffer,
    int32_t *locks,
    uint32_t shape_m,
    uint32_t top_k,
    bool use_int64_expert_layout) {

  using Ctx = UmmaPipelineContext<
      MmaOpClass, ProblemShape, BlockShape, WarpShape, PadShape,
      ElementA, ElementB, ElementC, ElementBS, LayerConfig, ComputeConfig, TuningConfig>;
  using SharedStorage = typename Ctx::SharedStorage;
  using MainloopArith = MainloopArithmetic<Ctx>;
  using EpilogueArith = EpilogueArithmetic<Ctx>;
  using MMA = UMMA<Ctx, MainloopArith>;
  using Epilogue = EpiloguePipeline<Ctx, MMA, EpilogueArith>;
  using Producer = ProducerPipeline<Ctx>;
  using Consumer = ConsumerPipeline<Ctx>;
  constexpr uint32_t kNumStages = Ctx::kNumStages;
  constexpr uint32_t kNumOperandBuffers = MMA::kNumOperandBuffers;
  constexpr uint32_t kOutputGroups = MMA::kOutputGroups;
  constexpr uint32_t kDequantGroups = TuningConfig::kUmmaNumDequantWarpgroups;
  constexpr uint32_t kNumDequantThreads = Ctx::kUseUmmaSs ? 0 : 128 * kDequantGroups;
  static_assert(kDequantGroups == 1 || kDequantGroups == 2);
  static_assert(TuningConfig::kNumThreads == 256 + kNumDequantThreads);
  constexpr bool kCooperative = Ctx::kUmmaCtaGroupSize == 2;
  constexpr bool kHasChannelZeroPoint = Ctx::kHasZeroPoint && Ctx::kIsChannelWeightScale;
  constexpr uint32_t kChannelConsumers = kHasChannelZeroPoint ? 128 + kNumDequantThreads : 128;
  constexpr bool kSeparateOutputStorage = Ctx::kSmemReuseMode == SmemReuseMode::NONE;
  constexpr bool kNeedsEpilogueGate = !kSeparateOutputStorage || Consumer::kHasChannelData;

  extern __shared__ int4 shared_memory[];
  auto &smem = *reinterpret_cast<SharedStorage *>(shared_memory);
  MMA::init(smem);
  const KernelParams params{
      shape_m, top_k, use_int64_expert_layout,
      param_to_ptr(A), param_to_ptr(B), param_to_ptr(AS), param_to_ptr(AS2), param_to_ptr(BS),
      param_to_ptr(BZP), param_to_ptr(Bias), param_to_ptr(C), param_to_ptr(BS2),
      sorted_ids_ptr, expert_ids_ptr, num_tokens_padded_ptr, expert_layout_ptr, tensor_map_buffer, locks};
  Ctx ctx(smem, params);
  Scheduler<Ctx> scheduler(ctx);

  // An operand-ring wrap must retire the old stage before dequantization can
  // contribute to that stage's next ready phase.
  constexpr uint32_t kNumReadinessThreads = 128 - Ctx::kNumLoadThreads - 32;
  constexpr bool kEarlyWeightReuse = !Ctx::kUseUmmaSs && Ctx::kUseUmmaSplitLoads && kNumOperandBuffers <= kNumStages;
  if (ctx.is_load_thread()) Producer::template init_mbarrier<1, kChannelConsumers>(ctx);
  if (threadIdx.x < kNumStages) {
    __mbarrier_init(&smem.umma_operand_ready[threadIdx.x], (kNumDequantThreads + kNumReadinessThreads) * Ctx::kUmmaCtaGroupSize);
    if constexpr (Ctx::kUseUmmaSeparateInputScale && Ctx::kIsGroupInputScale)
      __mbarrier_init(&smem.umma_input_scale_ready[threadIdx.x], 1);
    if constexpr (Ctx::kUseUmmaSplitLoads) {
      __mbarrier_init(&smem.umma_weight_ready[threadIdx.x], 1);
      if constexpr (!Ctx::kUseUmmaSs)
        __mbarrier_init(&smem.umma_weight_free[threadIdx.x], kNumDequantThreads);
    }
  }
  if constexpr (!Ctx::kUseUmmaSs)
    if (threadIdx.x < kNumOperandBuffers) __mbarrier_init(&smem.umma_operand_free[threadIdx.x], 1);
  if (threadIdx.x == 0) {
    __mbarrier_init(&smem.umma_accumulator_ready, 1);
    __mbarrier_init(&smem.umma_accumulator_free, Ctx::kUmmaCtaGroupSize);
  }
  mbarrier_init_sync<kCooperative>();
  auto release_accumulator = [&]() {
    tcgen05_fence_before_thread_sync();
    ctx.sync_math_threads();
    if (ctx.math_thread_id() == 0) {
      if constexpr (kCooperative)
        mbarrier_arrive<true>(__cluster_map_shared_rank(&smem.umma_accumulator_free, 0));
      else mbarrier_arrive(&smem.umma_accumulator_free);
    }
  };
  auto arrive_operand_ready = [&](uint32_t stage) {
    if constexpr (kCooperative)
      mbarrier_arrive<true>(__cluster_map_shared_rank(&smem.umma_operand_ready[stage], 0));
    else mbarrier_arrive(&smem.umma_operand_ready[stage]);
  };
  if (ctx.is_math_thread()) release_accumulator();
  if constexpr (Ctx::kUsePdl) {
    griddepcontrol_wait();
    if (threadIdx.x == 0) griddepcontrol_launch_dependents();
  }
  // Every role advances the same stage ring, including across output tiles.
  uint32_t pipeline_stage = 0;
  uint32_t pipeline_phase = 0;
  uint32_t operand_step = 0;
  uint32_t tile_index = 0;
  auto advance_stage = [&]() {
    if (++pipeline_stage == kNumStages) {
      pipeline_stage = 0;
      pipeline_phase ^= 1;
    }
  };
  auto next_tile = [&]() {
    if constexpr (Ctx::kIsIndexedGemm) ctx.row_index_buffer = tile_index % 2;
    return scheduler.get_next_block();
  };
  if (ctx.is_load_thread()) {
    Producer producer(ctx);
    auto run_load_warp = [&](auto is_weight_warp) {
      while (true) {
        if constexpr (!kSeparateOutputStorage) producer.wait_math_epilogue();
        if (!next_tile()) break;
        producer.seek(scheduler.expert_id, scheduler.m_block_id, scheduler.n_block_id,
                      scheduler.k_block_id, scheduler.current_shape_m, scheduler.m_offset);
        producer.prefetch_stage();
        if constexpr (kSeparateOutputStorage) producer.wait_channel();
        producer.load_channel();
#pragma unroll 4
        for (uint32_t iter = 0; iter < scheduler.slice_iters; iter++) {
          auto *free_barrier = &smem.math_mbar[pipeline_stage];
          if constexpr (kEarlyWeightReuse && decltype(is_weight_warp)::value)
            free_barrier = &smem.umma_weight_free[pipeline_stage];
          mbarrier_wait(free_barrier, pipeline_phase ^ 1);
          if constexpr (!Ctx::kUseUmmaSplitLoads) producer.load_stage(pipeline_stage);
          else if constexpr (decltype(is_weight_warp)::value) producer.load_weight_stage(pipeline_stage);
          else producer.load_activation_stage(pipeline_stage);
          advance_stage();
        }
        if constexpr (Ctx::kIsIndexedGemm) {
          // Issuing this tile implies the preceding output has finished reading
          // its row indices. The producer may now reuse that index buffer.
          uint32_t last_stage = (pipeline_stage + kNumStages - 1) % kNumStages;
          uint32_t last_phase = pipeline_phase ^ (pipeline_stage == 0);
          mbarrier_wait(&smem.math_mbar[last_stage], last_phase);
        }
        tile_index++;
      }
    };
    if constexpr (Ctx::kUseUmmaSplitLoads) {
      if (threadIdx.x < 32) run_load_warp(CompileTimeConstant<1>{});
      else run_load_warp(CompileTimeConstant<0>{});
    } else run_load_warp(CompileTimeConstant<0>{});
  } else if (ctx.is_issuer_thread()) {
    if (blockIdx.x % Ctx::kUmmaCtaGroupSize == 0) {
      MainloopArith arith;
      MMA mma(ctx, arith);
      while (next_tile()) {
        mma.set_accumulator_tile(tile_index);
        mbarrier_wait(&smem.umma_accumulator_free, tile_index % 2);
        tcgen05_fence_after_thread_sync();
#pragma unroll 4
        for (uint32_t iter = 0; iter < scheduler.slice_iters; iter++) {
          mbarrier_wait(&smem.umma_operand_ready[pipeline_stage], pipeline_phase);
          if constexpr (Ctx::kUseUmmaCooperativeTma)
            mbarrier_wait(&smem.load_mbar[pipeline_stage], pipeline_phase);
          uint32_t buffer = operand_step % kNumOperandBuffers;
          tcgen05_fence_after_thread_sync();
          if (tcgen05_elect_leader()) {
            PRAGMA_UNROLL
            for (uint32_t group = 0; group < kOutputGroups; group++) {
              ctx.math_group = group;
              mma.issue(pipeline_stage, buffer, iter == 0, scheduler.k_block_id + iter);
            }
            // SS copies and consumes scales on the same tcgen05 issue stream.
            // Only TS needs to return TMEM buffers to a separate producer.
            if constexpr (!Ctx::kUseUmmaSs)
              tcgen05_commit<Ctx::kUmmaCtaGroupSize>(cast_smem_ptr_to_uint(&smem.umma_operand_free[buffer]));
            tcgen05_commit<Ctx::kUmmaCtaGroupSize>(cast_smem_ptr_to_uint(&smem.math_mbar[pipeline_stage]));
            if (iter + 1 == scheduler.slice_iters)
              tcgen05_commit<Ctx::kUmmaCtaGroupSize>(cast_smem_ptr_to_uint(&smem.umma_accumulator_ready));
          }
          operand_step++;
          advance_stage();
        }
        tile_index++;
      }
    }
  } else if (threadIdx.x < 128) {
    // SS operands need only the scale stores; TS joins A with the dequantization WG.
    MainloopArith arith;
    MMA mma(ctx, arith);
    while (next_tile()) {
#pragma unroll 4
      for (uint32_t iter = 0; iter < scheduler.slice_iters; iter++) {
        if constexpr (Ctx::kUseUmmaSeparateInputScale) {
          if constexpr (Ctx::kIsGroupInputScale)
            mbarrier_wait(&smem.umma_input_scale_ready[pipeline_stage], pipeline_phase);
          else
            mbarrier_wait(&smem.math_mbar[pipeline_stage], pipeline_phase ^ 1);
        } else mbarrier_wait(&smem.load_mbar[pipeline_stage], pipeline_phase);
        if constexpr (!Ctx::kUseTmaA) tma_fence_async_shared();
        if constexpr (Ctx::kUseUmmaSs) {
          if constexpr (Ctx::kUseUmmaSplitLoads && !Ctx::kUseUmmaCooperativeTma)
            mbarrier_wait(&smem.umma_weight_ready[pipeline_stage], pipeline_phase);
          uint32_t buffer = operand_step % kNumOperandBuffers;
          mma.set_operand_buffer(buffer);
          PRAGMA_UNROLL
          for (uint32_t group = 0; group < kOutputGroups; group++) {
            ctx.math_group = group;
            mma.store_weight_scales(pipeline_stage, scheduler.k_block_id + iter, scheduler.n_block_id);
          }
          mma.store_input_scales(pipeline_stage, buffer, scheduler.k_block_id + iter, scheduler.m_offset);
          __syncwarp();
          tma_fence_async_shared();
          operand_step++;
        }
        if constexpr (Ctx::kUseUmmaSeparateInputScale && !Ctx::kUseUmmaCooperativeTma)
          mbarrier_wait(&smem.load_mbar[pipeline_stage], pipeline_phase);
        arrive_operand_ready(pipeline_stage);
        advance_stage();
      }
      tile_index++;
    }
  } else {
    MainloopArith mainloop_arith;
    EpilogueArith epilogue_arith;
    MMA mma(ctx, mainloop_arith);
    Epilogue epilogue(ctx, epilogue_arith);
    S2RMemoryPipeline<Ctx, MMA, Epilogue> s2r(ctx, mma, epilogue);
    Consumer consumer(ctx);
    if constexpr (kNeedsEpilogueGate) {
      if (ctx.is_math_thread() || kHasChannelZeroPoint) consumer.arrive(kNumStages);
    }
    while (next_tile()) {
      s2r.seek(scheduler.m_offset);
      if (ctx.is_dequant_thread()) {
        constexpr bool kSplitN = kOutputGroups >= kDequantGroups;
        constexpr uint32_t kGroupStride = kSplitN ? kDequantGroups : 1;
        constexpr uint32_t kGroupsPerWarpgroup = kOutputGroups / kGroupStride;
        constexpr uint32_t kFragmentsPerGroup = Ctx::kWarpIters / (kSplitN ? 1 : kDequantGroups);
        static_assert(kSplitN || Ctx::kWarpIters % kDequantGroups == 0);
        uint32_t first_group = kSplitN && kDequantGroups > 1 ? ctx.dequant_group_id() : 0;
        uint32_t first_fragment = kSplitN ? 0 : ctx.dequant_group_id() * kFragmentsPerGroup;
        auto convert_weight_stage = [&](uint32_t stage, uint32_t buffer, uint32_t iter, auto wait_for_operand) {
          mma.set_operand_buffer(buffer);
          PRAGMA_UNROLL
          for (uint32_t group_index = 0; group_index < kGroupsPerWarpgroup; group_index++) {
            uint32_t group = first_group + group_index * kGroupStride;
            ctx.math_group = group;
            constexpr uint32_t kPreparedFragments = MIN(kFragmentsPerGroup, 4);
            constexpr uint32_t kFragmentWords = sizeof(mma.regs_b[0]) / sizeof(uint32_t);
            PRAGMA_UNROLL
            for (uint32_t fragment_index = 0; fragment_index < kFragmentsPerGroup; fragment_index += kPreparedFragments) {
              uint32_t fragment = first_fragment + fragment_index;
              // Prepare a wider store while UMMA still owns the TMEM buffer.
              uint32_t prepared[kPreparedFragments][kFragmentWords];
              PRAGMA_UNROLL
              for (uint32_t part = 0; part < kPreparedFragments; part++) {
                uint32_t register_buffer = (fragment + part) % 2;
                s2r.template load_stage_iter<true>(stage, fragment + part);
                mma.transform_b(register_buffer, fragment + part);
                const uint32_t *values = reinterpret_cast<const uint32_t *>(mma.regs_b[register_buffer]);
                PRAGMA_UNROLL
                for (uint32_t word = 0; word < kFragmentWords; word++) {
                  prepared[part][word] = values[word];
                }
              }
              if (group_index == 0 && fragment_index == 0) wait_for_operand();
              if (fragment_index == 0) mma.store_weight_scales(stage, scheduler.k_block_id + iter, scheduler.n_block_id);
              mma.store_b(prepared, fragment);
              if constexpr (kEarlyWeightReuse) {
                if (group_index + 1 == kGroupsPerWarpgroup && fragment_index + kPreparedFragments >= kFragmentsPerGroup) {
                  // Include scale reads and native operand stores before
                  // allowing the producer to overwrite raw weights.
                  __mbarrier_arrive(&smem.umma_weight_free[stage]);
                }
              }
            }
          }
        };
        if constexpr (kHasChannelZeroPoint) consumer.wait_channel();
#pragma unroll 4
        for (uint32_t iter = 0; iter < scheduler.slice_iters; iter++) {
          uint32_t buffer = operand_step % kNumOperandBuffers;
          auto wait_for_operand = [&]() {
            uint32_t free_phase = ((operand_step / kNumOperandBuffers) % 2) ^ 1;
            mbarrier_wait(&smem.umma_operand_free[buffer], free_phase);
            tcgen05_fence_after_thread_sync();
          };
          auto *ready_barrier = &smem.load_mbar[pipeline_stage];
          if constexpr (Ctx::kUseUmmaSplitLoads) ready_barrier = &smem.umma_weight_ready[pipeline_stage];
          mbarrier_wait(ready_barrier, pipeline_phase);
          if constexpr (Ctx::kUseUmmaSplitLoads && Ctx::kIsGroupInputScale && !Ctx::kUseBlockScaledMma)
            mbarrier_wait(&smem.load_mbar[pipeline_stage], pipeline_phase);
          // With three loading warps, the combined load barrier already covers A.
          if constexpr (kNumReadinessThreads == 0 && !Ctx::kUseTmaA) tma_fence_async_shared();
          convert_weight_stage(pipeline_stage, buffer, iter, wait_for_operand);
          if constexpr (Ctx::kUseBlockScaledMma) {
            if constexpr (Ctx::kUseUmmaSplitLoads)
              mbarrier_wait(&smem.load_mbar[pipeline_stage], pipeline_phase);
            PRAGMA_UNROLL
            for (uint32_t group_index = 0; group_index < kGroupsPerWarpgroup; group_index++) {
              ctx.math_group = first_group + group_index * kGroupStride;
              mma.store_input_scales(pipeline_stage, buffer, scheduler.k_block_id + iter, scheduler.m_offset);
            }
          }
          tcgen05_wait_st();
          tcgen05_fence_before_thread_sync();
          arrive_operand_ready(pipeline_stage);
          operand_step++;
          advance_stage();
        }
        if constexpr (kHasChannelZeroPoint) consumer.arrive(kNumStages);
      } else {
        mma.set_accumulator_tile(tile_index);
        epilogue.load_secondary_input_scale(scheduler.m_block_id, scheduler.current_shape_m, scheduler.m_offset);
        mbarrier_wait(&smem.umma_accumulator_ready, tile_index % 2);
        tcgen05_fence_after_thread_sync();
        if constexpr (Ctx::kUseTmaC && !Ctx::kUmmaOutputChunkRows) tma_wait_store_group<0, true>();
        ctx.sync_math_threads();
        consumer.wait_channel();
        epilogue.set_streamk_state(scheduler.slice_count, scheduler.slice_id, scheduler.locks_offset);
        if (scheduler.slice_count > 1) epilogue.acquire_gmem_barrier();
        PRAGMA_UNROLL
        for (uint32_t group = 0; group < kOutputGroups; group++) {
          ctx.math_group = group;
          epilogue.seek(scheduler.expert_id, scheduler.m_block_id, scheduler.n_block_id,
                        scheduler.current_shape_m, scheduler.m_offset);
          s2r.load_channel(scheduler.slice_id);
          if constexpr (Consumer::kHasChannelData && kSeparateOutputStorage) {
            if (group + 1 == kOutputGroups) {
              ctx.sync_math_threads();
              consumer.arrive(kNumStages);
            }
          }
          auto write_chunk = [&](uint32_t first_row, uint32_t rows, uint32_t buffer_offset) {
            if constexpr (Ctx::kUmmaOutputChunkRows)
              epilogue.gmem_writer.write_chunk(scheduler.slice_id, scheduler.slice_count, first_row, rows, buffer_offset);
          };
          epilogue.smem_writer.write_umma(mma, scheduler.slice_id, scheduler.slice_count, write_chunk);
        }
        if constexpr (!Ctx::kUmmaOutputChunkRows && !Ctx::kIsIndexedGemm) release_accumulator();
        if constexpr (Ctx::kUseTmaC) tma_fence_async_shared();
        ctx.sync_math_threads();
        if constexpr (!Ctx::kUmmaOutputChunkRows)
          epilogue.gmem_writer.write(scheduler.slice_id, scheduler.slice_count, 0);
        if constexpr (Ctx::kIsIndexedGemm) release_accumulator();
        if (scheduler.slice_count > 1) epilogue.release_gmem_barrier();
        if constexpr (!kSeparateOutputStorage) {
          if constexpr (Ctx::kUseTmaC) tma_wait_store_group<0, true>();
          ctx.sync_math_threads();
          consumer.arrive(kNumStages);
        }
      }
      tile_index++;
    }
  }
  if constexpr (Ctx::kUseTmaC) {
    if (ctx.is_math_thread()) tma_wait_store_group<0>();
  }
  if constexpr (kCooperative)
    asm volatile("barrier.cluster.arrive.aligned; barrier.cluster.wait.aligned;" ::: "memory");
  else __syncthreads();
  MMA::dealloc(smem);
}
