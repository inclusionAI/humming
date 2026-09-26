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
  static_assert(TuningConfig::kNumThreads == 384);
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

  if (ctx.is_load_thread()) Producer::template init_mbarrier<1>(ctx);
  if (threadIdx.x < kNumOperandBuffers) {
    __mbarrier_init(&smem.umma_operand_ready[threadIdx.x], 128);
    __mbarrier_init(&smem.umma_operand_free[threadIdx.x], 1);
  }
  if constexpr (Ctx::kUseUmmaSplitLoads) {
    if (threadIdx.x < kNumStages + 1) __mbarrier_init(&smem.umma_weight_ready[threadIdx.x], 1);
    if (threadIdx.x < kNumStages) __mbarrier_init(&smem.umma_weight_free[threadIdx.x], 128);
  }
  mbarrier_init_sync<false>();
  if (threadIdx.x < 128) asm volatile("setmaxnreg.dec.sync.aligned.u32 40;" ::: "memory");
  if (ctx.is_load_thread()) {
    Producer producer(ctx);
    auto preload_tile = [&]() {
      producer.seek(scheduler.expert_id, scheduler.m_block_id, scheduler.n_block_id, scheduler.k_block_id, scheduler.current_shape_m, scheduler.m_offset);
      producer.prefetch_stage();
      producer.template load_stage<true, true>(0);
      PRAGMA_UNROLL
      for (uint32_t stage = 1; stage < kNumStages; stage++) {
        producer.load_stage(stage, stage < scheduler.slice_iters);
      }
    };
    bool pdl_waited = false;
    uint32_t next_row_index_buffer = 0;
    while (true) {
      if constexpr (!kSeparateOutputStorage) producer.wait_math_epilogue();
      if constexpr (Ctx::kIsIndexedGemm) {
        ctx.row_index_buffer = next_row_index_buffer;
      }
      if (!scheduler.get_next_block()) break;
      if constexpr (Ctx::kUsePdl) {
        if (!pdl_waited) {
          griddepcontrol_wait();
          if (threadIdx.x == 0) griddepcontrol_launch_dependents();
          pdl_waited = true;
        }
      }
      if constexpr (Ctx::kIsIndexedGemm) next_row_index_buffer ^= 1;
      preload_tile();
      // Channel storage is separate from the stages, so operand loads may
      // proceed before the previous epilogue releases its channel data.
      if constexpr (kSeparateOutputStorage) producer.wait_channel();
      producer.load_channel();
      if constexpr (Ctx::kUseUmmaSplitLoads) {
        // Specialize each producer loop once instead of branching on the
        // warp role at every stage. Each warp owns its barrier phases.
        auto run_load_warp = [&](auto is_weight_warp) {
          for (uint32_t base_iter = 0; base_iter < scheduler.slice_iters; base_iter += 2 * kNumStages) {
            static_for<0, 2 * kNumStages>([&](auto step) {
              uint32_t iter = base_iter + decltype(step)::value;
              if (iter >= scheduler.slice_iters) return;

              constexpr uint32_t stage = decltype(step)::value % kNumStages;
              if constexpr (decltype(is_weight_warp)::value) {
                if (iter + kNumStages < scheduler.slice_iters) {
                  producer.wait_weight_consumed(stage);
                  producer.load_weight_stage(stage);
                }
              } else {
                producer.wait_activation_consumed(stage);
                if (iter + kNumStages < scheduler.slice_iters) producer.load_activation_stage(stage);
              }
            });
          }
          if constexpr (decltype(is_weight_warp)::value) {
            PRAGMA_UNROLL
            for (uint32_t stage = 0; stage < kNumStages; stage++) {
              if (stage < scheduler.slice_iters) producer.wait_weight_consumed(stage);
            }
          }
        };
        if (threadIdx.x < 32) run_load_warp(std::true_type{});
        else if (threadIdx.x < 64) run_load_warp(std::false_type{});
        // Retire both operand streams before starting another tile; otherwise
        // weight readiness can advance into the next tile's barrier phases.
        ctx.sync_load_threads();
      } else {
        for (uint32_t base_iter = 0; base_iter < scheduler.slice_iters; base_iter += 2 * kNumStages) {
          static_for<0, 2 * kNumStages>([&](auto step) {
            uint32_t iter = base_iter + decltype(step)::value;
            if (iter >= scheduler.slice_iters) return;

            constexpr uint32_t stage = decltype(step)::value % kNumStages;
            producer.wait_stage(stage);
            if (iter + kNumStages < scheduler.slice_iters) producer.load_stage(stage);
          });
        }
      }
    }
  } else {
    // Divide the register budget among the output and dequantization WGs.
    constexpr uint32_t kMathRegisterLimit = TuningConfig::kNumCtasPerSm == 2 ? 96 : 224;
    constexpr uint32_t kMathRegisters = MIN(kMathRegisterLimit,
                                            ((65536 / TuningConfig::kNumCtasPerSm - 128 * 40) / 256 / 8) * 8);
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(kMathRegisters) : "memory");
    MainloopArith mainloop_arith;
    EpilogueArith epilogue_arith;
    MMA mma(ctx, mainloop_arith);
    Epilogue epilogue(ctx, epilogue_arith);
    S2RMemoryPipeline<Ctx, MMA, Epilogue> s2r(ctx, mma, epilogue);
    Consumer consumer(ctx);
    uint32_t ready_phase[kNumOperandBuffers] = {};
    uint32_t free_phase[kNumOperandBuffers] = {};
    bool operand_used[kNumOperandBuffers] = {};
    if constexpr (kNeedsEpilogueGate) {
      if (ctx.is_math_thread()) consumer.arrive(kNumStages);
    }

    uint32_t next_row_index_buffer = 0;
    while (scheduler.get_next_block()) {
      if constexpr (Ctx::kIsIndexedGemm) {
        ctx.row_index_buffer = next_row_index_buffer;
        next_row_index_buffer ^= 1;
      }
      s2r.seek(scheduler.m_offset);
      if (ctx.is_dequant_thread()) {
        for (uint32_t base_iter = 0; base_iter < scheduler.slice_iters; base_iter += 2 * kNumStages) {
          static_for<0, 2 * kNumStages>([&](auto step) {
            uint32_t iter = base_iter + decltype(step)::value;
            if (iter >= scheduler.slice_iters) return;

            constexpr uint32_t stage = decltype(step)::value % kNumStages;
            constexpr uint32_t buffer = decltype(step)::value % kNumOperandBuffers;
            auto wait_for_operand = [&]() {
              if (operand_used[buffer]) {
                if constexpr (kNumStages == kNumOperandBuffers) {
                  mbarrier_wait(&smem.math_mbar[stage], free_phase[buffer]);
                } else {
                  mbarrier_wait(&smem.umma_operand_free[buffer], free_phase[buffer]);
                }
                free_phase[buffer] ^= 1;
                tcgen05_fence_after_thread_sync();
              }
              operand_used[buffer] = true;
            };
            if (iter == 0) consumer.template wait_stage<true>(0);
            else consumer.wait_stage(stage);
            // Without early weight reuse, weight readiness also protects TMEM:
            // the producer reloads a stage only after its UMMA completes.
            if constexpr (Ctx::kUseUmmaSplitLoads) tcgen05_fence_after_thread_sync();
            mma.set_operand_buffer(buffer);
            PRAGMA_UNROLL
            for (uint32_t group = 0; group < kOutputGroups; group++) {
              ctx.math_group = group;
              constexpr uint32_t kPreparedFragments = MIN(Ctx::kWarpIters, 4);
              constexpr uint32_t kFragmentWords = sizeof(mma.regs_b[0]) / sizeof(uint32_t);
              PRAGMA_UNROLL
              for (uint32_t fragment = 0; fragment < Ctx::kWarpIters; fragment += kPreparedFragments) {
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
                if (group == 0 && fragment == 0) wait_for_operand();
                if constexpr (Ctx::kUseUmmaSplitLoads) {
                  if (group + 1 == kOutputGroups && fragment + kPreparedFragments >= Ctx::kWarpIters) {
                    // All raw weights have been consumed; TMEM stores need
                    // not delay the producer's next shared-memory load.
                    __mbarrier_arrive(&smem.umma_weight_free[stage]);
                  }
                }
                mma.store_b(prepared, fragment);
              }
            }
            tcgen05_wait_st();
            tcgen05_fence_before_thread_sync();
            __mbarrier_arrive(&smem.umma_operand_ready[buffer]);
          });
        }
      } else {
        if (ctx.is_issuer_thread()) {
          for (uint32_t base_iter = 0; base_iter < scheduler.slice_iters; base_iter += 2 * kNumStages) {
            static_for<0, 2 * kNumStages>([&](auto step) {
              uint32_t iter = base_iter + decltype(step)::value;
              if (iter >= scheduler.slice_iters) return;

              constexpr uint32_t stage = decltype(step)::value % kNumStages;
              constexpr uint32_t buffer = decltype(step)::value % kNumOperandBuffers;
              mbarrier_wait(&smem.umma_operand_ready[buffer], ready_phase[buffer]);
              ready_phase[buffer] ^= 1;
              if constexpr (Ctx::kUseUmmaSplitLoads) {
                if (iter == 0) consumer.template wait_activation<true>(0);
                else consumer.wait_activation(stage);
              }
              tcgen05_fence_after_thread_sync();
              if constexpr (!Ctx::kUseTmaA) {
                if (ctx.math_thread_id() == 0) tma_fence_async_shared();
              }
              PRAGMA_UNROLL
              for (uint32_t group = 0; group < kOutputGroups; group++) {
                ctx.math_group = group;
                mma.issue(stage, buffer, iter == 0);
              }
              if (ctx.math_thread_id() == 0) {
                if constexpr (kNumStages != kNumOperandBuffers) {
                  tcgen05_commit(cast_smem_ptr_to_uint(&smem.umma_operand_free[buffer]));
                }
                tcgen05_commit(cast_smem_ptr_to_uint(&smem.math_mbar[stage]));
              }
              free_phase[buffer] ^= 1;
            });
          }

          uint32_t last_buffer = (scheduler.slice_iters - 1) % kNumOperandBuffers;
          uint32_t last_phase = 0;
          // Keep phase counters in registers, including the final partial loop.
          // A dynamic array index forces a local-memory copy of every counter.
          static_for<0, kNumOperandBuffers>([&](auto buffer) {
            if (last_buffer == decltype(buffer)::value) last_phase = free_phase[decltype(buffer)::value];
          });
          if constexpr (kNumStages == kNumOperandBuffers) {
            mbarrier_wait(&smem.math_mbar[last_buffer], last_phase ^ 1);
          } else {
            mbarrier_wait(&smem.umma_operand_free[last_buffer], last_phase ^ 1);
          }
        }
        if (ctx.is_math_thread()) {
          // Let the previous output transfer overlap this tile's mainloop, then
          // protect its shared source before the native writer reuses it.
          if constexpr (Ctx::kUseTmaC) tma_wait_store_group<0, true>();
          ctx.sync_math_threads();
          tcgen05_fence_after_thread_sync();
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
              // Release once per tile, after every partition has read its
              // channel data. The producer may then overwrite that shared data.
              if (group + 1 == kOutputGroups) {
                ctx.sync_math_threads();
                consumer.arrive(kNumStages);
              }
            }
            epilogue.smem_writer.write_umma(mma);
          }
          if constexpr (Ctx::kUseTmaC) tma_fence_async_shared();
          ctx.sync_math_threads();
          epilogue.gmem_writer.write(scheduler.slice_id, scheduler.slice_count, 0);
          if (scheduler.slice_count > 1) epilogue.release_gmem_barrier();
          if constexpr (!kSeparateOutputStorage) {
            if constexpr (Ctx::kUseTmaC) tma_wait_store_group<0, true>();
            ctx.sync_math_threads();
            consumer.arrive(kNumStages);
          }
        }
      }
    }
  }
  if constexpr (Ctx::kUseTmaC) {
    if (ctx.is_math_thread()) tma_wait_store_group<0>();
  }
  __syncthreads();
  MMA::dealloc(smem);
}
