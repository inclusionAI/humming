#pragma once

#include <humming/utils/all.cuh>

#include <humming/epilogue/gmem_writer.cuh>
#include <humming/epilogue/smem_reducer.cuh>
#include <humming/epilogue/smem_writer.cuh>


template <class Ctx, class MMA, class ArithClass>
class EpiloguePipeline {
private:
  using SharedStorage = typename Ctx::SharedStorage;
  using BlockShape = typename Ctx::BlockShape;
  using WarpShape = typename Ctx::WarpShape;

  using SmemReducer = EpilogueSmemReducer<Ctx, MMA>;
  using SmemWriter = EpilogueSmemWriter<Ctx, MMA, ArithClass>;
  using GmemWriter = EpilogueGmemWriter<Ctx, ArithClass>;

  static constexpr bool kIsGroupedGemm = Ctx::kIsGroupedGemm;
  static constexpr bool kHasTensorInputScale = Ctx::kIsTensorInputScale || Ctx::kIsTensorInputScale2;
  static constexpr uint32_t kOutputRows = SharedStorage::kOutputRows;
  static constexpr uint32_t kNumStreamKSyncThreads = GmemWriter::kUseWarpgroupEpilogue ? 128 : Ctx::kNumMathThreads;

  CUDA_INLINE uint32_t get_output_barrier_id() {
    return GmemWriter::kUseWarpgroupEpilogue ? 3 + ctx.math_thread_id() / 128 : 1;
  }

  CUDA_INLINE void sync_output_threads() {
    if constexpr (GmemWriter::kUseWarpgroupEpilogue) {
      // Barriers 1 and 2 belong to the math and producer pipelines.
      sync_part_threads<128, Ctx::kNumThreads>(get_output_barrier_id());
    } else {
      ctx.sync_math_threads();
    }
  }

public:
  Ctx &ctx;
  SmemReducer smem_reducer;
  SmemWriter smem_writer;
  GmemWriter gmem_writer;
  ArithClass &arith;
  int32_t *locks;

  uint32_t slice_count;
  uint32_t slice_id;
  uint32_t locks_offset;

  CUDA_INLINE
  EpiloguePipeline(Ctx &ctx, ArithClass &arith)
      : ctx(ctx), locks(ctx.params.locks), arith(arith),
        smem_reducer(ctx), smem_writer(ctx, arith), gmem_writer(ctx, arith) {
    if constexpr (Ctx::kUseTmaC) {
      if constexpr (SharedStorage::kUseDynamicOutputMap) {
        gmem_writer.update_tensor_map_ptr(ctx.params.tensor_map_buffer + blockIdx.x);
      } else if constexpr (!Ctx::kUseWarpSpec) {
        if (threadIdx.x == 0) prefetch_tensor_map(ctx.params.c);
      }
    }
    ctx.sync_math_threads();
  }

  CUDA_INLINE
  void call(uint32_t *regs_c_ptr) {
    // Issuing threads drain the previous tile before any math thread reuses its output storage.
    if constexpr (Ctx::kUseWarpSpec && Ctx::kUseTmaC && Ctx::kSmemReuseMode == SmemReuseMode::NONE)
      tma_wait_store_group<0, true>();
    // Reduction scratch and aliased input storage are shared across warpgroups.
    if constexpr (BlockShape::K > WarpShape::K || Ctx::kSmemReuseMode != SmemReuseMode::NONE) {
      ctx.sync_math_threads();
    } else {
      sync_output_threads();
    }
    if constexpr (BlockShape::K > WarpShape::K) smem_reducer.reduce(regs_c_ptr);
    if (slice_count > 1) acquire_gmem_barrier();
    PRAGMA_UNROLL
    for (uint32_t first_row = 0; first_row < BlockShape::M; first_row += kOutputRows) {
      smem_writer.write(regs_c_ptr, slice_count, first_row);
      if constexpr (Ctx::kUseTmaC) {
        if (ctx.is_math_thread()) tma_fence_async_shared();
      }
      sync_output_threads();
      if constexpr (Ctx::kOutputChunkRows) {
        gmem_writer.write_chunk(slice_id, slice_count, first_row, MIN(kOutputRows, BlockShape::M - first_row), 0);
        // This path reuses one buffer. Wait before the next chunk overwrites it.
        if constexpr (Ctx::kUseTmaC) tma_wait_store_group<0, true>();
      } else {
        gmem_writer.write(slice_id, slice_count, 0);
      }
      sync_output_threads();
    }
    if constexpr (Ctx::kOutputChunkRows && Ctx::kUseTmaC && Ctx::kUseStreamK) {
      if (slice_count > 1 && slice_id != slice_count - 1) tma_wait_store_group<0>();
    }
    if constexpr (GmemWriter::kUseWarpgroupEpilogue && Ctx::kUseTmaC && !Ctx::kUseWarpSpec) {
      tma_wait_store_group<0, true>();
    }
    if (slice_count > 1) release_gmem_barrier();
    // Independent output locks do not protect scratch, descriptors or row indices
    // shared with other warpgroups or the next tile.
    if constexpr (GmemWriter::kUseWarpgroupEpilogue &&
                  (BlockShape::K > WarpShape::K || Ctx::kIsIndexedGemm || Ctx::kSmemReuseMode != SmemReuseMode::NONE || (Ctx::kUseTmaC && !Ctx::kUseWarpSpec)))
      ctx.sync_math_threads();
  }

  CUDA_INLINE
  void acquire_gmem_barrier() {
    if constexpr (GmemWriter::kUseWarpgroupEpilogue) {
      if (ctx.k_warp_id() != 0) return;
    }
    const uint32_t thread_id = GmemWriter::kUseWarpgroupEpilogue ? ctx.math_thread_id() % 128 : ctx.math_thread_id();
    if (Ctx::kUseTmaC || slice_count > 3) {
      int32_t val = slice_id == 0 ? 0 : -1;
      barrier_acquire2<kNumStreamKSyncThreads, Ctx::kNumThreads>(&locks[locks_offset], val, thread_id, get_output_barrier_id());
    } else {
      barrier_acquire<kNumStreamKSyncThreads, Ctx::kNumThreads>(&locks[locks_offset], slice_id, thread_id, get_output_barrier_id());
    }
  }

  CUDA_INLINE
  void release_gmem_barrier() {
    if constexpr (GmemWriter::kUseWarpgroupEpilogue) {
      if (ctx.k_warp_id() != 0) return;
    }
    const uint32_t thread_id = GmemWriter::kUseWarpgroupEpilogue ? ctx.math_thread_id() % 128 : ctx.math_thread_id();
    if (Ctx::kUseTmaC || slice_count > 3) {
      int32_t val = slice_id == 0 ? 1 - static_cast<int32_t>(slice_count) : 0;
      barrier_release2<kNumStreamKSyncThreads, Ctx::kNumThreads>(&locks[locks_offset], val, thread_id, get_output_barrier_id());
    } else {
      barrier_release<kNumStreamKSyncThreads, Ctx::kNumThreads>(&locks[locks_offset], slice_id == slice_count - 1, thread_id, get_output_barrier_id());
    }
  }

  // Load directly into epilogue registers while UMMA is still accumulating. Indexed
  // rows come from the immutable routing table, not the next tile's shared indices.
  CUDA_INLINE
  void load_secondary_input_scale(uint32_t m_block_id, uint32_t current_shape_m, uint32_t m_offset) {
    if constexpr (Ctx::kHasInputScale2) {
      const uint32_t *scales = reinterpret_cast<const uint32_t *>(ctx.params.as2);
      if constexpr (Ctx::kIsTensorInputScale2) {
        arith.as[0] = scales[0];
      } else {
        PRAGMA_UNROLL
        for (uint32_t i = 0; i < WarpShape::M / 8; i++) {
          uint32_t row = i * 8 + ctx.lane_id() / 4;
          if constexpr (Ctx::kIsIndexedGemm) {
            row = ctx.params.sorted_ids_ptr[m_block_id * BlockShape::M + row] / ctx.params.top_k;
          } else if constexpr (kIsGroupedGemm) {
            row += m_offset;
          } else {
            row += m_block_id * BlockShape::M;
          }
          arith.as[i] = row < current_shape_m ? scales[row] : __float_as_uint(1.0f);
        }
      }
    }
  }

  CUDA_INLINE
  void seek(uint32_t expert_id, uint32_t m_block_id, uint32_t n_block_id, uint32_t current_shape_m, uint32_t m_offset) {
    gmem_writer.seek(m_block_id, n_block_id, current_shape_m, m_offset);
    if constexpr (kHasTensorInputScale && !(Ctx::kUseUmma && Ctx::kHasInputScale2)) {
      const uint32_t *as_ptr = reinterpret_cast<const uint32_t *>(Ctx::kIsTensorInputScale2 ? ctx.params.as2 : ctx.params.as);
      arith.as[0] = as_ptr[0];
    }
    if constexpr (Ctx::kHasTensorWeightScale) {
      const uint32_t *gs_ptr = reinterpret_cast<const uint32_t *>(Ctx::kIsTensorWeightScale2 ? ctx.params.bs2 : ctx.params.bs);
      arith.gs = gs_ptr[Ctx::kIsDenseGemm ? 0 : expert_id];
    }
  };

  CUDA_INLINE
  void set_streamk_state(uint32_t slice_count_, uint32_t slice_id_, uint32_t locks_offset_) {
    slice_count = slice_count_;
    slice_id = slice_id_;
    locks_offset = locks_offset_;
    if constexpr (GmemWriter::kUseWarpgroupEpilogue)
      locks_offset = locks_offset_ * SharedStorage::kNumStreamKLocksPerTile + ctx.math_thread_id() / 128;
  };
};
