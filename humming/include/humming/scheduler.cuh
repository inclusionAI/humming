#pragma once

#include <humming/utils/all.cuh>

template <class Ctx>
class Scheduler {
private:
  using ProblemShape = typename Ctx::ProblemShape;
  using BlockShape = typename Ctx::BlockShape;

  static constexpr bool kUseCpAsync = Ctx::kUseCpAsync;
  static constexpr bool kUseWarpSpec = Ctx::kUseWarpSpec;
  static constexpr bool kUseStreamK = Ctx::kUseStreamK;
  static constexpr bool kIsDenseGemm = Ctx::kIsDenseGemm;
  static constexpr bool kIsIndexedGemm = Ctx::kIsIndexedGemm;
  static constexpr bool kIsGroupedGemm = Ctx::kIsGroupedGemm;
  static constexpr bool kIsGroupedContiguousGemm = Ctx::kIsGroupedContiguousGemm;
  static constexpr uint32_t kNumExperts = Ctx::kNumExperts;
  static constexpr uint32_t kNumThreads = Ctx::kNumThreads;
  static constexpr uint32_t kNumMathThreads = Ctx::kNumMathThreads;
  static constexpr uint32_t kNumLoadThreads = Ctx::kNumLoadThreads;
  static constexpr uint32_t kLoadThreadOffset = Ctx::kLoadThreadOffset;
  // Multicast shares A; cooperative UMMA loads distinct halves of A.
  // Both group adjacent output tiles along N.
  static constexpr uint32_t kNumCtasDimN = Ctx::kMultiCastSizeA * Ctx::kUmmaCtaGroupSize;
  static constexpr uint32_t kNumCtasDimM = Ctx::kMultiCastSizeB;
  static constexpr uint32_t kCtaGroupSize = kNumCtasDimN * kNumCtasDimM;

  static constexpr uint32_t kInputScaleGroupSize = Ctx::kInputScaleGroupSize > 0 ? Ctx::kInputScaleGroupSize : 1;
  static constexpr uint32_t kWeightScaleGroupSize = Ctx::kWeightScaleGroupSize > 0 ? Ctx::kWeightScaleGroupSize : 1;
  static constexpr uint32_t kMaxGroupSize = MAX(kInputScaleGroupSize, kWeightScaleGroupSize);

  static constexpr bool kUseBlockScaledMma = Ctx::kUseBlockScaledMma;
  static constexpr uint32_t kAsBlocksPerWord = kUseBlockScaledMma ? MAX(1u, 4 * kInputScaleGroupSize / BlockShape::K) : 1;

  static constexpr uint32_t N_BLOCKS = ProblemShape::N / BlockShape::N / kNumCtasDimN;
  static constexpr uint32_t K_BLOCKS = ProblemShape::K / BlockShape::K;
  static constexpr uint32_t kStreamKBlockAlignment = MAX(1u, MAX(kMaxGroupSize / BlockShape::K, kAsBlocksPerWord));
  // Pad the scheduling space so every output tile starts at a complete input-scale word.
  // The padding contributes no MMA iterations.
  static constexpr uint32_t kStreamKBlocks = CEIL_DIV(K_BLOCKS, kStreamKBlockAlignment) * kStreamKBlockAlignment;

  static constexpr uint32_t kRasterGroupM = Ctx::kRasterGroupM;
  static constexpr bool kUseGroupedRaster = kIsGroupedContiguousGemm && kRasterGroupM > 1;
  static constexpr uint32_t kNumStages = Ctx::TuningConfig::kNumStages;

  static constexpr int32_t ct_gcd(int32_t a, int32_t b) { return b == 0 ? a : ct_gcd(b, a % b); }

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
  Ctx &ctx;

  uint32_t m_block_id;
  uint32_t n_block_id;
  uint32_t k_block_id;
  uint32_t old_m_block_id = 0;

  // for stream-k
  uint32_t slice_iters;
  uint32_t slice_count = 1;
  uint32_t slice_id = 0;
  uint32_t locks_offset = 0;

  // for tma multi-cast
  uint32_t cluster_rank = blockIdx.x % kCtaGroupSize;

  // for moe gemm (indexed gemm or grouped gemm)
  uint32_t old_expert_id = (1 << 30);
  uint32_t expert_id = 0;

  // for grouped gemm
  uint32_t current_shape_m;
  uint32_t expert_max_num_tokens;
  uint32_t m_block_in_expert = 0;
  uint32_t current_expert_num_tokens = 0;
  uint32_t current_expert_m_blocks = 0;
  uint32_t m_offset = 0;

  CUDA_INLINE
  Scheduler(Ctx &ctx) : ctx(ctx) {

    current_shape_m = ctx.params.shape_m;
    expert_max_num_tokens = ctx.params.shape_m / Ctx::kNumExperts;
    calc_m_blocks();
    if constexpr (kIsGroupedGemm) {
      current_expert_num_tokens = ctx.smem.expert_tokens[0];
      current_expert_m_blocks = CEIL_DIV(current_expert_num_tokens, BlockShape::M);
    }
    mn_blocks = m_blocks * N_BLOCKS;
    mnk_blocks = mn_blocks * kStreamKBlocks;
    uint32_t kNumCtaGroups = gridDim.x / kCtaGroupSize;

    if constexpr (kUseStreamK) {
      uint32_t streamk_mn_blocks = mn_blocks;
      if (mn_blocks > kNumCtaGroups) {
        streamk_mn_blocks = mn_blocks % kNumCtaGroups;
        if (streamk_mn_blocks && streamk_mn_blocks * 10 <= kNumCtaGroups) streamk_mn_blocks += kNumCtaGroups;
      }

      dp_mn_iters = (mn_blocks - streamk_mn_blocks) / kNumCtaGroups;

      uint32_t streamk_mnk_blocks = streamk_mn_blocks * kStreamKBlocks;

      streamk_mnk_total_iters = CEIL_DIV(streamk_mnk_blocks, kNumCtaGroups);

      // A tile can start in the last iteration of a CTA's work interval.
      // Include that partial first slice when bounding the total slice count.
      constexpr uint32_t kMaxSlices = 10;
      constexpr uint32_t kMinSliceIters = CEIL_DIV(kStreamKBlocks - 1, kMaxSlices - 1);
      streamk_mnk_total_iters = MAX(streamk_mnk_total_iters, kMinSliceIters);

      constexpr int32_t bpg = kStreamKBlockAlignment;
      constexpr int32_t align_iters = bpg / ct_gcd(bpg, (int32_t)kNumStages) * (int32_t)kNumStages;
      if constexpr (align_iters > 1) {
        streamk_mnk_total_iters = align_iters * CEIL_DIV(streamk_mnk_total_iters, align_iters);
      };

      streamk_mnk_next_index = kNumCtaGroups * dp_mn_iters * kStreamKBlocks + streamk_mnk_total_iters * (blockIdx.x / kCtaGroupSize);

      if (streamk_mnk_next_index >= mnk_blocks) {
        streamk_mnk_iters = 0;
      } else {
        streamk_mnk_iters = mnk_blocks - streamk_mnk_next_index;
        if (streamk_mnk_iters > streamk_mnk_total_iters) streamk_mnk_iters = streamk_mnk_total_iters;
      };
    } else {
      dp_mn_next_index = blockIdx.x / kCtaGroupSize;
      dp_mn_iters = dp_mn_next_index < mn_blocks;
    }

    dp_mn_total_iters = dp_mn_iters;
    if constexpr (kUseStreamK) {
      if (dp_mn_iters) dp_mn_next_index = blockIdx.x / kCtaGroupSize;
    }
  };

  CUDA_INLINE
  void calc_m_blocks() {
    if constexpr (kIsDenseGemm) {
      m_blocks = CEIL_DIV(ctx.params.shape_m, BlockShape::M * kNumCtasDimM);
    } else if constexpr (kIsIndexedGemm) {
      uint32_t padded_shape_m = ctx.params.num_tokens_padded_ptr[0];
      m_blocks = CEIL_DIV(padded_shape_m, BlockShape::M * kNumCtasDimM);
    } else if constexpr (kIsGroupedGemm) {
      auto &expert_layout = ctx.params.expert_layout_ptr;
      if constexpr (kIsGroupedContiguousGemm) {
        if (ctx.params.use_int64_expert_layout)
          legacy_load_2d<kUseCpAsync, kNumExperts + 1, kNumThreads, 2, 1>(expert_layout, ctx.smem.expert_offset);
        else
          legacy_load_2d<kUseCpAsync, kNumExperts + 1, kNumThreads, 1, 1>(expert_layout, ctx.smem.expert_offset);
      } else {
        if (ctx.params.use_int64_expert_layout)
          legacy_load_2d<kUseCpAsync, kNumExperts, kNumThreads, 2, 1>(expert_layout, ctx.smem.expert_tokens);
        else
          legacy_load_2d<kUseCpAsync, kNumExperts, kNumThreads, 1, 1>(expert_layout, ctx.smem.expert_tokens);
      }
      if constexpr (kUseCpAsync) cp_async_commit_group();
      if constexpr (kUseCpAsync) cp_async_wait_group<0>();
      __syncthreads();

      if constexpr (kUseGroupedRaster) {
        // Build the per-expert prefix of M tiles for a persistent, flat
        // grouped raster. Each warp lane owns one expert per 32-expert chunk.
        if (threadIdx.x < 32) {
          const uint32_t lane = threadIdx.x;
          uint32_t carry = 0;
          for (uint32_t base = 0; base < kNumExperts; base += 32) {
            const uint32_t expert = base + lane;
            uint32_t tokens = 0;
            if (expert < kNumExperts) {
              const uint32_t next_offset = ctx.smem.expert_offset[expert + 1];
              tokens = next_offset - ctx.smem.expert_offset[expert];
              ctx.smem.expert_tokens[expert] = tokens;
            }
            uint32_t blocks = CEIL_DIV(tokens, BlockShape::M);
            PRAGMA_UNROLL
            for (uint32_t delta = 1; delta < 32; delta <<= 1) {
              const uint32_t previous = __shfl_up_sync(0xffffffff, blocks, delta);
              if (lane >= delta) blocks += previous;
            }
            if (expert < kNumExperts)
              ctx.smem.expert_m_block_offset[expert + 1] = carry + blocks;
            carry += __shfl_sync(0xffffffff, blocks, 31);
          }
          if (lane == 0) {
            ctx.smem.expert_m_block_offset[0] = 0;
            ctx.smem.total_m_blocks[0] = carry;
          }
        }
      } else if (threadIdx.x < 32) {
        uint32_t tmp_m_blocks = 0;
        PRAGMA_UNROLL
        for (uint32_t i = 0; i < CEIL_DIV(kNumExperts, 32); i++) {
          uint32_t index = 32 * i + threadIdx.x;
          if (index < kNumExperts) {
            uint32_t expert_tokens;
            if constexpr (kIsGroupedContiguousGemm) {
              uint32_t next_offset = ctx.smem.expert_offset[index + 1];
              expert_tokens = next_offset - ctx.smem.expert_offset[index];
              ctx.smem.expert_tokens[index] = expert_tokens;
            } else {
              expert_tokens = ctx.smem.expert_tokens[index];
            }
            tmp_m_blocks += CEIL_DIV(expert_tokens, BlockShape::M);
          }
        }

        m_blocks = warp_reduce_add(tmp_m_blocks);
        if (threadIdx.x == 0) ctx.smem.total_m_blocks[0] = m_blocks;
      }

      __syncthreads();
      m_blocks = ctx.smem.total_m_blocks[0];
    }
  }

  CUDA_INLINE
  void map_mn_block(uint32_t mn_index, uint32_t &m_id, uint32_t &n_id) {
    if constexpr (kRasterGroupM <= 1 || !(kIsDenseGemm || kUseGroupedRaster)) {
      m_id = mn_index / N_BLOCKS;
      n_id = mn_index % N_BLOCKS;
    } else {
      const uint32_t blocks_per_group = kRasterGroupM * N_BLOCKS;
      const uint32_t group_id = mn_index / blocks_per_group;
      const uint32_t first_m = group_id * kRasterGroupM;
      const uint32_t group_m = MIN(m_blocks - first_m, kRasterGroupM);
      const uint32_t idx_in_group = mn_index - group_id * blocks_per_group;
      m_id = first_m + idx_in_group % group_m;
      n_id = idx_in_group / group_m;
    }
  }

  CUDA_INLINE
  bool get_next_block() {
    bool has_next_block = false;
    // Derive the non-Stream-K predicate from the index. Updating a cached predicate
    // inside this branch can make ptxas mispredicate the current tile's TMA coordinates.
    if (kUseStreamK ? dp_mn_iters != 0 : dp_mn_next_index < mn_blocks) {
      slice_iters = K_BLOCKS;

      map_mn_block(dp_mn_next_index, m_block_id, n_block_id);

      if constexpr (kNumCtasDimM > 1) {
        m_block_id = m_block_id * kNumCtasDimM + cluster_rank;
      } else if constexpr (kNumCtasDimN > 1) {
        n_block_id = n_block_id * kNumCtasDimN + cluster_rank;
      }
      k_block_id = 0;
      dp_mn_next_index += gridDim.x / kCtaGroupSize;
      if constexpr (kUseStreamK) dp_mn_iters--;
      has_next_block = true;
    } else if constexpr (kUseStreamK) {
      has_next_block = get_streamk_next_block();
    }

    if constexpr (kIsIndexedGemm) {
      if (has_next_block) fetch_moe_index_block();
    }
    if constexpr (kIsGroupedGemm) {
      if (has_next_block) fetch_moe_group_block();
    }

    return has_next_block;
  };

  CUDA_INLINE
  bool get_streamk_next_block() {
    if (!streamk_mnk_iters) return false;
    uint32_t streamk_mn_index = streamk_mnk_next_index / kStreamKBlocks;

    map_mn_block(streamk_mn_index, m_block_id, n_block_id);
    if constexpr (kNumCtasDimM > 1) {
      m_block_id = m_block_id * kNumCtasDimM + cluster_rank;
    } else if constexpr (kNumCtasDimN > 1) {
      n_block_id = n_block_id * kNumCtasDimN + cluster_rank;
    }
    k_block_id = streamk_mnk_next_index - streamk_mn_index * kStreamKBlocks;

    uint32_t scheduled_iters = MIN(kStreamKBlocks - k_block_id, streamk_mnk_iters);
    slice_iters = MIN(K_BLOCKS - k_block_id, scheduled_iters);

    streamk_mnk_iters -= scheduled_iters;
    streamk_mnk_next_index += scheduled_iters;

    if (k_block_id == 0) {
      slice_id = 0;
      slice_count = CEIL_DIV(kStreamKBlocks - scheduled_iters, streamk_mnk_total_iters) + 1;
    } else {
      slice_id = k_block_id / streamk_mnk_total_iters;
      uint32_t slice_first_block_iters = k_block_id - slice_id * streamk_mnk_total_iters;
      slice_count = CEIL_DIV(kStreamKBlocks - slice_first_block_iters, streamk_mnk_total_iters);
      if (slice_first_block_iters) {
        slice_id++;
        slice_count++;
      }
    }

    slice_id = slice_count - 1 - slice_id;

    locks_offset = streamk_mn_index - dp_mn_total_iters * gridDim.x / kCtaGroupSize;
    locks_offset = locks_offset * kCtaGroupSize + cluster_rank;

    return true;
  };

  CUDA_INLINE
  void fetch_moe_group_block() {
    if constexpr (kUseGroupedRaster) {
      uint32_t lower = 0;
      uint32_t upper = kNumExperts;
      PRAGMA_UNROLL
      for (uint32_t step = 0; step < constexpr_log2(2 * kNumExperts - 1); ++step) {
        const uint32_t middle = (lower + upper) >> 1;
        if (m_block_id >= ctx.smem.expert_m_block_offset[middle])
          lower = middle;
        else
          upper = middle;
      }
      expert_id = lower;
      const uint32_t first_block = ctx.smem.expert_m_block_offset[expert_id];
      m_offset = ctx.smem.expert_offset[expert_id] +
                 (m_block_id - first_block) * BlockShape::M;
      current_shape_m = ctx.smem.expert_offset[expert_id] +
                        ctx.smem.expert_tokens[expert_id];
      old_expert_id = expert_id;
      return;
    }
    uint32_t delta_m_block_id = m_block_id - old_m_block_id;
    m_block_in_expert += delta_m_block_id;

    while (m_block_in_expert >= current_expert_m_blocks) {
      uint32_t lane = threadIdx.x % 32;
      uint32_t index = expert_id + lane;
      uint32_t tokens = index < kNumExperts ? ctx.smem.expert_tokens[index] : 0;
      uint32_t blocks = CEIL_DIV(tokens, BlockShape::M);
      uint32_t prefix = blocks;
      PRAGMA_UNROLL
      for (uint32_t offset = 1; offset < 32; offset *= 2) {
        uint32_t preceding = __shfl_up_sync(0xffffffff, prefix, offset);
        if (lane >= offset) prefix += preceding;
      }
      uint32_t matches = __ballot_sync(0xffffffff, m_block_in_expert < prefix);
      if (matches) {
        uint32_t selected_lane = __ffs(matches) - 1;
        m_block_in_expert -= __shfl_sync(0xffffffff, prefix - blocks, selected_lane);
        current_expert_num_tokens = __shfl_sync(0xffffffff, tokens, selected_lane);
        current_expert_m_blocks = __shfl_sync(0xffffffff, blocks, selected_lane);
        expert_id += selected_lane;
        break;
      }
      m_block_in_expert -= __shfl_sync(0xffffffff, prefix, 31);
      expert_id += 32;
      current_expert_m_blocks = 0;
    }

    old_m_block_id = m_block_id;
    uint32_t offset_in_expert = m_block_in_expert * BlockShape::M;
    if constexpr (Ctx::kIsGroupedMaskedGemm) {
      m_offset = expert_id * expert_max_num_tokens + offset_in_expert;
      current_shape_m = expert_id * expert_max_num_tokens + current_expert_num_tokens;
    } else if constexpr (Ctx::kIsGroupedContiguousGemm) {
      m_offset = ctx.smem.expert_offset[expert_id] + offset_in_expert;
      current_shape_m = ctx.smem.expert_offset[expert_id] + current_expert_num_tokens;
    }

    old_expert_id = expert_id;
  };

  CUDA_INLINE
  void fetch_moe_index_block() {
    expert_id = ctx.params.expert_ids_ptr[m_block_id];
    if (kUseWarpSpec && !ctx.is_load_thread()) return;

    const uint32_t *gmem_ptr = ctx.params.sorted_ids_ptr + m_block_id * BlockShape::M;
    const int4 *gmem_ptr_load = reinterpret_cast<const int4 *>(gmem_ptr);
    uint32_t *wr_row_index = ctx.get_wr_row_index();
    uint32_t *rd_row_index = ctx.get_rd_row_index();
    int4 *smem_ptr_load = reinterpret_cast<int4 *>(wr_row_index);

    legacy_load_1d<kUseCpAsync, BlockShape::M / 4, kNumLoadThreads, kLoadThreadOffset>(gmem_ptr_load, smem_ptr_load);
    if constexpr (kUseCpAsync) cp_async_commit_group();
    if constexpr (kUseCpAsync) cp_async_wait_group<0>();

    ctx.sync_load_threads();

    uint32_t thread_id = threadIdx.x;
    if constexpr (kUseWarpSpec) thread_id = ctx.load_thread_id();
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < CEIL_DIV(BlockShape::M, kNumLoadThreads); i++) {
      uint32_t index = kNumLoadThreads * i + thread_id;
      if (index < BlockShape::M) {
        uint32_t idx = wr_row_index[index];
        rd_row_index[index] = idx / ctx.params.top_k;
      };
    }

    ctx.sync_load_threads();
  };
};
