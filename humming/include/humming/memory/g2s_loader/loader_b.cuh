#pragma once

#include <humming/utils/all.cuh>


template <class Ctx>
class G2SMemoryLoaderB {
private:
  using ProblemShape = typename Ctx::ProblemShape;
  using SharedStorage = typename Ctx::SharedStorage;
  using BlockShape = typename Ctx::BlockShape;
  using ElementA = typename Ctx::ElementA;
  using ElementB = typename Ctx::ElementB;

  static constexpr bool kUseWarpSpec = Ctx::kUseWarpSpec;
  static constexpr bool kUseTma = Ctx::kUseTmaB;
  static constexpr bool kUseCpAsync = Ctx::kUseCpAsync;
  static constexpr bool kUseAiu = USE_PPU && kUseCpAsync && !kUseTma;
  static constexpr uint32_t kNumLoadThreads = Ctx::kNumLoadThreads;
  static constexpr uint32_t kLoadThreadOffset = Ctx::kLoadThreadOffset;
  static constexpr uint32_t kMultiCastSizeB = Ctx::kMultiCastSizeB;
  // M-grouped traversal reuses B immediately across neighboring CTAs. Avoid
  // retaining streamed weights at the expense of A's reuse across N tiles.
  static constexpr bool kEvictWeightsFirst = Ctx::kUseUmmaSplitLoads && Ctx::kRasterGroupM > 1;

  static constexpr uint32_t kWeightSmemBits = SharedStorage::kWeightSmemBits;
  static constexpr uint32_t kSwizzleBytes = MIN(128u, SharedStorage::kWeightStageK *kWeightSmemBits / 8);
  static constexpr uint32_t kSwizzleK = kSwizzleBytes * 8 / kWeightSmemBits;
  static constexpr uint32_t kPackSizeK = Ctx::kUsePackedKLayout ? 64 : (256 / ElementA::kBits);
  static constexpr uint32_t kSmemStride = BlockShape::N * kPackSizeK * ElementB::kBits / 32 / 4;
  static constexpr uint32_t kGmemStride = ProblemShape::N * kPackSizeK * ElementB::kBits / 32 / 4;
  static constexpr uint32_t kGmemExpertStride = ProblemShape::N * ProblemShape::K * ElementB::kBits / 32 / 4;
  static constexpr uint32_t kNumInt4s = kSmemStride * BlockShape::K / kPackSizeK;

public:
  Ctx &ctx;
  const CUtensorMap *tensor_map_ptr;
  const int4 *gmem_ptr_raw;
  const int4 *gmem_ptr;

  uint32_t row_offset;
  uint32_t col_offset;
  uint32_t cluster_rank = blockIdx.x % kMultiCastSizeB;

  CUDA_INLINE
  G2SMemoryLoaderB(Ctx &ctx) : ctx(ctx) {
    const void *ptr = ctx.params.b;
    if constexpr (kUseTma) {
      tensor_map_ptr = reinterpret_cast<const CUtensorMap *>(ptr);
    } else {
      gmem_ptr_raw = reinterpret_cast<const int4 *>(ptr);
    }
  }

  template <bool kShouldAdvance = true>
  CUDA_INLINE void load(int4 *smem_ptr, void *mbar_ptr) {
    if constexpr (Ctx::kUseRawWeight && !kUseTma) load_raw(smem_ptr);
    else if constexpr (kUseAiu) load_aiu(smem_ptr);
    else if constexpr (kUseTma) load_tma(smem_ptr, mbar_ptr);
    else load_legacy(smem_ptr);
    if constexpr (kShouldAdvance) advance();
  }

  CUDA_INLINE
  void load_aiu(int4 *smem_ptr) {
    uint32_t warp_id = ctx.load_thread_id() / 32;
    if (warp_id == 0) {
      aiu_load_gmem_linear_3d(
          gmem_ptr, smem_ptr,
          ProblemShape::K / kPackSizeK, ProblemShape::N / 32, kPackSizeK * ElementB::kBits,
          row_offset, col_offset, 0,
          BlockShape::K / kPackSizeK, BlockShape::N / 32, kPackSizeK * ElementB::kBits);
    }
  }

  CUDA_INLINE
  void load_tma(int4 *smem_ptr, void *mbar_ptr) {
    if (ctx.load_thread_id() == 0) {
      if constexpr (Ctx::kUseRawWeight) {
        constexpr uint32_t kCoordinateBits = SharedStorage::kExpandUmmaWeight ? 1 : 32 / ElementB::kBits;
        uint32_t start_k = kCoordinateBits == 1 ? row_offset / 128 * 128 : row_offset;
        if constexpr (Ctx::kUseWgmmaTmaKPack) {
          if (cluster_rank == 0)
            tma_load_5d<kMultiCastSizeB>(tensor_map_ptr, smem_ptr, mbar_ptr,
                                       0, 0, col_offset / Ctx::WarpShape::N, 0, start_k / kSwizzleK);
        } else {
          PRAGMA_UNROLL
          for (uint32_t k = 0; k < SharedStorage::kWeightStageK; k += kSwizzleK) {
            PRAGMA_UNROLL
            for (uint32_t n = 0; n < SharedStorage::kWeightStageN; n += 256) {
              uint32_t offset = (k / kSwizzleK * SharedStorage::kWeightStageN + n) * kSwizzleBytes / sizeof(int4);
              if (cluster_rank == 0) tma_load_2d<kMultiCastSizeB, kEvictWeightsFirst, Ctx::kUseUmmaCooperativeTma ? 2 : 1>(
                  tensor_map_ptr, smem_ptr + offset, mbar_ptr, (start_k + k) / kCoordinateBits, col_offset + n);
            }
          }
        }
      } else if constexpr (kMultiCastSizeB == 1) {
        tma_load_3d<1, kEvictWeightsFirst, Ctx::kUseUmmaCooperativeTma ? 2 : 1>(tensor_map_ptr, smem_ptr, mbar_ptr, 0, col_offset, row_offset);
      } else if (cluster_rank == 0) {
        tma_load_3d<kMultiCastSizeB>(tensor_map_ptr, smem_ptr, mbar_ptr, 0, col_offset, row_offset);
      }
    }
  }

  CUDA_INLINE
  void prefetch_tma() {
    if constexpr (kUseTma && kMultiCastSizeB == 1) {
      if (ctx.load_thread_id() == 0) {
        if constexpr (!Ctx::kUseRawWeight) tma_prefetch_3d(tensor_map_ptr, 0, col_offset, row_offset);
      }
    }
  }

  CUDA_INLINE static uint32_t permute_raw_ss_row(uint32_t row) {
    constexpr uint32_t kWarpN = Ctx::WarpShape::N;
    constexpr uint32_t kGroupN = 4 * kWarpN;
    uint32_t row_in_group = row % kGroupN;
    return row / kGroupN * kGroupN +
           row_in_group % kWarpN / 16 * 64 + row_in_group / kWarpN * 16 + row_in_group % 16;
  }

  CUDA_INLINE
  void load_raw(int4 *smem_ptr) {
    if constexpr (kUseAiu) return load_raw_aiu(smem_ptr);
    static_assert(ElementA::kBits == ElementB::kBits);
    constexpr uint32_t kRowInt4s = BlockShape::K * ElementB::kBits / 128;
    constexpr uint32_t kGlobalRowInt4s = ProblemShape::K * ElementB::kBits / 128;
    constexpr uint32_t kSwizzleInt4s = kSwizzleBytes / sizeof(int4);
    uint32_t smem_base = cast_smem_ptr_to_uint(smem_ptr) / 128;
    constexpr uint32_t kRowsPerIteration = kNumLoadThreads / kRowInt4s;
    constexpr bool kPowerOfTwoRows = kRowsPerIteration != 0 && (kRowsPerIteration & (kRowsPerIteration - 1)) == 0;
    constexpr bool kPowerOfTwoWarpN = (Ctx::WarpShape::N & (Ctx::WarpShape::N - 1)) == 0;
    constexpr bool kSplitRowPermutation =
        Ctx::kUseWgmmaCpAsyncNPermute && kNumLoadThreads % kRowInt4s == 0 && kPowerOfTwoRows && kPowerOfTwoWarpN;
    uint32_t permuted_thread_row = 0;
    if constexpr (kSplitRowPermutation) {
      // The thread and unrolled iteration own disjoint row bits. Permute
      // thread bits once; the iteration contribution folds to constants.
      uint32_t thread_row = (ctx.load_thread_id() / kRowInt4s) % kRowsPerIteration;
      permuted_thread_row = permute_raw_ss_row(thread_row);
    }
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < CEIL_DIV(BlockShape::N * kRowInt4s, kNumLoadThreads); ++i) {
      uint32_t index = i * kNumLoadThreads + ctx.load_thread_id();
      uint32_t row = index / kRowInt4s;
      uint32_t col = index % kRowInt4s;
      uint32_t smem_row = row;
      if constexpr (Ctx::kUseWgmmaCpAsyncNPermute) {
        if constexpr (kSplitRowPermutation) {
          smem_row = permuted_thread_row + permute_raw_ss_row(i * kRowsPerIteration);
        } else smem_row = permute_raw_ss_row(row);
      }
      uint32_t offset = (col / kSwizzleInt4s * BlockShape::N + smem_row) * kSwizzleInt4s + col % kSwizzleInt4s;
      uint32_t swizzled_offset = offset ^ ((smem_base + offset / 8) % kSwizzleInt4s);
      legacy_load_pred<kUseCpAsync>(gmem_ptr + row * kGlobalRowInt4s + col,
                                    smem_ptr + swizzled_offset, row < BlockShape::N);
    }
  }

  CUDA_INLINE
  void load_raw_aiu(int4 *smem_ptr) {
    const uint32_t warp_id = ctx.load_thread_id() / 32;
    PRAGMA_UNROLL
    for (uint32_t slab = warp_id; slab < CEIL_DIV(BlockShape::K, kSwizzleK); slab += kNumLoadThreads / 32) {
      aiu_load_gmem<ElementB::kBits>(
          gmem_ptr_raw, smem_ptr + slab * BlockShape::N * kSwizzleBytes / sizeof(int4),
          ProblemShape::N * MAX(1u, Ctx::kNumExperts), ProblemShape::K,
          col_offset, row_offset + slab * kSwizzleK,
          BlockShape::N, kSwizzleK);
    }
  }

  CUDA_INLINE
  void load_legacy(int4 *smem_ptr) {
    legacy_load_2d<
        kUseCpAsync, kNumInt4s, kNumLoadThreads,
        kGmemStride, kSmemStride, kLoadThreadOffset>(gmem_ptr, smem_ptr);
  }

  CUDA_INLINE
  void advance() {
    if constexpr (Ctx::kUseRawWeight) {
      row_offset += BlockShape::K;
      if constexpr (!kUseTma) gmem_ptr += BlockShape::K * ElementB::kBits / 128;
      return;
    }
    row_offset += BlockShape::K / kPackSizeK;
    if constexpr (!kUseAiu) gmem_ptr += kGmemStride * BlockShape::K / kPackSizeK;
  }

  CUDA_INLINE
  void seek(uint32_t expert_id, uint32_t n_block_id, uint32_t k_block_id) {
    if constexpr (Ctx::kUseRawWeight) {
      row_offset = k_block_id * BlockShape::K;
      col_offset = expert_id * ProblemShape::N + n_block_id * BlockShape::N;
      if constexpr (!kUseTma) {
        uint64_t offset = uint64_t(col_offset) * ProblemShape::K + row_offset;
        gmem_ptr = gmem_ptr_raw + offset * ElementB::kBits / 128;
      }
      return;
    }
    row_offset = k_block_id * (BlockShape::K / kPackSizeK);
    if constexpr (kUseTma) row_offset += expert_id * (ProblemShape::K / kPackSizeK);
    col_offset = n_block_id * (kUseAiu ? BlockShape::N / 32 : BlockShape::N * ElementB::kBits / 32);

    uint64_t gmem_offset = expert_id * kGmemExpertStride;
    if constexpr (!kUseAiu) gmem_offset += n_block_id * kSmemStride + k_block_id * (kGmemStride * BlockShape::K / kPackSizeK);
    gmem_ptr = gmem_ptr_raw + gmem_offset;
  }
};
