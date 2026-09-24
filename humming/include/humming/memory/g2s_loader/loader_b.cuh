#pragma once

#include <humming/utils/all.cuh>


template <class Ctx>
class G2SMemoryLoaderB {
private:
  using ProblemShape = typename Ctx::ProblemShape;
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
    if constexpr (kUseAiu) load_aiu(smem_ptr);
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
      if constexpr (kMultiCastSizeB == 1) {
        tma_load_3d<1, kEvictWeightsFirst>(tensor_map_ptr, smem_ptr, mbar_ptr, 0, col_offset, row_offset);
      } else if (cluster_rank == 0) {
        tma_load_3d<kMultiCastSizeB>(tensor_map_ptr, smem_ptr, mbar_ptr, 0, col_offset, row_offset);
      }
    }
  }

  CUDA_INLINE
  void prefetch_tma() {
    if constexpr (kUseTma && kMultiCastSizeB == 1) {
      if (ctx.load_thread_id() == 0) {
        tma_prefetch_3d(tensor_map_ptr, 0, col_offset, row_offset);
      }
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
    row_offset += BlockShape::K / kPackSizeK;
    if constexpr (!kUseAiu) gmem_ptr += kGmemStride * BlockShape::K / kPackSizeK;
  }

  CUDA_INLINE
  void seek(uint32_t expert_id, uint32_t n_block_id, uint32_t k_block_id) {
    row_offset = k_block_id * (BlockShape::K / kPackSizeK);
    if constexpr (kUseTma) row_offset += expert_id * (ProblemShape::K / kPackSizeK);
    col_offset = n_block_id * (kUseAiu ? BlockShape::N / 32 : BlockShape::N * ElementB::kBits / 32);

    uint64_t gmem_offset = expert_id * kGmemExpertStride;
    if constexpr (!kUseAiu) gmem_offset += n_block_id * kSmemStride + k_block_id * (kGmemStride * BlockShape::K / kPackSizeK);
    gmem_ptr = gmem_ptr_raw + gmem_offset;
  }
};
