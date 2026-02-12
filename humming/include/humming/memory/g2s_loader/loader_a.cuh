#pragma once

#include <humming/utils/all.cuh>


template <
    class ProblemShape, class BlockShape,
    class ElementA, class PipelineConfig, class MoEConfig>
class G2SMemoryLoaderA {
private:
  static constexpr bool kUseWarpSpec = PipelineConfig::kUseWarpSpec;
  static constexpr bool kUseTma = PipelineConfig::kUseTmaA;
  static constexpr bool kUseCpAsync = PipelineConfig::kUseCpAsync;
  static constexpr bool kIsMoE = MoEConfig::kIsMoE;
  static constexpr uint32_t kNumLoadThreads = PipelineConfig::kNumLoadThreads;
  static constexpr uint32_t kLoadThreadOffset = PipelineConfig::kNumThreads - kNumLoadThreads;

  static constexpr uint32_t kSmemStride = BlockShape::K * ElementA::kBits / 32 / 4;
  static constexpr uint32_t kGmemStride = ProblemShape::K * ElementA::kBits / 32 / 4;
  static constexpr uint32_t kNumInt4s = kSmemStride * BlockShape::M;

  static_assert(BlockShape::K * ElementA::kBits >= 512);
  static constexpr uint32_t kSwizzleBytes = BlockShape::K * ElementA::kBits == 512 ? 64 : 128;
  static constexpr uint32_t kNumTmaLoadsPerLine = CEIL_DIV(BlockShape::K * ElementA::kBits, kSwizzleBytes * 8);
  static constexpr uint32_t kLoadIters = CEIL_DIV(kNumInt4s, kNumLoadThreads);

public:
  const uint32_t thread_id = threadIdx.x - kLoadThreadOffset;
  const CUtensorMap *tensor_map_ptr;
  const int4 *gmem_ptr_raw;
  const int4 *gmem_ptr;

  const uint32_t *row_index;
  const uint32_t shape_m;
  uint32_t block_shape_m;
  uint32_t load_row_index[kUseTma ? CEIL_DIV(BlockShape::M, kNumLoadThreads) : kLoadIters];
  uint32_t row_offset;
  uint32_t col_offset;

  CUDA_INLINE
  G2SMemoryLoaderA(const void *ptr, const uint32_t *row_index, uint32_t shape_m)
      : row_index(row_index), shape_m(shape_m) {
    if constexpr (kUseTma) {
      tensor_map_ptr = reinterpret_cast<const CUtensorMap *>(ptr);
    } else {
      gmem_ptr_raw = reinterpret_cast<const int4 *>(ptr);
    }
  }

  template <bool kShouldAdvance = true>
  CUDA_INLINE void load(int4 *smem_ptr, void *mbar_ptr) {
    if constexpr (kUseTma) load_tma(smem_ptr, mbar_ptr);
    else load_legacy(smem_ptr);
    if constexpr (kShouldAdvance) advance();
  }

  CUDA_INLINE
  void load_tma(int4 *smem_ptr, void *mbar_ptr) {
    if constexpr (!kIsMoE) {
      load_tma_dense(smem_ptr, mbar_ptr);
    } else {
      load_tma_moe(smem_ptr, mbar_ptr);
    }
  }

  CUDA_INLINE
  void load_tma_dense(int4 *smem_ptr, void *mbar_ptr) {
    if (thread_id < kNumTmaLoadsPerLine) {
      const uint32_t block_idx = thread_id;
      const uint32_t smem_offset = BlockShape::M * 8 * block_idx;
      const uint32_t col_offset2 = col_offset + (1024 / ElementA::kBits) * block_idx;
      tma_load_2d(tensor_map_ptr, smem_ptr + smem_offset, mbar_ptr, col_offset2, row_offset);
    }
  }

  CUDA_INLINE
  void load_tma_moe(int4 *smem_ptr, void *mbar_ptr) {
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < CEIL_DIV(BlockShape::M, kNumLoadThreads); i++) {
      uint32_t row = thread_id + kNumLoadThreads * i;
      if (row >= BlockShape::M) break;
      uint32_t gmem_row = load_row_index[i];

      PRAGMA_UNROLL
      for (uint32_t block_idx = 0; i < kNumTmaLoadsPerLine; i++) {
        const uint32_t smem_offset = row * (kSwizzleBytes / 16) + BlockShape::M * 8 * block_idx;
        const uint32_t col_offset2 = col_offset + (1024 / ElementA::kBits) * block_idx;
        tma_load_2d(tensor_map_ptr, smem_ptr + smem_offset, mbar_ptr, col_offset2, gmem_row);
      }
    };
  }

  CUDA_INLINE
  void load_legacy(int4 *smem_ptr) {
    if constexpr (kSwizzleBytes == 128) {
      load_legacy_swizzled_128B(smem_ptr);
    } else if (kSwizzleBytes == 64) {
      load_legacy_swizzled_64B(smem_ptr);
    }
  }

  CUDA_INLINE
  void load_legacy_swizzled_128B(int4 *smem_ptr) {
    static_assert(BlockShape::K * ElementA::kBits >= 1024);
    uint32_t smem = cast_smem_ptr_to_uint(smem_ptr);
    uint32_t smem_swizzled_col = (thread_id % 8) ^ (((thread_id % 64) / 8 + smem / 128)) % 8;

    PRAGMA_UNROLL
    for (uint32_t i = 0; i < kLoadIters; i++) {
      uint32_t smem_offset = i * kNumLoadThreads + thread_id;
      uint32_t smem_row = smem_offset / 8;
      uint32_t smem_col = smem_offset % 8;
      uint32_t smem_swizzled_offset = smem_row * 8 + smem_swizzled_col;

      uint32_t gmem_col = smem_row / BlockShape::M * 8 + smem_col;
      uint32_t gmem_row = kIsMoE ? load_row_index[i] : (smem_row % BlockShape::M);
      uint32_t gmem_offset = gmem_row * kGmemStride + gmem_col;

      bool pred1 = kNumInt4s % kNumLoadThreads == 0 || i != kLoadIters - 1 || smem_offset < kNumInt4s;
      bool pred2 = gmem_row < (kIsMoE ? shape_m : block_shape_m);

      legacy_load_pred<kUseCpAsync>(gmem_ptr + gmem_offset, smem_ptr + smem_swizzled_offset, pred1 && pred2);
    }
  }

  CUDA_INLINE
  void load_legacy_swizzled_64B(int4 *smem_ptr) {
    static_assert(BlockShape::K * ElementA::kBits == 512);
    uint32_t smem = cast_smem_ptr_to_uint(smem_ptr);
    uint32_t smem_swizzled_col = (thread_id % 8) ^ (((thread_id % 32) / 8 + smem / 128) % 4);

    PRAGMA_UNROLL
    for (uint32_t i = 0; i < kLoadIters; i++) {
      uint32_t smem_offset = i * kNumLoadThreads + thread_id;
      uint32_t smem_row = smem_offset / 8;
      uint32_t smem_col = smem_offset % 8;
      uint32_t smem_swizzled_offset = smem_row * 8 + smem_swizzled_col;

      uint32_t gmem_row = smem_row % (BlockShape::M / 2) * 2 + smem_col / 4;
      gmem_row = kIsMoE ? load_row_index[i] : gmem_row;
      uint32_t gmem_col = smem_col % 4;
      uint32_t gmem_offset = gmem_row * kGmemStride + gmem_col;

      bool pred1 = kNumInt4s % kNumLoadThreads == 0 || i != kLoadIters - 1 || smem_offset < kNumInt4s;
      bool pred2 = gmem_row < (kIsMoE ? shape_m : block_shape_m);

      legacy_load_pred<kUseCpAsync>(gmem_ptr + gmem_offset, smem_ptr + smem_swizzled_offset, pred1 && pred2);
    }
  }

  CUDA_INLINE
  void advance() {
    col_offset += BlockShape::K;
    gmem_ptr += kSmemStride;
  }

  CUDA_INLINE
  void seek(uint32_t m_block_id, uint32_t k_block_id) {
    row_offset = m_block_id * BlockShape::M;
    col_offset = k_block_id * BlockShape::K;
    block_shape_m = shape_m - row_offset;

    uint32_t gmem_offset = k_block_id * kSmemStride;
    gmem_offset += kIsMoE ? 0 : (m_block_id * (kGmemStride * BlockShape::M));
    gmem_ptr = gmem_ptr_raw + gmem_offset;

    if constexpr (kIsMoE && kUseTma) {
      static_assert(CEIL_DIV(BlockShape::M, kNumLoadThreads) == 1);
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < CEIL_DIV(BlockShape::M, kNumLoadThreads); i++) {
        uint32_t row = thread_id + kNumLoadThreads * i;
        if (row < BlockShape::M) load_row_index[i] = row_index[row];
      }
    } else if constexpr (kIsMoE && !kUseTma) {
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < kLoadIters; i++) {
        uint32_t smem_offset = i * kNumLoadThreads + thread_id;
        uint32_t smem_row = smem_offset / 8;
        uint32_t smem_col = smem_offset % 8;
        uint32_t gmem_row;

        if constexpr (BlockShape::K * ElementA::kBits >= 1024) {
          gmem_row = smem_row % BlockShape::M;
        } else {
          gmem_row = smem_row % (BlockShape::M / 2) * 2 + smem_col / 4;
        }
        load_row_index[i] = row_index[gmem_row];
      }
    }
  }
};
