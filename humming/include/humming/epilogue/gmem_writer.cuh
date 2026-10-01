#pragma once

#include <humming/utils/all.cuh>


template <typename scalar_t2, typename T>
CUDA_INLINE T reduce_add_f162(T &a, T &b) {
  scalar_t2 *a_half2_ptr = reinterpret_cast<scalar_t2 *>(&a);
  scalar_t2 *b_half2_ptr = reinterpret_cast<scalar_t2 *>(&b);

  PRAGMA_UNROLL
  for (uint32_t i = 0; i < sizeof(T) / 4; i++) {
    a_half2_ptr[i] = __hadd2(a_half2_ptr[i], b_half2_ptr[i]);
  };
  return a;
};


template <typename scalar_t2, typename T>
CUDA_INLINE T atomic_reduce_add_f162(T &a, T &b) {
  scalar_t2 *a_half2_ptr = reinterpret_cast<scalar_t2 *>(&a);
  scalar_t2 *b_half2_ptr = reinterpret_cast<scalar_t2 *>(&b);

  PRAGMA_UNROLL
  for (uint32_t i = 0; i < sizeof(T) / 4; i++) {
    atomicAdd(&b_half2_ptr[i], a_half2_ptr[i]);
  };
  return a;
};

template <class Ctx, class ArithClass>
class EpilogueGmemWriter : F16Conversion<typename Ctx::ElementC> {
private:
  using SharedStorage = typename Ctx::SharedStorage;
  using ProblemShape = typename Ctx::ProblemShape;
  using BlockShape = typename Ctx::BlockShape;
  using PadShape = typename Ctx::PadShape;
  using ElementC = typename Ctx::ElementC;

  static constexpr bool kUseStreamK = Ctx::kUseStreamK;
  static constexpr bool kUseTmaC = Ctx::kUseTmaC;

  static constexpr bool kIsIndexedGemm = Ctx::kIsIndexedGemm;
  static constexpr bool kIsGroupedGemm = Ctx::kIsGroupedGemm;

  static constexpr uint32_t kNumMathThreads = Ctx::kNumMathThreads;
  static constexpr uint32_t kNumWriteSplits = Ctx::kNumWriteSplits;

  using scalar_t = typename F16Conversion<ElementC>::scalar_t;
  using scalar_t2 = typename F16Conversion<ElementC>::scalar_t2;

public:
  Ctx &ctx;
  ArithClass &arith;
  int4 *gmem_ptr_raw;
  int4 *gmem_ptr;
  const CUtensorMap *tensor_map_ptr;

  uint32_t row_offset;
  uint32_t col_offset;
  uint32_t output_shape_m;
  uint32_t block_output_shape_m;

  CUDA_INLINE
  EpilogueGmemWriter(Ctx &ctx, ArithClass &arith) : ctx(ctx), arith(arith) {
    const void *ptr = ctx.params.c;
    if constexpr (kUseTmaC) {
      tensor_map_ptr = reinterpret_cast<const CUtensorMap *>(ptr);
    } else {
      gmem_ptr_raw = reinterpret_cast<int4 *>(const_cast<void *>(ptr));
    }

    output_shape_m = ctx.params.shape_m * (kIsIndexedGemm ? ctx.params.top_k : 1);
  }

  CUDA_INLINE
  void write(uint32_t slice_id, uint32_t slice_count, uint32_t split_idx) {
    if constexpr (kUseTmaC) {
      write_tma(slice_id, slice_count);
    } else {
      write_legacy(slice_id, slice_count, split_idx);
    }
  };

  template <uint32_t kRows = BlockShape::M / kNumWriteSplits,
            uint32_t kColumns = BlockShape::N, uint32_t kStorageRows = kRows>
  CUDA_INLINE void write_legacy(uint32_t slice_id, uint32_t slice_count, uint32_t split_idx,
                                uint32_t first_row = 0, uint32_t first_column = 0,
                                uint32_t buffer_offset = 0, uint32_t valid_rows = kRows) {
    constexpr uint32_t total_write_int4s = kRows * kColumns / 8;
    constexpr uint32_t iters = CEIL_DIV(total_write_int4s, kNumMathThreads);
    uint32_t smem_base = offsetof(SharedStorage, reduce) / 128 % 8;

    PRAGMA_UNROLL
    for (uint32_t i = 0; i < iters; i++) {
      uint32_t index = ctx.math_thread_id() + kNumMathThreads * i;
      if (index < total_write_int4s) {
        uint32_t local_row = index / 8 % kRows;
        if (local_row >= valid_rows) continue;
        uint32_t column_block = index / (8 * kRows);
        uint32_t smem_row = (first_column / 64 + column_block) * kStorageRows + local_row;
        uint32_t smem_col = index % 8;
        uint32_t swizzled_col = smem_col ^ ((smem_row + smem_base) % 8);
        uint32_t smem_offset = buffer_offset + smem_row * 8 + swizzled_col;
        uint32_t gmem_row = first_row + local_row;
        if constexpr (kNumWriteSplits == 2) gmem_row += split_idx * (BlockShape::M / 2);
        if constexpr (kIsIndexedGemm) gmem_row = ctx.get_wr_row_index()[gmem_row];
        uint32_t gmem_col = first_column / 8 + column_block * 8 + smem_col;
        bool valid_row = gmem_row < (kIsIndexedGemm ? output_shape_m : block_output_shape_m);
        bool valid_column = PadShape::N == 0 || col_offset + gmem_col * 8 < ProblemShape::N - PadShape::N;
        if (!valid_row || !valid_column) continue;

        int4 val = ctx.smem.reduce[smem_offset];
        uint32_t gmem_offset = gmem_row * ((ProblemShape::N - PadShape::N) / 8) + gmem_col;
        if (!kUseStreamK || slice_count == 1 || slice_id == 0) {
          gmem_ptr[gmem_offset] = val;
        } else if (slice_count > 3) {
          atomic_reduce_add_f162<scalar_t2>(val, gmem_ptr[gmem_offset]);
        } else {
          gmem_ptr[gmem_offset] = reduce_add_f162<scalar_t2>(val, gmem_ptr[gmem_offset]);
        }
      }
    }
  }

  CUDA_INLINE
  void write_chunk(uint32_t slice_id, uint32_t slice_count, uint32_t first_row,
                   uint32_t rows, uint32_t buffer_offset) {
    constexpr uint32_t kRows = Ctx::kUmmaOutputChunkRows;
    constexpr uint32_t kColumns = 128;
    uint32_t first_column = ctx.math_group * 128;
    if constexpr (kUseTmaC) {
      // The descriptor height divides block M so a tail cannot overwrite the next tile.
      constexpr uint32_t kTmaRows = BlockShape::M % 32 == 0 ? 32 : (BlockShape::M % 16 == 0 ? 16 : 8);
      uint32_t column_block = ctx.math_thread_id();
      if (column_block < kColumns / 64) {
        uint32_t smem_offset = buffer_offset + (first_column / 64 + column_block) * kRows * 8;
        PRAGMA_UNROLL
        for (uint32_t row = 0; row < rows; row += kTmaRows) {
          write_tma_tile(slice_id, slice_count, smem_offset + row * 8,
                         col_offset + first_column + column_block * 64, row_offset + first_row + row);
        }
      }
    } else {
      write_legacy<kRows, kColumns>(slice_id, slice_count, 0, first_row, first_column, buffer_offset, rows);
    }
  }

  CUDA_INLINE
  void write_tma_tile(uint32_t slice_id, uint32_t slice_count, uint32_t smem_offset,
                      uint32_t column, uint32_t row) {
    if (!kUseStreamK || slice_count == 1 || slice_id == 0)
      tma_store_2d(ctx.smem.reduce + smem_offset, tensor_map_ptr, column, row);
    else
      tma_reduce_add_2d(ctx.smem.reduce + smem_offset, tensor_map_ptr, column, row);
    tma_commit_store_group();
  }

  CUDA_INLINE
  void write_tma(uint32_t slice_id, uint32_t slice_count) {
    static_assert(!kIsIndexedGemm);
    constexpr uint32_t count = BlockShape::N / 64;
    const uint32_t block_idx = ctx.math_thread_id();
    const uint32_t smem_offset = BlockShape::M * 64 / 8 * block_idx;
    const uint32_t col_offset2 = col_offset + 64 * block_idx;
    if (block_idx < count) {
      write_tma_tile(slice_id, slice_count, smem_offset, col_offset2, row_offset);
      if constexpr (kUseStreamK) {
        if (slice_count > 1 && slice_id != slice_count - 1) tma_wait_store_group<0>();
      }
    }
  }

  CUDA_INLINE
  void seek(uint32_t m_block_id, uint32_t n_block_id, uint32_t current_shape_m, uint32_t m_offset) {
    if constexpr (kIsGroupedGemm) {
      output_shape_m = current_shape_m;
      row_offset = m_offset;
    } else {
      row_offset = m_block_id * BlockShape::M;
    }
    block_output_shape_m = row_offset < output_shape_m ? MIN(output_shape_m - row_offset, BlockShape::M) : 0;
    col_offset = n_block_id * BlockShape::N;

    uint32_t offset;
    offset = n_block_id * (BlockShape::N * 2 / 16);
    if constexpr (!kIsIndexedGemm) {
      constexpr uint32_t kShapeN = ProblemShape::N - PadShape::N;
      offset += MIN(row_offset, output_shape_m) * (kShapeN / 8);
    }
    gmem_ptr = gmem_ptr_raw + offset;
  };

  CUDA_INLINE
  void update_tensor_map_ptr(const CUtensorMap *tensor_map_ptr_) {
    tensor_map_ptr = tensor_map_ptr_;
  };
};
