#pragma once

#include <humming/utils/all.cuh>


template <typename scalar_t2, typename T>
CUDA_INLINE T reduce_add_f162(T a, T b) {
  scalar_t2 *a_half2_ptr = reinterpret_cast<scalar_t2 *>(&a);
  scalar_t2 *b_half2_ptr = reinterpret_cast<scalar_t2 *>(&b);

  PRAGMA_UNROLL
  for (uint32_t i = 0; i < sizeof(T) / 4; i++) {
    a_half2_ptr[i] = __hadd2(a_half2_ptr[i], b_half2_ptr[i]);
  };
  return a;
};

template <
    class ArithClass,
    class ProblemShape, class BlockShape, class PadShape, class ElementC,
    class SchedulerConfig, class PipelineConfig, class EpilogueConfig, class MoEConfig>
class EpilogueGmemWriter : F16Conversion<ElementC> {
private:
  static constexpr bool kUseStreamK = SchedulerConfig::kUseStreamK;
  static constexpr bool kUseTmaC = PipelineConfig::kUseTmaC;
  static constexpr bool kIsMoE = MoEConfig::kIsMoE;
  static constexpr bool kIsMoEDown = MoEConfig::kIsMoEDown;
  static constexpr uint32_t kTopK = MoEConfig::kTopK;

  static constexpr bool kHasActivation = EpilogueConfig::kActivationType != ActivationType::NONE;
  static constexpr bool kHasGLUActivation = EpilogueConfig::kActivationType == ActivationType::SILU_GLU ||
                                            EpilogueConfig::kActivationType == ActivationType::CUSTOM_GLU;
  static constexpr bool kHasIndentityActivation = kHasActivation && !kHasGLUActivation;

  static constexpr uint32_t kNumMathThreads = PipelineConfig::kNumMathThreads;

  using scalar_t = typename F16Conversion<ElementC>::scalar_t;
  using scalar_t2 = typename F16Conversion<ElementC>::scalar_t2;
  using OutputPtrType = std::conditional_t<kUseTmaC, const void *, void *>;

public:
  ArithClass &arith;
  int4 *smem_ptr;
  int4 *gmem_ptr_raw;
  int4 *gmem_ptr;
  const CUtensorMap *tensor_map_ptr;
  const uint32_t *moe_row_index;

  uint32_t row_offset;
  uint32_t col_offset;
  uint32_t output_shape_m;
  uint32_t block_output_shape_m;

  CUDA_INLINE
  EpilogueGmemWriter(
      ArithClass &arith,
      int4 *smem_ptr, OutputPtrType output_ptr,
      const uint32_t *moe_row_index, uint32_t shape_m)
      : arith(arith), smem_ptr(smem_ptr), moe_row_index(moe_row_index) {

    if constexpr (kUseTmaC) {
      tensor_map_ptr = reinterpret_cast<const CUtensorMap *>(output_ptr);
    } else {
      gmem_ptr_raw = reinterpret_cast<int4 *>(output_ptr);
    }

    if constexpr (kIsMoE && !kIsMoEDown) {
      output_shape_m = shape_m * kTopK;
    } else {
      output_shape_m = shape_m;
    }
  }

  CUDA_INLINE
  void write(uint32_t slice_id, uint32_t slice_count) {
    constexpr bool kHasActivation = EpilogueConfig::kActivationType != ActivationType::NONE;
    if constexpr (kUseTmaC && kIsMoE) {
      write_tma_moe(slice_id, slice_count);
    } else if constexpr (kUseTmaC && !kIsMoE) {
      write_tma(slice_id, slice_count);
    } else {
      write_legacy(slice_id, slice_count);
    }
  };

  CUDA_INLINE
  void write_legacy(uint32_t slice_id, uint32_t slice_count) {
    constexpr uint32_t total_write_int4s = BlockShape::M * BlockShape::N * 2 / 16;
    constexpr bool is_full_div = total_write_int4s % kNumMathThreads == 0;
    constexpr uint32_t iters = CEIL_DIV(total_write_int4s, kNumMathThreads);
    uint32_t smem = cast_smem_ptr_to_uint(smem_ptr) / 128;

    PRAGMA_UNROLL
    for (uint32_t i = 0; i < iters; i++) {
      uint32_t smem_offset = threadIdx.x + kNumMathThreads * i;
      if (is_full_div || i != iters - 1 || smem_offset < total_write_int4s) {
        uint32_t smem_row = smem_offset / 8;
        uint32_t smem_col = smem_offset % 8;

        uint32_t smem_col_swizzled = smem_col ^ ((smem_row + smem) % 8);
        uint32_t smem_offset_swizzled = smem_row * 8 + smem_col_swizzled;
        uint32_t gmem_row = smem_row % BlockShape::M;
        gmem_row = kIsMoE ? moe_row_index[gmem_row] : gmem_row;
        uint32_t gmem_col = smem_row / BlockShape::M * 8 + smem_col;
        bool pred1 = gmem_row < (kIsMoE ? output_shape_m : block_output_shape_m);
        bool pred2 = PadShape::N == 0 || (col_offset + gmem_col * 8 < ProblemShape::N - PadShape::N);

        if (!pred1 || !pred2) continue;

        int4 val = smem_ptr[smem_offset_swizzled];
        bool should_apply_activation = kHasActivation && (kUseStreamK && slice_count > 1);
        bool should_apply_glu_activation = kHasGLUActivation && should_apply_activation;
        bool should_apply_identity_activation = kHasIndentityActivation && should_apply_activation;

        uint32_t gmem_offset = gmem_row * ((ProblemShape::N - PadShape::N) / 8) + gmem_col;
        if (!should_apply_activation && !kHasGLUActivation) {
          if (!kUseStreamK || slice_count == 1 || slice_id == 0) {
            gmem_ptr[gmem_offset] = val;
          } else {
            gmem_ptr[gmem_offset] = reduce_add_f162<scalar_t2>(val, gmem_ptr[gmem_offset]);
          }
        } else if (!should_apply_activation && kHasGLUActivation && slice_count == 1) {
          int2 *gmem_ptr_int2 = reinterpret_cast<int2 *>(gmem_ptr);
          scalar_t *val_ptr = reinterpret_cast<scalar_t *>(&val);
          val_ptr[1] = val_ptr[2];
          val_ptr[2] = val_ptr[4];
          val_ptr[3] = val_ptr[6];
          gmem_ptr_int2[gmem_offset] = reinterpret_cast<int2 *>(&val)[0];
        } else if (should_apply_identity_activation) {
          if (slice_id > 0) val = reduce_add_f162<scalar_t2>(val, gmem_ptr[gmem_offset]);
          if (slice_id == slice_count - 1) arith.apply_activation_on_gmem_write(val);
          gmem_ptr[gmem_offset] = val;
        } else if (should_apply_glu_activation) {
          int2 *gmem_ptr_int2 = reinterpret_cast<int2 *>(gmem_ptr);
          int2 *val_int2_ptr = reinterpret_cast<int2 *>(&val);
          uint32_t gmem_offset2 = gmem_offset + output_shape_m * (ProblemShape::N * 2 / 16);
          if (slice_id > 0) {
            val_int2_ptr[0] = reduce_add_f162<scalar_t2>(val_int2_ptr[0], gmem_ptr_int2[gmem_offset]);
            val_int2_ptr[1] = reduce_add_f162<scalar_t2>(val_int2_ptr[1], gmem_ptr_int2[gmem_offset2]);
          }

          if (slice_id == slice_count - 1) arith.apply_activation_on_gmem_write(val);
          gmem_ptr_int2[gmem_offset] = val_int2_ptr[0];
          if (slice_id != slice_count - 1) gmem_ptr_int2[gmem_offset2] = val_int2_ptr[1];
        }
      };
    };
  };

  CUDA_INLINE
  void write_tma(uint32_t slice_id, uint32_t slice_count) {
    constexpr uint32_t count = BlockShape::N / 64;
    const uint32_t block_idx = threadIdx.x;
    const uint32_t smem_offset = BlockShape::M * 64 / 8 * block_idx;
    const uint32_t col_offset2 = col_offset + 64 * block_idx;
    if (block_idx < count) {
      if constexpr (!kUseStreamK) {
        tma_store_2d(smem_ptr + smem_offset, tensor_map_ptr, col_offset2, row_offset);
      } else if (slice_count == 1 || slice_id == 0) {
        tma_store_2d(smem_ptr + smem_offset, tensor_map_ptr, col_offset2, row_offset);
        if (slice_count > 1) tma_wait_store_group<0>();
      } else {
        tma_reduce_add_2d(smem_ptr + smem_offset, tensor_map_ptr, col_offset2, row_offset);
        if (slice_id != slice_count - 1) tma_wait_store_group<0>();
      }
    }
  }

  CUDA_INLINE
  void write_tma_moe(uint32_t slice_id, uint32_t slice_count) {
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < CEIL_DIV(BlockShape::M, kNumMathThreads); i++) {
      uint32_t row = threadIdx.x + kNumMathThreads * i;
      if (row >= BlockShape::M) break;
      uint32_t gmem_row = moe_row_index[row];

      PRAGMA_UNROLL
      for (uint32_t block_idx = 0; block_idx < (BlockShape::N / 64); block_idx++) {
        uint32_t col_offset2 = col_offset + 64 * block_idx;
        const uint32_t smem_offset = row * 8 + BlockShape::M * 64 / 8 * block_idx;
        if constexpr (!kUseStreamK) {
          tma_store_2d(smem_ptr + smem_offset, tensor_map_ptr, col_offset2, gmem_row);
        } else if (slice_count == 1 || slice_id == 0) {
          tma_store_2d(smem_ptr + smem_offset, tensor_map_ptr, col_offset2, gmem_row);
          if (slice_count > 1) tma_wait_store_group<0>();
        } else {
          tma_reduce_add_2d(smem_ptr + smem_offset, tensor_map_ptr, col_offset2, gmem_row);
          if (slice_id != slice_count - 1) tma_wait_store_group<0>();
        }
      }
    }
  }

  CUDA_INLINE
  void seek(uint32_t m_block_id, uint32_t n_block_id) {
    row_offset = m_block_id * BlockShape::M;
    block_output_shape_m = output_shape_m - row_offset;
    col_offset = n_block_id * BlockShape::N;

    uint32_t offset;
    if constexpr (kHasGLUActivation) {
      offset = n_block_id * (BlockShape::N / 2 * 2 / 16);
    } else {
      offset = n_block_id * (BlockShape::N * 2 / 16);
    }
    if constexpr (!kIsMoE) {
      constexpr uint32_t kShapeN = ProblemShape::N - PadShape::N;
      if constexpr (kHasGLUActivation) {
        offset += m_block_id * (kShapeN / 2 * BlockShape::M * 2 / 16);
      } else {
        offset += m_block_id * (kShapeN * BlockShape::M * 2 / 16);
      }
    }
    gmem_ptr = gmem_ptr_raw + offset;
  };
};
