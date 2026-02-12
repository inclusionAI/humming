#pragma once

#include <humming/utils/all.cuh>


template <class BlockShape, class MoEConfig>
class G2SMemoryLoaderTopKWeights {
public:
  const uint32_t *gmem_ptr;

  const uint32_t *row_index_ptr;
  const uint32_t shape_m;
  uint32_t row_index;

  CUDA_INLINE
  G2SMemoryLoaderTopKWeights(const void *ptr, const uint32_t *row_index_ptr, uint32_t shape_m)
      : row_index_ptr(row_index_ptr), shape_m(shape_m) {
    gmem_ptr = reinterpret_cast<const uint32_t *>(ptr);
  }

  CUDA_INLINE void load(void *smem_ptr) {
    uint32_t *smem_ptr_load = reinterpret_cast<uint32_t *>(smem_ptr);
    if constexpr (MoEConfig::kIsMoEDown) {
      if (row_index < shape_m) smem_ptr_load[threadIdx.x] = gmem_ptr[row_index];
    }
  }

  CUDA_INLINE void seek() {
    if (threadIdx.x < BlockShape::M) {
      row_index = row_index_ptr[threadIdx.x];
    } else {
      row_index = shape_m;
    }
  }
};
