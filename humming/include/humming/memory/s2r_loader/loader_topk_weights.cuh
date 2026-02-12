#pragma once

#include <humming/utils/all.cuh>


template <class BlockShape, class WarpShape>
class S2RMemoryLoaderTopKWeights {
private:
  static constexpr uint32_t M_WARPS = BlockShape::M / WarpShape::M;
  static constexpr uint32_t N_WARPS = BlockShape::N / WarpShape::N;

public:
  CUDA_INLINE
  void load(const void *smem_ptr, uint32_t *regs_ptr) {
    const uint32_t *smem_ptr_load = reinterpret_cast<const uint32_t *>(smem_ptr);
    uint32_t sub_row = (threadIdx.x % 32) / 4;
    uint32_t offset = 0;

    if constexpr (M_WARPS > 1) {
      offset += ((threadIdx.x / 32) / N_WARPS % M_WARPS) * WarpShape::M;
    }

    PRAGMA_UNROLL
    for (uint32_t i = 0; i < WarpShape::M / 8; i++) {
      uint32_t row = i * 8 + sub_row;
      regs_ptr[i] = smem_ptr_load[row];
    }
  }
};
