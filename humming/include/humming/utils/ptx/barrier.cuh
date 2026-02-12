#pragma once

#include <humming/utils/base.cuh>


template <uint32_t kNumSyncThreads, uint32_t kNumThreads>
CUDA_INLINE uint32_t sync_part_threads() {
  if constexpr (kNumSyncThreads == kNumThreads) {
    __syncthreads();
  } else {
    static_assert(kNumThreads >= kNumSyncThreads);
    static_assert(kNumSyncThreads > 0);
    constexpr uint32_t barrier_id = 1;
    asm volatile("bar.sync %0, %1;" : : "r"(barrier_id), "r"(kNumSyncThreads));
  }
}

template <uint32_t kNumSyncThreads = 0, uint32_t kNumThreads = 0>
CUDA_INLINE void barrier_acquire(int *lock, int count) {
  if (threadIdx.x == 0) {
    int state = -1;
    do {
      asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n"
                   : "=r"(state)
                   : "l"(lock));
    } while (state != count);
  }
  sync_part_threads<kNumSyncThreads, kNumThreads>();
}

template <uint32_t kNumSyncThreads = 0, uint32_t kNumThreads = 0>
CUDA_INLINE void barrier_release(int *lock, bool reset = false) {
  sync_part_threads<kNumSyncThreads, kNumThreads>();
  if (threadIdx.x == 0) {
    if (reset) {
      lock[0] = 0;
    } else {
      int32_t val = 1;
      asm volatile("fence.acq_rel.gpu;\n");
      asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n"
                   :
                   : "l"(lock), "r"(val));
    }
  }
}

CUDA_INLINE
void mbarrier_wait(uint64_t *barrier, bool phase_parity) {
  uint32_t smem_int_mbar = cast_smem_ptr_to_uint(barrier);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("{\n"
               "  .reg .pred p;\n"
               "  waitLoop:\n"
               "  mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
               "  @p bra done;\n"
               "  bra waitLoop;\n"
               "  done:\n"
               "}\n"
               :
               : "r"(smem_int_mbar), "r"((uint32_t)phase_parity)
               : "memory");
#else
  asm volatile("{\n"
               "  .reg .pred p;\n"
               "  waitLoop:\n"
               "  mbarrier.test_wait.parity.shared::cta.b64 p, [%0], %1;\n"
               "  @p bra done;\n"
               "  bra waitLoop;\n"
               "  done:\n"
               "}\n"
               :
               : "r"(smem_int_mbar), "r"((uint32_t)phase_parity)
               : "memory");
#endif
};
