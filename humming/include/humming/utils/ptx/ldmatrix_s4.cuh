#pragma once

#include <humming/utils/base.cuh>

// PTX ISA 9.4, available with CUDA 13.4+.
constexpr bool kLdmatrixS4Ptx94Available =
#if defined(__CUDACC_VER_MAJOR__) && \
    (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4))
    true;
#else
    false;
#endif

// Loads four contiguous x4 subtiles into `regs`.
// Guard call sites with `if constexpr` on supported targets.
CUDA_INLINE void ld_shared_s4x4(const void* smem_ptr, uint32_t* regs) {
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "ldmatrix.sync.aligned.m8n16.x4.shared::cta.s8.s4 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
      : "r"(smem));
}
