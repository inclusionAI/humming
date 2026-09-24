#pragma once

#include <humming/utils/base.cuh>

template <int count>
CUDA_INLINE void ld_shared(const int4 *smem_ptr, int4 *regs_ptr) {
  uint32_t *a = reinterpret_cast<uint32_t *>(regs_ptr);
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  if constexpr (count == 4) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(smem));
  } else if constexpr (count == 2) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(a[0]), "=r"(a[1])
                 : "r"(smem));
  } else if constexpr (count == 1) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                 : "=r"(a[0])
                 : "r"(smem));
  } else {
    static_assert(count == 1 || count == 2 || count == 4, "invalid count");
  }
}

template <int count, bool kTranspose = false>
CUDA_INLINE void st_shared(uint32_t smem, const uint32_t *a) {
  if constexpr (count == 4) {
    if constexpr (kTranspose) {
      asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared::cta.b16 [%0], {%1, %2, %3, %4};"
                   :: "r"(smem), "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]) : "memory");
    } else {
      asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};"
                   :: "r"(smem), "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]) : "memory");
    }
  } else if constexpr (count == 2) {
    if constexpr (kTranspose) {
      asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared::cta.b16 [%0], {%1, %2};"
                   :: "r"(smem), "r"(a[0]), "r"(a[1]) : "memory");
    } else {
      asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};"
                   :: "r"(smem), "r"(a[0]), "r"(a[1]) : "memory");
    }
  } else if constexpr (count == 1) {
    if constexpr (kTranspose) {
      asm volatile("stmatrix.sync.aligned.m8n8.x1.trans.shared::cta.b16 [%0], {%1};"
                   :: "r"(smem), "r"(a[0]) : "memory");
    } else {
      asm volatile("stmatrix.sync.aligned.m8n8.x1.shared.b16 [%0], {%1};"
                   :: "r"(smem), "r"(a[0]) : "memory");
    }
  } else {
    static_assert(count == 1 || count == 2 || count == 4, "invalid count");
  }
}


template <int count>
CUDA_INLINE void st_shared(const int4 *smem_ptr, int4 *regs_ptr) {
  const uint32_t *values = reinterpret_cast<const uint32_t *>(regs_ptr);
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  st_shared<count>(smem, values);
}


template <int count>
CUDA_INLINE void aiu_ld_shared(
    const void *smem_ptr, uint32_t *regs_ptr,
    uint32_t smem_dim,
    uint32_t smem_offset1, uint32_t smem_offset2) {
  uint32_t *a = reinterpret_cast<uint32_t *>(regs_ptr);
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  if constexpr (count == 4) {
    asm volatile("ppu.ldmatrix.sync.aligned.m8n8.x4.swzl.shared.b16 {%0, %1, %2, %3}, [%4], "
                 "{0, %5, 1, %6, 1, %7};\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(smem), "r"(smem_offset1), "r"(smem_dim), "r"(smem_offset2));
  } else if constexpr (count == 2) {
    asm volatile("ppu.ldmatrix.sync.aligned.m8n8.x2.swzl.shared.b16 {%0, %1}, [%2], "
                 "{0, %3, 1, %4, 1, %5};\n"
                 : "=r"(a[0]), "=r"(a[1])
                 : "r"(smem), "r"(smem_offset1), "r"(smem_dim), "r"(smem_offset2));
  } else {
    static_assert(count == 2 || count == 4, "invalid count");
  }
}
