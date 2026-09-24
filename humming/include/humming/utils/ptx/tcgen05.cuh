#pragma once

#include <humming/utils/base.cuh>


template <uint32_t kColumns>
CUDA_INLINE void tcgen05_alloc(uint32_t smem_address) {
  static_assert(kColumns >= 32 && kColumns <= 512 && !(kColumns & (kColumns - 1)));
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_address), "n"(kColumns) : "memory");
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
}


template <uint32_t kColumns>
CUDA_INLINE void tcgen05_dealloc(uint32_t tmem_address) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem_address), "n"(kColumns) : "memory");
}


CUDA_INLINE void tcgen05_fence_before_thread_sync() {
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}


CUDA_INLINE void tcgen05_fence_after_thread_sync() {
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}


CUDA_INLINE void tcgen05_wait_st() {
  asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}


CUDA_INLINE void tcgen05_wait_ld() {
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}


CUDA_INLINE void tcgen05_commit(uint32_t mbarrier_address) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(mbarrier_address) : "memory");
}


template <uint32_t kSwizzleBytes>
CUDA_INLINE uint64_t tcgen05_smem_desc_f16(const void *ptr) {
  static_assert(kSwizzleBytes == 64 || kSwizzleBytes == 128);
  constexpr uint64_t kSwizzleMode = kSwizzleBytes == 128 ? 2 : 4;
  constexpr uint64_t kStride = kSwizzleBytes / 2;
  return ((uint64_t(cast_smem_ptr_to_uint(ptr)) >> 4) & 0x3fff) | (uint64_t(1) << 16) | (kStride << 32) | (uint64_t(1) << 46) | (kSwizzleMode << 61);
}


template <uint32_t kN, bool kUseBf16>
CUDA_INLINE void tcgen05_mma_f16(uint32_t d, uint32_t a, uint64_t b, bool accumulate) {
  constexpr uint32_t input_format = kUseBf16 ? (1u << 7) | (1u << 10) : 0u;
  constexpr uint32_t descriptor = (1u << 4) | input_format | ((kN / 8) << 17) | (8u << 24);
  asm volatile(
      "{\n"
      "  .reg .pred p, leader;\n"
      "  elect.sync _|leader, 0xffffffff;\n"
      "  setp.ne.b32 p, %4, 0;\n"
      "  @leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, {%5, %5, %5, %5}, p;\n"
      "}\n" ::"r"(d),
      "r"(a), "l"(b), "r"(descriptor), "r"(uint32_t(accumulate)), "r"(0u)
      : "memory");
}


CUDA_INLINE void tcgen05_st_16x128b_x2(uint32_t address, const uint32_t *values) {
  asm volatile("tcgen05.st.sync.aligned.16x128b.x2.b32 [%0], {%1, %2, %3, %4};" ::"r"(address), "r"(values[0]), "r"(values[2]),
               "r"(values[1]), "r"(values[3]) : "memory");
}


CUDA_INLINE void tcgen05_st_16x128b_x4(uint32_t address, const uint32_t *first, const uint32_t *second) {
  asm volatile(
      "tcgen05.st.sync.aligned.16x128b.x4.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" ::"r"(address), "r"(first[0]), "r"(first[2]), "r"(first[1]), "r"(first[3]),
      "r"(second[0]), "r"(second[2]), "r"(second[1]), "r"(second[3]) : "memory");
}


CUDA_INLINE void tcgen05_st_16x128b_x8(uint32_t address, const uint32_t *first, const uint32_t *second,
                                       const uint32_t *third, const uint32_t *fourth) {
  asm volatile(
      "tcgen05.st.sync.aligned.16x128b.x8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};" ::"r"(address), "r"(first[0]), "r"(first[2]), "r"(first[1]), "r"(first[3]),
      "r"(second[0]), "r"(second[2]), "r"(second[1]), "r"(second[3]),
      "r"(third[0]), "r"(third[2]), "r"(third[1]), "r"(third[3]),
      "r"(fourth[0]), "r"(fourth[2]), "r"(fourth[1]), "r"(fourth[3]) : "memory");
}


CUDA_INLINE void tcgen05_ld_32x32b_x8(uint32_t address, uint32_t *values) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
      : "=r"(values[0]), "=r"(values[1]), "=r"(values[2]), "=r"(values[3]),
        "=r"(values[4]), "=r"(values[5]), "=r"(values[6]), "=r"(values[7])
      : "r"(address) : "memory");
}


CUDA_INLINE void tcgen05_ld_16x128b_x8(uint32_t address, uint32_t *values) {
  asm volatile(
      "tcgen05.ld.sync.aligned.16x128b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
      : "=r"(values[0]), "=r"(values[1]), "=r"(values[2]), "=r"(values[3]),
        "=r"(values[4]), "=r"(values[5]), "=r"(values[6]), "=r"(values[7]),
        "=r"(values[8]), "=r"(values[9]), "=r"(values[10]), "=r"(values[11]),
        "=r"(values[12]), "=r"(values[13]), "=r"(values[14]), "=r"(values[15])
      : "r"(address) : "memory");
}


CUDA_INLINE void tcgen05_ld_16x128b_x4(uint32_t address, uint32_t *values) {
  asm volatile(
      "tcgen05.ld.sync.aligned.16x128b.x4.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
      : "=r"(values[0]), "=r"(values[1]), "=r"(values[2]), "=r"(values[3]),
        "=r"(values[4]), "=r"(values[5]), "=r"(values[6]), "=r"(values[7])
      : "r"(address) : "memory");
}


CUDA_INLINE void tcgen05_ld_16x128b_x2(uint32_t address, uint32_t *values) {
  asm volatile(
      "tcgen05.ld.sync.aligned.16x128b.x2.b32 {%0, %1, %2, %3}, [%4];"
      : "=r"(values[0]), "=r"(values[1]), "=r"(values[2]), "=r"(values[3])
      : "r"(address) : "memory");
}
