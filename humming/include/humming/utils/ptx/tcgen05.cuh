#pragma once

#include <humming/utils/base.cuh>


template <uint32_t kColumns>
CUDA_INLINE void tcgen05_alloc(uint32_t smem_address) {
  static_assert(kColumns >= 32 && kColumns <= 512 && !(kColumns & (kColumns - 1)));
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
               :: "r"(smem_address), "n"(kColumns) : "memory");
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
}


template <uint32_t kColumns>
CUDA_INLINE void tcgen05_dealloc(uint32_t tmem_address) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
               :: "r"(tmem_address), "n"(kColumns) : "memory");
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
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
               :: "r"(mbarrier_address) : "memory");
}


CUDA_INLINE uint64_t tcgen05_smem_desc_bf16(const void *ptr) {
  // K-major, 64 BF16 per row, 128-byte swizzle.
  return ((uint64_t(cast_smem_ptr_to_uint(ptr)) >> 4) & 0x3fff)
         | (uint64_t(1) << 16) | (uint64_t(64) << 32)
         | (uint64_t(1) << 46) | (uint64_t(2) << 61);
}


template <uint32_t kN>
CUDA_INLINE void tcgen05_mma_bf16(uint32_t d, uint32_t a, uint64_t b, bool accumulate) {
  constexpr uint32_t descriptor = (1u << 4) | (1u << 7) | (1u << 10)
                                  | ((kN / 8) << 17) | (8u << 24);
  asm volatile(
      "{\n"
      "  .reg .pred p;\n"
      "  setp.ne.b32 p, %4, 0;\n"
      "  tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, {%5, %5, %5, %5}, p;\n"
      "}\n"
      :: "r"(d), "r"(a), "l"(b), "r"(descriptor), "r"(uint32_t(accumulate)), "r"(0u)
      : "memory");
}


CUDA_INLINE void tcgen05_st_16x128b_x2(uint32_t address, const uint32_t *values) {
  asm volatile("tcgen05.st.sync.aligned.16x128b.x2.b32 [%0], {%1, %2, %3, %4};"
               :: "r"(address), "r"(values[0]), "r"(values[2]),
                  "r"(values[1]), "r"(values[3]) : "memory");
}


CUDA_INLINE void tcgen05_ld_32x32b_x8(uint32_t address, uint32_t *values) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
      : "=r"(values[0]), "=r"(values[1]), "=r"(values[2]), "=r"(values[3]),
        "=r"(values[4]), "=r"(values[5]), "=r"(values[6]), "=r"(values[7])
      : "r"(address) : "memory");
}


CUDA_INLINE uint64_t tcgen05_smem_desc_fp8(const void *ptr) {
  // K-major, 64 FP8 per row, 64-byte swizzle.
  return ((uint64_t(cast_smem_ptr_to_uint(ptr)) >> 4) & 0x3fff)
         | (uint64_t(1) << 16) | (uint64_t(32) << 32)
         | (uint64_t(1) << 46) | (uint64_t(4) << 61);
}


template <uint32_t kN, bool kE5M2>
CUDA_INLINE void tcgen05_mma_fp8(uint32_t d, uint32_t a, uint64_t b, bool accumulate) {
  constexpr uint32_t descriptor = (1u << 4) | (uint32_t(kE5M2) << 7)
                                  | (uint32_t(kE5M2) << 10)
                                  | ((kN / 8) << 17) | (8u << 24);
  asm volatile(
      "{\n"
      "  .reg .pred p;\n"
      "  setp.ne.b32 p, %4, 0;\n"
      "  tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%1], %2, %3, {%5, %5, %5, %5}, p;\n"
      "}\n"
      :: "r"(d), "r"(a), "l"(b), "r"(descriptor), "r"(uint32_t(accumulate)), "r"(0u)
      : "memory");
}


CUDA_INLINE void tcgen05_st_32x32b_x4(uint32_t address, const uint32_t *values) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1, %2, %3, %4};"
               :: "r"(address), "r"(values[0]), "r"(values[1]),
                  "r"(values[2]), "r"(values[3]) : "memory");
}


template <uint32_t kN, uint32_t kAType, uint32_t kBType>
CUDA_INLINE void tcgen05_mma_mxfp8(uint32_t d, uint32_t a, uint64_t b,
                            uint32_t sfa, uint32_t sfb, uint32_t scale_id,
                            bool accumulate) {
  constexpr uint32_t descriptor = (kAType << 7) | (kBType << 10)
      | ((kN / 8) << 17) | (1u << 23) | (1u << 27);
  uint32_t idesc = descriptor | (scale_id << 29) | (scale_id << 4);
  asm volatile(
      "{ .reg .pred p; setp.ne.b32 p, %6, 0;\n"
      "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale "
      "[%0], [%1], %2, %3, [%4], [%5], p; }"
      :: "r"(d), "r"(a), "l"(b), "r"(idesc), "r"(sfa), "r"(sfb),
         "r"(uint32_t(accumulate)) : "memory");
}
