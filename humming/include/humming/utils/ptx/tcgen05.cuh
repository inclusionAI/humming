#pragma once

#include <humming/utils/base.cuh>


template <uint32_t kColumns, uint32_t kCtaGroupSize = 1>
CUDA_INLINE void tcgen05_alloc(uint32_t smem_address) {
  static_assert(kColumns >= 32 && kColumns <= 512 && !(kColumns & (kColumns - 1)));
  if constexpr (kCtaGroupSize == 2) {
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_address), "n"(kColumns) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" ::: "memory");
  } else {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_address), "n"(kColumns) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
  }
}


template <uint32_t kColumns, uint32_t kCtaGroupSize = 1>
CUDA_INLINE void tcgen05_dealloc(uint32_t tmem_address) {
  if constexpr (kCtaGroupSize == 2)
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" ::"r"(tmem_address), "n"(kColumns) : "memory");
  else asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem_address), "n"(kColumns) : "memory");
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


template <uint32_t kCtaGroupSize = 1>
CUDA_INLINE void tcgen05_commit(uint32_t mbarrier_address) {
  if constexpr (kCtaGroupSize == 2)
    asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.multicast::cluster.b64 [%0], %1;" ::"r"(mbarrier_address), "h"(uint16_t(3)) : "memory");
  else asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(mbarrier_address) : "memory");
}


template <uint32_t kSwizzleBytes>
CUDA_INLINE uint64_t tcgen05_smem_desc(const void *ptr) {
  static_assert(kSwizzleBytes == 64 || kSwizzleBytes == 128);
  constexpr uint64_t kSwizzleMode = kSwizzleBytes == 128 ? 2 : 4;
  constexpr uint64_t kStride = kSwizzleBytes / 2;
  return ((uint64_t(cast_smem_ptr_to_uint(ptr)) >> 4) & 0x3fff) | (uint64_t(1) << 16) | (kStride << 32) | (uint64_t(1) << 46) | (kSwizzleMode << 61);
}


// Elect once for a group of MMA instructions and their completion commits.
CUDA_INLINE bool tcgen05_elect_leader() {
  uint32_t leader;
  asm volatile("{ .reg .pred p; elect.sync _|p, 0xffffffff; selp.u32 %0, 1, 0, p; }" : "=r"(leader));
  return leader;
}


template <uint32_t kN, bool kUseBf16, uint32_t kCtaGroupSize = 1>
CUDA_INLINE void tcgen05_mma_f16(uint32_t d, uint32_t a, uint64_t b, bool accumulate) {
  constexpr uint32_t input_format = kUseBf16 ? (1u << 7) | (1u << 10) : 0u;
  constexpr uint32_t descriptor = (1u << 4) | input_format | ((kN / 8) << 17) | ((8u * kCtaGroupSize) << 24);
  if constexpr (kCtaGroupSize == 2) {
    asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; "
                 "tcgen05.mma.cta_group::2.kind::f16 [%0], [%1], %2, %3, {%5,%5,%5,%5,%5,%5,%5,%5}, p; }" ::"r"(d),
                 "r"(a), "l"(b), "r"(descriptor), "r"(uint32_t(accumulate)), "r"(0u) : "memory");
  } else {
    asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; "
                 "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, {%5,%5,%5,%5}, p; }" ::"r"(d),
                 "r"(a), "l"(b), "r"(descriptor), "r"(uint32_t(accumulate)), "r"(0u) : "memory");
  }
}


template <uint32_t kN, uint32_t kAFormat, uint32_t kBFormat, uint32_t kCtaGroupSize = 1>
CUDA_INLINE void tcgen05_mma_f8f6f4(uint32_t d, uint32_t a, uint64_t b, bool accumulate) {
  constexpr uint32_t input_format = (kAFormat << 7) | (kBFormat << 10);
  constexpr uint32_t descriptor = (1u << 4) | input_format | ((kN / 8) << 17) | ((8u * kCtaGroupSize) << 24);
  if constexpr (kCtaGroupSize == 2) {
    asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; "
                 "tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [%1], %2, %3, {%5,%5,%5,%5,%5,%5,%5,%5}, p; }" ::"r"(d),
                 "r"(a), "l"(b), "r"(descriptor), "r"(uint32_t(accumulate)), "r"(0u) : "memory");
  } else {
    asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; "
                 "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%1], %2, %3, {%5,%5,%5,%5}, p; }" ::"r"(d),
                 "r"(a), "l"(b), "r"(descriptor), "r"(uint32_t(accumulate)), "r"(0u) : "memory");
  }
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


template <uint32_t kN, uint32_t kWeightFormat, uint32_t kInputFormat, uint32_t kCtaGroupSize = 1>
CUDA_INLINE void tcgen05_mma_mxf8f6f4(uint32_t d, uint32_t a, uint64_t b,
                                      uint32_t sfa, uint32_t sfb, uint32_t scale_id, bool accumulate) {
  constexpr uint32_t descriptor_base = (kWeightFormat << 7) | (kInputFormat << 10) |
                                       ((kN / 8) << 17) | (1u << 23) | (kCtaGroupSize << 27);
  uint32_t descriptor = descriptor_base | (scale_id << 4) | (scale_id << 29);
  if constexpr (kCtaGroupSize == 2) {
    asm volatile("{ .reg .pred p; setp.ne.b32 p, %6, 0; "
                 "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.block32 "
                 "[%0], [%1], %2, %3, [%4], [%5], p; }" ::"r"(d),
                 "r"(a), "l"(b), "r"(descriptor), "r"(sfa), "r"(sfb), "r"(uint32_t(accumulate)) : "memory");
  } else {
    asm volatile("{ .reg .pred p; setp.ne.b32 p, %6, 0; "
                 "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.block32 "
                 "[%0], [%1], %2, %3, [%4], [%5], p; }" ::"r"(d),
                 "r"(a), "l"(b), "r"(descriptor), "r"(sfa), "r"(sfb), "r"(uint32_t(accumulate)) : "memory");
  }
}


CUDA_INLINE void tcgen05_st_32x32b(uint32_t address, uint32_t value) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};" ::"r"(address), "r"(value) : "memory");
}


template <uint32_t kCount>
CUDA_INLINE void tcgen05_st_32x32b(uint32_t address, const uint32_t *values) {
  static_assert(kCount >= 1 && kCount <= 8);
  if constexpr (kCount == 8) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" ::"r"(address), "r"(values[0]), "r"(values[1]), "r"(values[2]), "r"(values[3]),
                 "r"(values[4]), "r"(values[5]), "r"(values[6]), "r"(values[7]) : "memory");
  } else if constexpr (kCount >= 4) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1, %2, %3, %4};" ::"r"(address), "r"(values[0]), "r"(values[1]), "r"(values[2]), "r"(values[3]) : "memory");
    if constexpr (kCount > 4) tcgen05_st_32x32b<kCount - 4>(address + 4, values + 4);
  } else if constexpr (kCount >= 2) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x2.b32 [%0], {%1, %2};" ::"r"(address), "r"(values[0]), "r"(values[1]) : "memory");
    if constexpr (kCount > 2) tcgen05_st_32x32b<1>(address + 2, values + 2);
  } else {
    tcgen05_st_32x32b(address, values[0]);
  }
}


CUDA_INLINE void tcgen05_ld_16x256b_x4(uint32_t address, uint32_t *values) {
  asm volatile("tcgen05.ld.sync.aligned.16x256b.x4.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
               : "=r"(values[0]), "=r"(values[1]), "=r"(values[2]), "=r"(values[3]),
                 "=r"(values[4]), "=r"(values[5]), "=r"(values[6]), "=r"(values[7]),
                 "=r"(values[8]), "=r"(values[9]), "=r"(values[10]), "=r"(values[11]),
                 "=r"(values[12]), "=r"(values[13]), "=r"(values[14]), "=r"(values[15]) : "r"(address) : "memory");
}


template <uint32_t kN, uint32_t kGroupSize, bool kScaleIsE4M3, uint32_t kCtaGroupSize = 1>
CUDA_INLINE void tcgen05_mma_mxf4nvf4(uint32_t d, uint32_t a, uint64_t b,
                                      uint32_t sfa, uint32_t sfb, uint32_t scale_id, bool accumulate) {
  constexpr uint32_t descriptor_base = (1u << 7) | (1u << 10) | ((kN / 8) << 17) |
                                       (uint32_t(!kScaleIsE4M3) << 23) | (kCtaGroupSize << 27);
  uint32_t descriptor = descriptor_base | (scale_id << 4) | (scale_id << 29);
  if constexpr (kCtaGroupSize == 2 && kGroupSize == 16) {
    asm volatile("{ .reg .pred p; setp.ne.b32 p, %6, 0; "
                 "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16 "
                 "[%0], [%1], %2, %3, [%4], [%5], p; }" ::"r"(d),
                 "r"(a), "l"(b), "r"(descriptor), "r"(sfa), "r"(sfb), "r"(uint32_t(accumulate)) : "memory");
  } else if constexpr (kCtaGroupSize == 2 && kGroupSize == 32) {
    asm volatile("{ .reg .pred p; setp.ne.b32 p, %6, 0; "
                 "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block32 "
                 "[%0], [%1], %2, %3, [%4], [%5], p; }" ::"r"(d),
                 "r"(a), "l"(b), "r"(descriptor), "r"(sfa), "r"(sfb), "r"(uint32_t(accumulate)) : "memory");
  } else if constexpr (kCtaGroupSize == 1 && kGroupSize == 16) {
    asm volatile("{ .reg .pred p; setp.ne.b32 p, %6, 0; "
                 "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 "
                 "[%0], [%1], %2, %3, [%4], [%5], p; }" ::"r"(d),
                 "r"(a), "l"(b), "r"(descriptor), "r"(sfa), "r"(sfb), "r"(uint32_t(accumulate)) : "memory");
  } else if constexpr (kCtaGroupSize == 1 && kGroupSize == 32) {
    asm volatile("{ .reg .pred p; setp.ne.b32 p, %6, 0; "
                 "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block32 "
                 "[%0], [%1], %2, %3, [%4], [%5], p; }" ::"r"(d),
                 "r"(a), "l"(b), "r"(descriptor), "r"(sfa), "r"(sfb), "r"(uint32_t(accumulate)) : "memory");
  }
}
