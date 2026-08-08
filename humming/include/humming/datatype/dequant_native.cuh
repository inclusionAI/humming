#pragma once

#include <humming/datatype/dequant_prepare.cuh>
#include <humming/datatype/dtypes.cuh>
#include <humming/utils/all.cuh>


// CUDA 11.8, 12.7 and 13.2 are the first toolkits with PTX ISA 7.8, 8.6 and 9.2.
constexpr bool kNativeDequantPtx78Available =
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890 && defined(__CUDACC_VER_MAJOR__) && \
    (__CUDACC_VER_MAJOR__ > 11 || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 8))
    true;
#else
    false;
#endif


constexpr bool kNativeDequantLowBitFamily =
#if defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && __CUDA_ARCH_FAMILY_SPECIFIC__ >= 1000
    true;
#else
    false;
#endif


constexpr bool kNativeDequantPtx86Available =
#if defined(__CUDACC_VER_MAJOR__) && \
    (__CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 7))
    kNativeDequantLowBitFamily;
#else
    false;
#endif


constexpr bool kNativeDequantPtx92Available =
#if defined(__CUDACC_VER_MAJOR__) && \
    (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
    kNativeDequantLowBitFamily;
#else
    false;
#endif


template <class SourceType, class TargetType>
constexpr bool kNativeDequantSupported =
    std::is_same<TargetType, Float16>::value
        ? (kNativeDequantPtx78Available &&
               (std::is_same<SourceType, Float8E4M3>::value || std::is_same<SourceType, Float8E5M2>::value) ||
           kNativeDequantPtx86Available &&
               (std::is_same<SourceType, Float4E2M1>::value || std::is_same<SourceType, Float6E3M2>::value ||
                std::is_same<SourceType, Float6E2M3>::value))
        : std::is_same<TargetType, BFloat16>::value &&
              kNativeDequantPtx92Available &&
              (std::is_same<SourceType, Float4E2M1>::value || std::is_same<SourceType, Float6E3M2>::value ||
               std::is_same<SourceType, Float6E2M3>::value || std::is_same<SourceType, Float8E4M3>::value ||
               std::is_same<SourceType, Float8E5M2>::value);


template <class SourceType, class TargetType>
constexpr bool kUseNativeWeightDequant = kNativeDequantSupported<SourceType, TargetType>;


template <class SourceType, class TargetType>
CUDA_INLINE uint32_t dequant_native_x2(uint16_t packed) {
  static_assert(kNativeDequantSupported<SourceType, TargetType>);
  uint32_t result;

  if constexpr (std::is_same<SourceType, Float4E2M1>::value) {
    if constexpr (std::is_same<TargetType, Float16>::value) {
      asm volatile(
          "{ .reg .b8 a, b; mov.b16 {a, b}, %1; cvt.rn.f16x2.e2m1x2 %0, a; }"
          : "=r"(result) : "h"(packed));
    } else {
      asm volatile(
          "{ .reg .b8 a, b; mov.b16 {a, b}, %1; cvt.rn.bf16x2.e2m1x2 %0, a; }"
          : "=r"(result) : "h"(packed));
    }
  } else if constexpr (std::is_same<SourceType, Float6E3M2>::value) {
    if constexpr (std::is_same<TargetType, Float16>::value) {
      asm volatile("cvt.rn.f16x2.e3m2x2 %0, %1;" : "=r"(result) : "h"(packed));
    } else {
      asm volatile("cvt.rn.bf16x2.e3m2x2 %0, %1;" : "=r"(result) : "h"(packed));
    }
  } else if constexpr (std::is_same<SourceType, Float6E2M3>::value) {
    if constexpr (std::is_same<TargetType, Float16>::value) {
      asm volatile("cvt.rn.f16x2.e2m3x2 %0, %1;" : "=r"(result) : "h"(packed));
    } else {
      asm volatile("cvt.rn.bf16x2.e2m3x2 %0, %1;" : "=r"(result) : "h"(packed));
    }
  } else if constexpr (std::is_same<SourceType, Float8E4M3>::value) {
    if constexpr (std::is_same<TargetType, Float16>::value) {
      asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(result) : "h"(packed));
    } else {
      asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(result) : "h"(packed));
    }
  } else if constexpr (std::is_same<SourceType, Float8E5M2>::value) {
    if constexpr (std::is_same<TargetType, Float16>::value) {
      asm volatile("cvt.rn.f16x2.e5m2x2 %0, %1;" : "=r"(result) : "h"(packed));
    } else {
      asm volatile("cvt.rn.bf16x2.e5m2x2 %0, %1;" : "=r"(result) : "h"(packed));
    }
  }
  return result;
}


template <class TargetType>
CUDA_INLINE void dequant_e2m1_native_x8(uint32_t packed, uint32_t *dst) {
  static_assert(kNativeDequantSupported<Float4E2M1, TargetType>);
  if constexpr (std::is_same<TargetType, Float16>::value) {
    asm volatile(
        "{ .reg .b8 a, b, c, d; mov.b32 {a, b, c, d}, %4; "
        "cvt.rn.f16x2.e2m1x2 %0, a; cvt.rn.f16x2.e2m1x2 %1, b; "
        "cvt.rn.f16x2.e2m1x2 %2, c; cvt.rn.f16x2.e2m1x2 %3, d; }"
        : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3]) : "r"(packed));
  } else {
    asm volatile(
        "{ .reg .b8 a, b, c, d; mov.b32 {a, b, c, d}, %4; "
        "cvt.rn.bf16x2.e2m1x2 %0, a; cvt.rn.bf16x2.e2m1x2 %1, b; "
        "cvt.rn.bf16x2.e2m1x2 %2, c; cvt.rn.bf16x2.e2m1x2 %3, d; }"
        : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3]) : "r"(packed));
  }
}


template <class SourceType, class TargetType>
CUDA_INLINE void dequant_native_x8(uint32_t packed0, uint32_t packed1, uint32_t *dst) {
  static_assert(kNativeDequantSupported<SourceType, TargetType>);
  uint16_t packed[4];
  asm volatile(
      "mov.b32 {%0, %1}, %4; mov.b32 {%2, %3}, %5;"
      : "=h"(packed[0]), "=h"(packed[1]), "=h"(packed[2]), "=h"(packed[3])
      : "r"(packed0), "r"(packed1));

  PRAGMA_UNROLL
  for (uint32_t i = 0; i < 4; i++) {
    dst[i] = dequant_native_x2<SourceType, TargetType>(packed[i]);
  }
}


template <class SourceType, class TargetType>
CUDA_INLINE void dequant_native(const uint32_t *src, uint32_t *dst, uint32_t index) {
  uint32_t packed[2];
  if constexpr (std::is_same<SourceType, Float4E2M1>::value) {
    dequant_e2m1_native_x8<TargetType>(src[index], dst);
  } else {
    if constexpr (SourceType::kBits == 6) {
      packed[0] = get_quanted_value_group<6, false>(src, index * 2);
      packed[1] = get_quanted_value_group<6, false>(src, index * 2 + 1);
      packed[0] = (packed[0] & 0xFCFCFCFCu) >> 2;
      packed[1] = (packed[1] & 0xFCFCFCFCu) >> 2;
    } else {
      packed[0] = src[index * 2];
      packed[1] = src[index * 2 + 1];
    }
    dequant_native_x8<SourceType, TargetType>(packed[0], packed[1], dst);
  }
}
