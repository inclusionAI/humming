#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <humming/datatype/dtypes.cuh>
#include <humming/utils/all.cuh>


template <class ElementC>
class F16Conversion {};

template <>
class F16Conversion<Float16> {
public:
  using scalar_t = half;
  using scalar_t2 = half2;

  CUDA_INLINE
  static half2 num2num2(half x) {
    return __half2half2(x);
  };

  CUDA_INLINE
  static half2 float2num2(float x) {
    return __float2half2_rn(x);
  };

  CUDA_INLINE
  static half2 float22num2(float2 x) {
    return __float22half2_rn(x);
  };

  CUDA_INLINE
  static half2 floats2num2(float x, float y) {
    return __floats2half2_rn(x, y);
  };

  CUDA_INLINE
  static float2 num22float2(half2 x) {
    return __half22float2(x);
  };
};

template <>
class F16Conversion<BFloat16> {
public:
  using scalar_t = nv_bfloat16;
  using scalar_t2 = nv_bfloat162;

  CUDA_INLINE
  static nv_bfloat162 num2num2(nv_bfloat16 x) {
    return __bfloat162bfloat162(x);
  };

  CUDA_INLINE
  static nv_bfloat162 float2num2(float x) {
    return __float2bfloat162_rn(x);
  };

  CUDA_INLINE
  static nv_bfloat162 float22num2(float2 x) {
    return __float22bfloat162_rn(x);
  };

  CUDA_INLINE
  static nv_bfloat162 floats2num2(float x, float y) {
    return __floats2bfloat162_rn(x, y);
  };

  CUDA_INLINE
  static float2 num22float2(nv_bfloat162 x) {
    return __bfloat1622float2(x);
  };
};
