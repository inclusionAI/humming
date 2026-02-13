#pragma once

#include <humming/utils/all.cuh>
#include <math.h>

#define SQRT_1_OVER_2 0.70710678118654752440f  // 1 / sqrt(2)
#define SQRT_2_OVER_PI 0.79788456080286535587f // sqrt(2 / pi)
#define GELU_COEF 0.044715f
#define QUICK_GELU_COEF 1.702f

template <ActivationType kActivationType>
CUDA_INLINE float activation_func(const float a);


template <ActivationType kActivationType>
CUDA_INLINE float activation_func(const float2 a);


template <>
CUDA_INLINE float activation_func<ActivationType::SIGMOID>(const float a) {
  return __fdividef(1.0f, 1.0f + __expf(-a));
}

template <>
CUDA_INLINE float activation_func<ActivationType::TANH>(const float a) {
  return tanhf(a);
}

template <>
CUDA_INLINE float activation_func<ActivationType::RELU>(const float a) {
  return fmaxf(0.0f, a);
}

template <>
CUDA_INLINE float activation_func<ActivationType::GELU>(const float a) {
  return 0.5f * a * (1.0f + erff(a * SQRT_1_OVER_2));
}

template <>
CUDA_INLINE float activation_func<ActivationType::FASTGELU>(const float a) {
  float x_sq = a * a;
  float poly = fmaf(GELU_COEF, x_sq, 1.0f);
  float inner = SQRT_2_OVER_PI * a * poly;
  return 0.5f * a * (1.0f + tanhf(inner));
}

template <>
CUDA_INLINE float activation_func<ActivationType::QUICKGELU>(const float a) {
  return __fdividef(a, 1.0f + __expf(-a * QUICK_GELU_COEF));
}

template <>
CUDA_INLINE float activation_func<ActivationType::SILU>(const float a) {
  return __fdividef(a, 1.0f + __expf(-a));
}

template <>
CUDA_INLINE float activation_func<ActivationType::SILU_GLU>(const float2 a) {
  return a.x * __fdividef(a.y, 1.0f + __expf(-a.y));
}
