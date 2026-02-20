#pragma once

#ifdef USE_TORCH_STABLE_API

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor_inl.h>

using Tensor = torch::stable::Tensor;
using IntArrayRef = torch::headeronly::IntHeaderOnlyArrayRef;
using ScalarType = torch::headeronly::ScalarType;

#define ASSERT_CHECK STD_TORCH_CHECK
#define COMMON_TORCH_LIBRARY STABLE_TORCH_LIBRARY
#define COMMON_TORCH_LIBRARY_IMPL STABLE_TORCH_LIBRARY_IMPL
#define COMMON_TORCH_BOX TORCH_BOX
#define DTYPE_TO_STRING torch::headeronly::toString

#else

#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>
#include <ATen/ops/empty.h>

using Tensor = at::Tensor;
using IntArrayRef = at::IntArrayRef;
using ScalarType = at::ScalarType;

#define ASSERT_CHECK TORCH_CHECK
#define COMMON_TORCH_LIBRARY TORCH_LIBRARY
#define COMMON_TORCH_LIBRARY_IMPL TORCH_LIBRARY_IMPL
#define COMMON_TORCH_BOX
#define DTYPE_TO_STRING at::toString

#endif
