#pragma once

#include <cuda.h>
#include "./torch_api.h"
#include "./tma.h"
#include "./utils.h"
#include <ATen/EmptyTensor.h>

inline Tensor may_make_tensor_c(std::optional<Tensor> &c, const Tensor &a, KernelData& kernel_data) {
  if (c.has_value()) return c.value();

  auto sizes = a.sym_sizes().vec();
  sizes.pop_back();
  if (kernel_data.is_moe && !kernel_data.is_moe_down) sizes.push_back(kernel_data.top_k);
  sizes.push_back(kernel_data.problem_shape_n - kernel_data.pad_shape_n);

  auto c_dtype = dtype_id_to_tensor_dtype(kernel_data.c_dtype_id);
  auto options = a.options().dtype(c_dtype);
  return at::empty_symint(sizes, options);
}

inline Tensor may_reshape_tensor_c(Tensor &c, KernelData& kernel_data) {
  if (!kernel_data.is_glu_activation) {
    return c;
  } else if (c.is_meta()) {
    auto sizes = c.sym_sizes().vec();
    sizes.pop_back();
    sizes.push_back(c.sym_size(-1) / 2);
    return at::empty_symint(sizes, c.options());
  } else {
    Tensor c_new = c.view({-1, c.size(-1) / 2});
    c_new = c_new.slice(0, 0, c_new.size(0) / 2);
    if (!kernel_data.is_moe) return c_new;
    return c_new.view({-1, kernel_data.top_k, c_new.size(-1)});
  }
}

inline void check_tensor_common(
    const Tensor &tensor, std::string name,
    int64_t expected_dev,
    ScalarType expected_dtype,
    std::optional<std::vector<int64_t>> expected_shape_ = std::nullopt) {

  ASSERT_CHECK(tensor.is_contiguous(), "error: ", name, ".is_contiguous() != true");
  ASSERT_CHECK(tensor.is_cuda(), "error: ", name, ".is_cuda() != true");
  ASSERT_CHECK(tensor.get_device() == expected_dev, "error: ", name, ".get_device() != a.get_device()");

  if (expected_shape_.has_value()) {
    auto &expected_shape = expected_shape_.value();
    if (expected_shape.size() == 1 && expected_shape[0] == 1) {
      int64_t actual = tensor.dim();
      int64_t expected = 1;
      ASSERT_CHECK(tensor.dim() == 0 || tensor.dim() == 1, name, ".dim() != expected_shape.size() => ",
                   tensor.dim(), " not in [0, 1]");
      if (tensor.dim() == 1) {
        ASSERT_CHECK(tensor.size(0) == 1, name, ".size(0) != expected_shape[0] => ",
                     tensor.size(0), " != 1");
      }
    } else {
      ASSERT_CHECK(tensor.dim() == expected_shape.size(), name, ".dim() != expected_shape.size() => ",
                   tensor.dim(), " != ", expected_shape.size());

      for (int64_t i = 0; i < tensor.dim(); i++) {
        ASSERT_CHECK(tensor.size(i) == expected_shape[i], name, ".size(", i, ") != expected_shape[", i, "] => ",
                     tensor.size(i), " != ", expected_shape[i]);
      }
    }
  }

  ASSERT_CHECK(
      tensor.scalar_type() == expected_dtype, name, ".dtype != expected_dtype => ",
      DTYPE_TO_STRING(tensor.scalar_type()), " != ", DTYPE_TO_STRING(expected_dtype));
}

inline void check_tensor_a(const Tensor &tensor, KernelData &kernel_data, int64_t dev) {
  std::vector<int64_t> expected_shape = {tensor.size(0)};
  if (kernel_data.is_moe && kernel_data.is_moe_down) expected_shape.push_back(kernel_data.top_k);

  int64_t shape_k = kernel_data.problem_shape_k - kernel_data.pad_shape_k;
  if (get_dtype_num_bits(kernel_data.a_dtype_id) == 4) {
    expected_shape.push_back(shape_k / 2);
  } else {
    expected_shape.push_back(shape_k);
  }
  auto expected_dtype = dtype_id_to_tensor_dtype(kernel_data.a_dtype_id);
  check_tensor_common(tensor, "a", dev, expected_dtype, expected_shape);
};

inline void check_tensor_b(Tensor &tensor, KernelData &kernel_data, int64_t dev) {
  uint32_t problem_shape_n = kernel_data.problem_shape_n;
  uint32_t problem_shape_k = kernel_data.problem_shape_k;
  uint32_t num_bits = get_dtype_num_bits(kernel_data.b_dtype_id);
  uint32_t pack_size_k = 256 / get_dtype_num_bits(kernel_data.a_dtype_id);
  uint32_t tensor_shape_n = problem_shape_n * pack_size_k * num_bits / 32;
  uint32_t tensor_shape_k = problem_shape_k / pack_size_k;

  std::vector<int64_t> expected_shape = {};
  if (kernel_data.is_moe) expected_shape.push_back(tensor.size(0));
  expected_shape.push_back(problem_shape_k / pack_size_k);
  expected_shape.push_back(problem_shape_n * pack_size_k * num_bits / 32);
  check_tensor_common(tensor, "b", dev, ScalarType::Int, expected_shape);
};

inline void check_tensor_c(Tensor &tensor, KernelData &kernel_data, int64_t dev, int64_t shape_m) {
  std::vector<int64_t> expected_shape = {shape_m};
  if (kernel_data.is_moe) expected_shape.push_back(kernel_data.top_k);
  expected_shape.push_back(kernel_data.problem_shape_n - kernel_data.pad_shape_n);
  auto expected_dtype = dtype_id_to_tensor_dtype(kernel_data.c_dtype_id);
  check_tensor_common(tensor, "c", dev, expected_dtype, expected_shape);
};

inline void check_tensor_as(std::optional<Tensor> &tensor, KernelData &kernel_data, int64_t dev, int64_t shape_m) {
  if (!kernel_data.has_input_scale) return;
  ASSERT_CHECK(tensor.has_value(), "as must not be none if has_input_scale");

  uint32_t problem_shape_k = kernel_data.problem_shape_k;
  uint32_t group_size = kernel_data.input_scale_group_size;
  uint32_t num_groups = group_size == 0 ? 1 : CEIL_DIV(problem_shape_k, group_size);

  std::vector<int64_t> expected_shape = {shape_m};
  if (kernel_data.is_moe && kernel_data.is_moe_down) expected_shape.push_back(kernel_data.top_k);
  expected_shape.push_back(num_groups);
  check_tensor_common(tensor.value(), "as", dev, ScalarType::Float, expected_shape);
};

inline void check_tensor_bs(std::optional<Tensor> &tensor, KernelData &kernel_data, int64_t dev, int64_t num_experts) {
  if (!kernel_data.has_weight_scale) return;
  ASSERT_CHECK(tensor.has_value(), "bs must not be none if has_weight_scale");

  uint32_t problem_shape_k = kernel_data.problem_shape_k;
  uint32_t group_size = kernel_data.weight_scale_group_size;
  uint32_t num_groups = group_size == 0 ? 1 : CEIL_DIV(problem_shape_k, group_size);

  std::vector<int64_t> expected_shape = {};
  if (kernel_data.is_moe) expected_shape.push_back(num_experts);
  expected_shape.push_back(num_groups);
  expected_shape.push_back(kernel_data.problem_shape_n);
  auto expected_dtype = dtype_id_to_tensor_dtype(kernel_data.bs_dtype_id);
  check_tensor_common(tensor.value(), "bs", dev, expected_dtype, expected_shape);
};

inline void check_tensor_bzp(std::optional<Tensor> &tensor, KernelData &kernel_data, int64_t dev, int64_t num_experts) {
  if (!kernel_data.has_dynamic_zero_point) return;
  ASSERT_CHECK(tensor.has_value(), "bzp must not be none if has_dynamic_zero_point");

  uint32_t num_bits = get_dtype_num_bits(kernel_data.b_dtype_id) <= 4 ? 4 : 8;
  uint32_t problem_shape_k = kernel_data.problem_shape_k;
  uint32_t group_size = kernel_data.weight_scale_group_size;
  uint32_t num_groups = group_size == 0 ? 1 : CEIL_DIV(problem_shape_k, group_size);

  std::vector<int64_t> expected_shape = {};
  if (kernel_data.is_moe) expected_shape.push_back(num_experts);
  expected_shape.push_back(num_groups);
  expected_shape.push_back(kernel_data.problem_shape_n * num_bits / 32);
  check_tensor_common(tensor.value(), "bzp", dev, ScalarType::Int, expected_shape);
};

inline void check_tensor_bias(std::optional<Tensor> &tensor, KernelData &kernel_data, int64_t dev, int64_t num_experts) {
  if (!kernel_data.has_bias) return;
  ASSERT_CHECK(tensor.has_value(), "bias must not be none if has_bias");
  std::vector<int64_t> expected_shape = {};
  if (kernel_data.is_moe) expected_shape.push_back(num_experts);
  expected_shape.push_back(kernel_data.problem_shape_n);
  auto expected_dtype = dtype_id_to_tensor_dtype(kernel_data.c_dtype_id);
  check_tensor_common(tensor.value(), "bias", dev, expected_dtype, expected_shape);
};

inline void check_tensor_gs(std::optional<Tensor> &tensor, KernelData &kernel_data, int64_t dev, int64_t num_experts) {
  if (!kernel_data.has_global_scale) return;
  ASSERT_CHECK(tensor.has_value(), "gs must not be none if has_global_scale");
  std::vector<int64_t> expected_shape = {kernel_data.is_moe ? num_experts : 1};
  check_tensor_common(tensor.value(), "gs", dev, ScalarType::Float, expected_shape);
};

inline void check_tensor_locks(std::optional<Tensor> &tensor, KernelData &kernel_data, int64_t dev) {
  if (!kernel_data.use_stream_k) return;
  ASSERT_CHECK(tensor.has_value(), "locks must not be none if use_stream_k");
  check_tensor_common(tensor.value(), "locks", dev, ScalarType::Int);
};

inline void check_tensor_moe(
    std::optional<Tensor> &topk_weights,
    std::optional<Tensor> &sorted_token_ids,
    std::optional<Tensor> &expert_ids,
    std::optional<Tensor> &num_tokens_post_padded,
    KernelData &kernel_data,
    int64_t dev) {

  if (!kernel_data.is_moe) return;
  ASSERT_CHECK(sorted_token_ids.has_value(), "sorted_token_ids must not be none if is_moe");
  ASSERT_CHECK(expert_ids.has_value(), "expert_ids must not be none if is_moe");
  ASSERT_CHECK(num_tokens_post_padded.has_value(), "num_tokens_post_padded must not be none if is_moe");
  check_tensor_common(sorted_token_ids.value(), "sorted_token_ids", dev, ScalarType::Int);
  check_tensor_common(expert_ids.value(), "expert_ids", dev, ScalarType::Int);
  check_tensor_common(num_tokens_post_padded.value(), "num_tokens_post_padded", dev, ScalarType::Int);

  if (!kernel_data.is_moe_down) return;
  ASSERT_CHECK(topk_weights.has_value(), "topk_weights must not be none if is_moe_down");
  check_tensor_common(topk_weights.value(), "topk_weights", dev, ScalarType::Float);
};

inline CUtensorMap make_tma_desc_a(const Tensor &tensor, KernelData &kernel_data) {
  if (!kernel_data.use_tma_a) return CUtensorMap();

  uint32_t tma_block_shape_m = kernel_data.is_moe ? 1 : kernel_data.block_shape_m;
  uint32_t tma_block_shape_k = kernel_data.block_shape_k;
  uint32_t swizzle_bytes = 128;
  uint32_t a_dtype_num_bits = get_dtype_num_bits(kernel_data.a_dtype_id);

  if (kernel_data.block_shape_k * a_dtype_num_bits == 512) {
    swizzle_bytes = 64;
  } else {
    tma_block_shape_k = 1024 / a_dtype_num_bits;
  }

  return make_tma_desc(tensor, {tma_block_shape_m, tma_block_shape_k}, swizzle_bytes);
}

inline CUtensorMap make_tma_desc_b(Tensor &tensor, KernelData &kernel_data) {
  if (!kernel_data.use_tma_b) return CUtensorMap();

  uint32_t num_bits = get_dtype_num_bits(kernel_data.b_dtype_id);
  uint32_t pack_size_k = 256 / get_dtype_num_bits(kernel_data.a_dtype_id);
  uint32_t block_shape_n = kernel_data.block_shape_n;
  uint32_t block_shape_k = kernel_data.block_shape_k;

  tensor = tensor.view({-1, tensor.size(-1)});
  tensor = tensor.view({tensor.size(0), -1, num_bits * pack_size_k});

  return make_tma_desc(tensor, {num_bits * pack_size_k, block_shape_n / 32, block_shape_k / pack_size_k});
}

inline CUtensorMap make_tma_desc_c(Tensor &tensor, KernelData &kernel_data) {
  if (!kernel_data.use_tma_c) return CUtensorMap();
  uint32_t tma_block_shape_m = kernel_data.is_moe ? 1 : kernel_data.block_shape_m;
  return make_tma_desc(tensor, {64, tma_block_shape_m});
}

inline CUtensorMap make_tma_desc_bs(std::optional<Tensor> &tensor_, KernelData &kernel_data) {
  if (!tensor_.has_value() || !kernel_data.use_tma_bs) return CUtensorMap();

  uint32_t block_shape_n = kernel_data.block_shape_n;
  uint32_t block_shape_k = kernel_data.block_shape_k;
  uint32_t group_size = kernel_data.weight_scale_group_size;
  uint32_t num_groups = group_size == 0 ? 1 : CEIL_DIV(block_shape_k, group_size);

  auto tensor = tensor_.value();
  tensor = tensor.view({-1, tensor.size(-1)});
  tensor = tensor.view({tensor.size(0), -1, 16});

  return make_tma_desc(tensor, {16, block_shape_n / 16, num_groups});
}

inline CUtensorMap make_tma_desc_bzp(std::optional<Tensor> &tensor_, KernelData &kernel_data) {
  if (!tensor_.has_value() || !kernel_data.use_tma_bzp) return CUtensorMap();

  uint32_t num_bits = get_dtype_num_bits(kernel_data.b_dtype_id) <= 4 ? 4 : 8;
  uint32_t block_shape_n = kernel_data.block_shape_n;
  uint32_t block_shape_k = kernel_data.block_shape_k;
  uint32_t group_size = kernel_data.weight_scale_group_size;
  uint32_t num_groups = group_size == 0 ? 1 : CEIL_DIV(block_shape_k, group_size);

  auto tensor = tensor_.value();

  return make_tma_desc(tensor, {16, block_shape_n * num_bits / 32, num_groups});
}

inline CUtensorMap make_tma_desc_bias(std::optional<Tensor> &tensor_, KernelData &kernel_data) {
  if (!tensor_.has_value() || !kernel_data.use_tma_bias) return CUtensorMap();

  uint32_t block_shape_n = kernel_data.block_shape_n;

  auto tensor = tensor_.value();
  tensor = tensor.view({-1, 64});

  return make_tma_desc(tensor, {64, block_shape_n / 64});
}
