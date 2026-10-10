#pragma once

#include "./tma.h"
#include "./torch_api.h"
#include "./utils.h"
#include <cuda.h>

inline Tensor may_make_tensor_c(std::optional<Tensor> &c, const Tensor &a, KernelData &kernel_data, int64_t top_k) {
  if (c.has_value()) return c.value();

  int64_t shape_m = a.size(0);
  int64_t shape_n = kernel_data.problem_shape_n - kernel_data.pad_shape_n;
  if (kernel_data.gemm_type_id == 1) shape_m = shape_m * top_k;

  auto c_dtype = dtype_id_to_tensor_dtype(kernel_data.c_dtype_id);
  return torch_empty({shape_m, shape_n}, c_dtype, a.device());
}

inline Tensor make_tensor_map_buffer(const Tensor &a, KernelData &kernel_data, uint32_t num_ctas) {
  int64_t size = 0;

  uint32_t rows = kernel_data.output_chunk_rows ? std::min(kernel_data.output_chunk_rows, kernel_data.block_shape_m) : kernel_data.block_shape_m;
  bool needs_tile_boundary = kernel_data.block_shape_m % rows != 0;
  if (kernel_data.use_tma_c && (kernel_data.gemm_type_id == 2 || kernel_data.gemm_type_id == 3 || needs_tile_boundary)) {
    size = 32; // 32 int32 = 128 bytes
  }

  return torch_empty({size * num_ctas}, ScalarType::Int, a.device());
}

inline void check_tensor_common(
    const Tensor &tensor, std::string name,
    int64_t expected_dev,
    ScalarType expected_dtype,
    std::optional<std::vector<int64_t>> expected_shape_ = std::nullopt,
    bool require_data_alignment = true) {

  ASSERT_CHECK(tensor.is_contiguous(), "error: ", name, ".is_contiguous() != true");
  ASSERT_CHECK(tensor.is_cuda(), "error: ", name, ".is_cuda() != true");
  ASSERT_CHECK(tensor.get_device() == expected_dev, "error: ", name, ".get_device() != a.get_device()");
  if (require_data_alignment) check_tensor_data_alignment(tensor, name.c_str());

  if (expected_shape_.has_value()) {
    auto &expected_shape = expected_shape_.value();
    if (expected_shape.size() == 1 && expected_shape[0] == 1) {
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
  if (kernel_data.use_packed_k_layout) pack_size_k = 64;

  std::vector<int64_t> expected_shape = {};
  if (kernel_data.gemm_type_id != 0) expected_shape.push_back(kernel_data.num_experts);
  if (kernel_data.use_raw_weight) {
    if (get_dtype_num_bits(kernel_data.a_dtype_id) == 8 && num_bits < 8)
      problem_shape_k = CEIL_DIV(problem_shape_k, 128) * 128;
    expected_shape.push_back(problem_shape_n);
    expected_shape.push_back(problem_shape_k * num_bits / 32);
  } else {
    expected_shape.push_back(problem_shape_k / pack_size_k);
    expected_shape.push_back(problem_shape_n * pack_size_k * num_bits / 32);
  }
  check_tensor_common(tensor, "b", dev, ScalarType::Int, expected_shape);
};

inline void check_tensor_c(Tensor &tensor, KernelData &kernel_data, int64_t dev, int64_t shape_m, int64_t top_k) {
  std::vector<int64_t> expected_shape = {shape_m * (kernel_data.gemm_type_id == 1 ? top_k : 1)};
  expected_shape.push_back(kernel_data.problem_shape_n - kernel_data.pad_shape_n);
  auto expected_dtype = dtype_id_to_tensor_dtype(kernel_data.c_dtype_id);
  check_tensor_common(tensor, "c", dev, expected_dtype, expected_shape);
};

inline void check_tensor_as(std::optional<Tensor> &tensor, KernelData &kernel_data, int64_t dev, int64_t shape_m, int64_t top_k) {
  if (get_dtype_num_bits(kernel_data.a_dtype_id) == 16) {
    ASSERT_CHECK(!tensor.has_value(), "as must be none when input quantization is disabled");
    return;
  }
  ASSERT_CHECK(tensor.has_value(), "as must not be none for 4b or 8b activation");

  uint32_t problem_shape_k = kernel_data.problem_shape_k;
  uint32_t group_size = kernel_data.input_scale_group_size;
  uint32_t num_groups = group_size == 0 ? 1 : CEIL_DIV(problem_shape_k, group_size);
  constexpr int64_t input_scale_m_alignment = 4;
  int64_t m_pad = (shape_m + input_scale_m_alignment - 1) / input_scale_m_alignment * input_scale_m_alignment;
  if (kernel_data.use_block_scaled_mma && group_size > 0) {
    std::vector<int64_t> expected_shape;
    if (kernel_data.use_tma_as || kernel_data.use_m_major_input_scale) {
      expected_shape = {(int64_t)CEIL_DIV(num_groups, 4), m_pad};
    } else {
      expected_shape = {shape_m, (int64_t)CEIL_DIV(num_groups, 4)};
    }
    check_tensor_common(tensor.value(), "as", dev, ScalarType::Int, expected_shape);
  } else if (kernel_data.is_tensor_input_scale) {
    std::vector<int64_t> expected_shape = {1};
    check_tensor_common(tensor.value(), "as", dev, ScalarType::Float, expected_shape);
  } else {
    std::vector<int64_t> expected_shape;
    if (kernel_data.use_m_major_input_scale && group_size > 0) {
      expected_shape = {(int64_t)num_groups, m_pad};
    } else if (kernel_data.use_m_major_input_scale) {
      expected_shape = {1, shape_m};
    } else {
      expected_shape = {shape_m, (int64_t)num_groups};
    }
    check_tensor_common(tensor.value(), "as", dev, ScalarType::Float, expected_shape);
  }
};

inline void check_tensor_as2(std::optional<Tensor> &tensor, KernelData &kernel_data, int64_t dev, int64_t shape_m) {
  if (!kernel_data.has_input_scale_2) {
    ASSERT_CHECK(!tensor.has_value(), "as2 is not used by input_quant_mode");
    return;
  }
  ASSERT_CHECK(tensor.has_value(), "as2 must not be none for the configured input_quant_mode");
  std::vector<int64_t> expected_shape;
  if (kernel_data.is_tensor_input_scale_2) {
    expected_shape = {1};
  } else if (kernel_data.use_m_major_input_scale) {
    expected_shape = {1, shape_m};
  } else {
    expected_shape = {shape_m, 1};
  }
  check_tensor_common(tensor.value(), "as2", dev, ScalarType::Float, expected_shape);
};

inline void check_tensor_bs(Tensor &tensor, KernelData &kernel_data, int64_t dev) {
  if (kernel_data.is_tensor_weight_scale) {
    std::vector<int64_t> expected_shape = {kernel_data.gemm_type_id != 0 ? kernel_data.num_experts : 1};
    check_tensor_common(tensor, "bs", dev, ScalarType::Float, expected_shape);
    return;
  }
  uint32_t problem_shape_k = kernel_data.problem_shape_k;
  uint32_t problem_shape_n = kernel_data.problem_shape_n;
  uint32_t group_size = kernel_data.weight_scale_group_size;
  uint32_t group_size_n = kernel_data.weight_scale_group_size_n;

  uint32_t num_groups = group_size == 0 ? 1 : CEIL_DIV(problem_shape_k, group_size);
  uint32_t num_groups_n = group_size_n == 0 ? 1 : CEIL_DIV(problem_shape_n, group_size_n);

  std::vector<int64_t> expected_shape = {};
  auto expected_dtype = dtype_id_to_tensor_dtype(kernel_data.bs_dtype_id);
  if (kernel_data.gemm_type_id != 0) expected_shape.push_back(kernel_data.num_experts);
  if (kernel_data.use_block_scaled_mma && group_size > 0) {
    expected_dtype = ScalarType::Int;
    uint32_t num_bits = get_dtype_num_bits(kernel_data.a_dtype_id);
    uint32_t scale_vec = 256 / num_bits / group_size;
    if (kernel_data.mma_type == MmaType::UMMA) {
      expected_shape.push_back(CEIL_DIV(num_groups, 4));
      expected_shape.push_back(CEIL_DIV(problem_shape_n, 128) * 128);
    } else {
      expected_shape.push_back(num_groups / (scale_vec == 1 ? 2 : 4));
      expected_shape.push_back(problem_shape_n / (scale_vec == 1 ? 2 : 1));
    }
  } else {
    expected_shape.push_back(num_groups);
    if (kernel_data.is_block_weight_scale) {
      expected_shape.push_back(num_groups_n);
    } else {
      expected_shape.push_back(kernel_data.problem_shape_n);
    }
  }
  check_tensor_common(tensor, "bs", dev, expected_dtype, expected_shape);
};

inline void check_tensor_bzp(std::optional<Tensor> &tensor, KernelData &kernel_data, int64_t dev) {
  if (!kernel_data.has_zero_point) return;
  ASSERT_CHECK(tensor.has_value(), "bzp must not be none if has_zero_point");

  uint32_t num_bits = get_dtype_num_bits(kernel_data.b_dtype_id) <= 4 ? 4 : 8;
  uint32_t problem_shape_k = kernel_data.problem_shape_k;
  uint32_t group_size = kernel_data.weight_scale_group_size;
  uint32_t num_groups = group_size == 0 ? 1 : CEIL_DIV(problem_shape_k, group_size);

  std::vector<int64_t> expected_shape = {};
  if (kernel_data.gemm_type_id != 0) expected_shape.push_back(kernel_data.num_experts);
  ScalarType expected_dtype;
  if (kernel_data.is_fp_zero_point) {
    expected_shape.push_back(num_groups);
    expected_shape.push_back(kernel_data.problem_shape_n);
    expected_dtype = dtype_id_to_tensor_dtype(kernel_data.c_dtype_id);
  } else {
    uint32_t scale_vec = 1;
    if (kernel_data.use_block_scaled_mma && get_dtype_num_bits(kernel_data.a_dtype_id) == 4 && group_size > 0) {
      scale_vec = 256 / 4 / group_size;
    }
    expected_shape.push_back(num_groups / scale_vec);
    expected_shape.push_back(kernel_data.problem_shape_n * num_bits * scale_vec / 32);
    expected_dtype = ScalarType::Int;
  }
  check_tensor_common(tensor.value(), "bzp", dev, expected_dtype, expected_shape);
};

inline void check_tensor_bias(std::optional<Tensor> &tensor, KernelData &kernel_data, int64_t dev) {
  if (!kernel_data.has_bias) return;
  ASSERT_CHECK(tensor.has_value(), "bias must not be none if has_bias");
  std::vector<int64_t> expected_shape = {};
  if (kernel_data.gemm_type_id != 0) expected_shape.push_back(kernel_data.num_experts);
  expected_shape.push_back(kernel_data.problem_shape_n);
  auto expected_dtype = dtype_id_to_tensor_dtype(kernel_data.c_dtype_id);
  check_tensor_common(tensor.value(), "bias", dev, expected_dtype, expected_shape);
};

inline void check_tensor_bs2(std::optional<Tensor> &tensor, KernelData &kernel_data, int64_t dev) {
  if (kernel_data.is_channel_weight_scale_2) {
    ASSERT_CHECK(tensor.has_value(), "bs2 must not be none if is_channel_weight_scale_2");
    std::vector<int64_t> expected_shape = {};
    if (kernel_data.gemm_type_id != 0) expected_shape.push_back(kernel_data.num_experts);
    expected_shape.push_back(kernel_data.problem_shape_n);
    auto expected_dtype = dtype_id_to_tensor_dtype(kernel_data.c_dtype_id);
    check_tensor_common(tensor.value(), "bs2", dev, expected_dtype, expected_shape);
    return;
  } else if (kernel_data.is_tensor_weight_scale_2) {
    ASSERT_CHECK(tensor.has_value(), "bs2 must not be none if is_tensor_weight_scale_2");
    std::vector<int64_t> expected_shape = {kernel_data.gemm_type_id != 0 ? kernel_data.num_experts : 1};
    check_tensor_common(tensor.value(), "bs2", dev, ScalarType::Float, expected_shape);
  }
};

inline void check_tensor_locks(std::optional<Tensor> &tensor, KernelData &kernel_data, int64_t dev, uint32_t num_ctas) {
  if (!kernel_data.use_stream_k) return;
  ASSERT_CHECK(tensor.has_value(), "locks must not be none if use_stream_k");
  check_tensor_common(tensor.value(), "locks", dev, ScalarType::Int);
  const uint32_t cluster_size = kernel_data.multi_cast_size_a * kernel_data.multi_cast_size_b * kernel_data.umma_cta_group_size;
  const uint32_t num_cta_groups = num_ctas / cluster_size;
  const uint64_t max_stream_k_tiles = uint64_t(num_cta_groups + num_cta_groups / 10) * cluster_size;
  const uint64_t required_locks = max_stream_k_tiles * kernel_data.num_stream_k_locks_per_tile;
  ASSERT_CHECK(uint64_t(tensor->numel()) >= required_locks, "locks requires at least ", required_locks, " int32 elements, got ", tensor->numel());
};

inline void check_tensor_moe(
    std::optional<Tensor> &sorted_ids,
    std::optional<Tensor> &expert_ids,
    std::optional<Tensor> &num_tokens_padded,
    std::optional<Tensor> &expert_layout,
    KernelData &kernel_data,
    int64_t dev,
    int64_t shape_m) {

  if (kernel_data.gemm_type_id == 1) {
    ASSERT_CHECK(sorted_ids.has_value(), "sorted_ids must not be none for indexed gemm");
    ASSERT_CHECK(expert_ids.has_value(), "expert_ids must not be none for indexed gemm");
    ASSERT_CHECK(num_tokens_padded.has_value(), "num_tokens_padded must not be none for indexed gemm");
    check_tensor_common(sorted_ids.value(), "sorted_ids", dev, ScalarType::Int, std::nullopt, false);
    check_tensor_common(expert_ids.value(), "expert_ids", dev, ScalarType::Int, std::nullopt, false);
    check_tensor_common(num_tokens_padded.value(), "num_tokens_padded", dev, ScalarType::Int, std::nullopt, false);
  }
  if (kernel_data.gemm_type_id == 2) {
    ASSERT_CHECK(expert_layout.has_value(), "expert_layout must not be none for grouped gemm");
    ASSERT_CHECK(expert_layout.value().scalar_type() == ScalarType::Int || expert_layout.value().scalar_type() == ScalarType::Long,
        "expert_layout.dtype must be int32 or int64, got ", DTYPE_TO_STRING(expert_layout.value().scalar_type()));
    std::vector<int64_t> expected_shape = {kernel_data.num_experts + 1};
    check_tensor_common(expert_layout.value(), "expert_token_offset", dev, expert_layout.value().scalar_type(), expected_shape, false);
  }
  if (kernel_data.gemm_type_id == 3) {
    ASSERT_CHECK(kernel_data.num_experts > 0, "grouped masked requires at least one expert");
    ASSERT_CHECK(shape_m % kernel_data.num_experts == 0, "grouped masked input rows must divide evenly by experts");
    ASSERT_CHECK(expert_layout.has_value(), "expert_layout must not be none for grouped gemm");
    ASSERT_CHECK(expert_layout.value().scalar_type() == ScalarType::Int || expert_layout.value().scalar_type() == ScalarType::Long,
        "expert_layout.dtype must be int32 or int64, got ", DTYPE_TO_STRING(expert_layout.value().scalar_type()));
    std::vector<int64_t> expected_shape = {kernel_data.num_experts};
    check_tensor_common(expert_layout.value(), "expert_num_tokens", dev, expert_layout.value().scalar_type(), expected_shape, false);
  }
};

inline CUtensorMap make_tma_desc_a(Tensor tensor, KernelData &kernel_data) {
  if (!kernel_data.use_tma_a) return CUtensorMap();

  uint32_t tma_block_shape_m = kernel_data.block_shape_m / kernel_data.umma_cta_group_size;
  uint32_t tma_block_shape_k = kernel_data.block_shape_k;
  uint32_t swizzle_bytes = 128;
  uint32_t a_dtype_num_bits = get_dtype_num_bits(kernel_data.a_dtype_id);

  if (kernel_data.block_shape_k * a_dtype_num_bits == 512) {
    swizzle_bytes = 64;
  } else {
    tma_block_shape_k = 1024 / a_dtype_num_bits;
  }

  uint32_t tma_block_shape_k_packed =
      a_dtype_num_bits < 8 ? tma_block_shape_k * a_dtype_num_bits / 8 : tma_block_shape_k;

  tensor = torch_view_shape(tensor, {-1, tensor.size(-1)});
  if (kernel_data.mma_type == MmaType::WGMMA && a_dtype_num_bits >= 8 &&
      tensor.size(1) * a_dtype_num_bits % 1024 == 0 && kernel_data.block_shape_k * a_dtype_num_bits > 1024) {
    CUtensorMap descriptor{};
    // Split physical K into complete 128-byte slabs; the last stage may be OOB.
    uint32_t slab_elements = 1024 / a_dtype_num_bits;
    uint64_t dimensions[] = {slab_elements, uint64_t(tensor.size(0)), uint64_t(tensor.size(1)) / slab_elements};
    uint64_t strides[] = {uint64_t(tensor.size(1)) * a_dtype_num_bits / 8, 128};
    uint32_t box[] = {slab_elements, tma_block_shape_m, kernel_data.block_shape_k / slab_elements};
    uint32_t element_strides[] = {1, 1, 1};
    CUresult status = cuTensorMapEncodeTiled(&descriptor, get_tma_dtype(tensor.scalar_type()), 3,
        tensor.data_ptr(), dimensions, strides, box, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        get_swizzle_enum(128), CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    ASSERT_CHECK(status == CUDA_SUCCESS, "TMA Encode Failed for A K slabs: ", int(status));
    return descriptor;
  }
  return make_tma_desc(tensor, {tma_block_shape_k_packed, tma_block_shape_m}, swizzle_bytes, "a");
}

inline CUtensorMap make_tma_desc_as(std::optional<Tensor> &tensor_, KernelData &kernel_data) {
  if (!tensor_.has_value() || !kernel_data.use_tma_as) return CUtensorMap();

  uint32_t block_shape_m = kernel_data.block_shape_m;
  if (kernel_data.gemm_type_id == 2 || kernel_data.gemm_type_id == 3) block_shape_m += 4;
  uint32_t block_shape_k = kernel_data.block_shape_k;
  uint32_t group_size = kernel_data.input_scale_group_size;
  uint32_t num_groups = group_size == 0 ? 1 : CEIL_DIV(block_shape_k, group_size);

  auto tensor = tensor_.value();
  if (kernel_data.use_block_scaled_mma && group_size > 0) {
    tensor = torch_view_shape(tensor, {-1, tensor.size(-1)});
    return make_tma_desc(tensor, {block_shape_m, CEIL_DIV(num_groups, 4)}, 0, "as");
  }
  if (group_size == 0) {
    tensor = torch_view_shape(tensor, {-1});
    return make_tma_desc(tensor, {block_shape_m}, 0, "as");
  }

  tensor = torch_view_shape(tensor, {-1, tensor.size(-1)});
  return make_tma_desc(tensor, {block_shape_m, num_groups}, 0, "as");
}

inline CUtensorMap make_tma_desc_as2(std::optional<Tensor> &tensor_, KernelData &kernel_data) {
  if (!tensor_.has_value() || !kernel_data.use_tma_as2) return CUtensorMap();

  uint32_t block_shape_m = kernel_data.block_shape_m;
  if (kernel_data.gemm_type_id == 2 || kernel_data.gemm_type_id == 3) block_shape_m += 4;
  auto tensor = tensor_.value();
  tensor = torch_view_shape(tensor, {-1});
  return make_tma_desc(tensor, {block_shape_m}, 0, "as2");
}

inline CUtensorMap make_tma_desc_b(Tensor &tensor, KernelData &kernel_data) {
  if (!kernel_data.use_tma_b) return CUtensorMap();

  uint32_t num_bits = get_dtype_num_bits(kernel_data.b_dtype_id);
  uint32_t block_shape_n = kernel_data.block_shape_n;
  uint32_t block_shape_k = kernel_data.block_shape_k;

  if (kernel_data.use_raw_weight) {
    // FP4/FP6 in an f8f6f4 instruction uses padded 16-element blocks in SMEM.
    bool expand_low_bit = num_bits < 8 && get_dtype_num_bits(kernel_data.a_dtype_id) == 8;
    uint32_t smem_bits = expand_low_bit ? 8 : num_bits;
    uint32_t swizzle_bytes = expand_low_bit ? 128 : std::min(128u, block_shape_k * smem_bits / 8);
    if (expand_low_bit) {
      CUtensorMap descriptor{};
      uint64_t dimensions[] = {uint64_t(tensor.size(-1)) * 32 / num_bits, uint64_t(kernel_data.problem_shape_n) * std::max(1u, kernel_data.num_experts)};
      uint64_t strides[] = {uint64_t(tensor.size(-1)) * 4};
      uint32_t box[] = {swizzle_bytes, std::min(256u, kernel_data.use_umma_ss ? std::max(128u, block_shape_n) : block_shape_n)};
      uint32_t element_strides[] = {1, 1};
      CUresult status = cuTensorMapEncodeTiled(
          &descriptor, num_bits == 4 ? CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B : CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B, 2, tensor.data_ptr(),
          dimensions, strides, box, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
          get_swizzle_enum(swizzle_bytes), CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      ASSERT_CHECK(status == CUDA_SUCCESS, "TMA Encode Failed for expanded low-bit weights");
      return descriptor;
    }
    if (kernel_data.mma_type == MmaType::WGMMA) {
      CUtensorMap descriptor{};
      uint64_t w = kernel_data.warp_shape_n;
      uint64_t row_bytes = uint64_t(tensor.size(-1)) * 4;
      // Fragment planes span all N warps in the CTA, leaving the fifth
      // dimension available for K slabs even with multiple N warpgroups.
      uint64_t dimensions[] = {swizzle_bytes / 4, 16,
          uint64_t(kernel_data.problem_shape_n) * std::max(1u, kernel_data.num_experts) / w,
          w / 16, row_bytes / swizzle_bytes};
      uint64_t strides[] = {row_bytes, w * row_bytes, 16 * row_bytes, swizzle_bytes};
      uint32_t box[] = {swizzle_bytes / 4, 16, uint32_t(block_shape_n / w), uint32_t(w / 16),
          block_shape_k * num_bits / 8 / swizzle_bytes};
      uint32_t element_strides[] = {1, 1, 1, 1, 1};
      CUresult status = cuTensorMapEncodeTiled(&descriptor, CU_TENSOR_MAP_DATA_TYPE_INT32, 5,
          tensor.data_ptr(), dimensions, strides, box, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
          get_swizzle_enum(swizzle_bytes), CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      ASSERT_CHECK(status == CUDA_SUCCESS, "TMA Encode Failed for WGMMA N permutation: ", int(status));
      return descriptor;
    }
    auto rows = torch_view_shape(tensor, {-1, tensor.size(-1)});
    return make_tma_desc(rows, {swizzle_bytes / 4, std::min(256u, kernel_data.use_umma_ss ? std::max(128u, block_shape_n) : block_shape_n)}, swizzle_bytes, "b");
  }

  uint32_t pack_size_k = 256 / get_dtype_num_bits(kernel_data.a_dtype_id);
  if (kernel_data.use_packed_k_layout) pack_size_k = 64;

  tensor = torch_view_shape(tensor, {-1, tensor.size(-1)});
  tensor = torch_view_shape(tensor, {tensor.size(0), -1, pack_size_k});

  return make_tma_desc(tensor, {pack_size_k, block_shape_n * num_bits / 32, block_shape_k / pack_size_k}, 0, "b");
}

inline CUtensorMap make_tma_desc_c(Tensor tensor, KernelData &kernel_data) {
  if (!kernel_data.use_tma_c) return CUtensorMap();
  tensor = torch_view_shape(tensor, {-1, tensor.size(-1)});
  const uint32_t rows = kernel_data.output_tile_rows;
  if (tensor.size(-1) % 64 != 0) return make_tma_desc(tensor, {64, rows}, 128, "c");

  const uint32_t columns = kernel_data.output_tile_columns;
  CUtensorMap descriptor = {};
  uint64_t dimensions[] = {64, uint64_t(tensor.size(0)), uint64_t(tensor.size(1) / 64)};
  uint64_t strides[] = {uint64_t(tensor.size(1) * 2), 128};
  uint32_t box[] = {64, rows, columns / 64};
  uint32_t element_strides[] = {1, 1, 1};
  CUresult status = cuTensorMapEncodeTiled(&descriptor, get_tma_dtype(tensor.scalar_type()), 3,
      tensor.data_ptr(), dimensions, strides, box, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      get_swizzle_enum(128), CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  ASSERT_CHECK(status == CUDA_SUCCESS, "TMA Encode Failed for C N slabs: ", int(status));
  return descriptor;
}

inline CUtensorMap make_tma_desc_bs(Tensor tensor, KernelData &kernel_data) {
  if (!kernel_data.use_tma_bs) return CUtensorMap();

  uint32_t block_shape_n = kernel_data.block_shape_n;
  uint32_t block_shape_k = kernel_data.block_shape_k;
  uint32_t group_size = kernel_data.weight_scale_group_size;
  uint32_t num_groups = group_size == 0 ? 1 : CEIL_DIV(block_shape_k, group_size);

  tensor = torch_view_shape(tensor, {-1, tensor.size(-1)});

  if (kernel_data.use_block_scaled_mma && kernel_data.is_group_weight_scale) {
    uint32_t num_bits = get_dtype_num_bits(kernel_data.a_dtype_id);
    uint32_t scale_vec = 256 / num_bits / group_size;
    uint32_t packed_block_n = block_shape_n / (scale_vec == 1 ? 2 : 1);
    uint32_t packed_num_groups = num_groups / (scale_vec == 1 ? 2 : 4);
    if (kernel_data.mma_type == MmaType::UMMA) {
      packed_block_n = std::max(block_shape_n, 128u);
      packed_num_groups = CEIL_DIV(num_groups, 4);
    }
    if (packed_block_n > 256) {
      ASSERT_CHECK(packed_block_n % 256 == 0, "MXMMA BS TMA width must be divisible by 256");
      ASSERT_CHECK(tensor.size(-1) % 256 == 0, "MXMMA packed BS width must be divisible by 256");
      tensor = torch_view_shape(tensor, {-1, tensor.size(-1) / 256, 256});
      return make_tma_desc(tensor, {256, packed_block_n / 256, packed_num_groups}, 0, "bs");
    }
    return make_tma_desc(tensor, {packed_block_n, packed_num_groups}, 0, "bs");
  }

  tensor = torch_view_shape(tensor, {tensor.size(0), -1, 16});

  return make_tma_desc(tensor, {16, block_shape_n / 16, num_groups}, 0, "bs");
}

inline CUtensorMap make_tma_desc_bzp(std::optional<Tensor> &tensor_, KernelData &kernel_data) {
  if (!tensor_.has_value() || !kernel_data.use_tma_bzp) return CUtensorMap();

  uint32_t block_shape_n = kernel_data.block_shape_n;
  uint32_t block_shape_k = kernel_data.block_shape_k;
  uint32_t group_size = kernel_data.weight_scale_group_size;
  uint32_t num_groups = group_size == 0 ? 1 : CEIL_DIV(block_shape_k, group_size);

  auto tensor = tensor_.value();
  tensor = torch_view_shape(tensor, {-1, tensor.size(-1)});

  if (kernel_data.is_fp_zero_point) {
    return make_tma_desc(tensor, {block_shape_n, num_groups}, 0, "bzp");
  }

  uint32_t num_bits = get_dtype_num_bits(kernel_data.b_dtype_id) <= 4 ? 4 : 8;
  if (kernel_data.use_block_scaled_mma && get_dtype_num_bits(kernel_data.a_dtype_id) == 4 && group_size > 0) {
    uint32_t scale_vec = 256 / 4 / group_size;
    if (scale_vec > 1) return make_tma_desc(tensor, {block_shape_n * num_bits * scale_vec / 32, num_groups / scale_vec}, 0, "bzp");
  }
  return make_tma_desc(tensor, {block_shape_n * num_bits / 32, num_groups}, 0, "bzp");
}

inline CUtensorMap make_tma_desc_bias(std::optional<Tensor> &tensor_, KernelData &kernel_data) {
  if (!tensor_.has_value() || !kernel_data.use_tma_bias) return CUtensorMap();

  uint32_t block_shape_n = kernel_data.block_shape_n;

  auto tensor = tensor_.value();
  tensor = torch_view_shape(tensor, {-1, 64});

  return make_tma_desc(tensor, {64, block_shape_n / 64}, 0, "bias");
}

inline CUtensorMap make_tma_desc_bs2(std::optional<Tensor> &tensor_, KernelData &kernel_data) {
  if (!tensor_.has_value() || !kernel_data.use_tma_bs2) return CUtensorMap();

  uint32_t block_shape_n = kernel_data.block_shape_n;

  auto tensor = tensor_.value();
  tensor = torch_view_shape(tensor, {-1, 64});

  return make_tma_desc(tensor, {64, block_shape_n / 64}, 0, "bs2");
}
