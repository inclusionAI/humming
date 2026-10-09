#pragma once

#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "./elf.h"
#include "./mapped_file.h"
#include "./utils.h"

enum class InputQuantizationMode : uint32_t {
  Disabled = 0,
  StaticTensor = 1,
  DynamicToken = 2,
  DynamicGroup = 3,
  StaticTensorDynamicGroup = 4,
  DynamicGroupToken = 5,
};

enum class QuantizationPhase : uint32_t {
  Fused = 0,
  CollectAbsmax = 1,
  Quantize = 2,
};

inline InputQuantizationMode process_input_quantization_mode(uint32_t id) {
  ASSERT_CHECK(id <= static_cast<uint32_t>(InputQuantizationMode::DynamicGroupToken), "invalid quantization mode: ", id);
  return static_cast<InputQuantizationMode>(id);
}

inline QuantizationPhase process_input_quantization_phase(uint32_t id) {
  ASSERT_CHECK(id <= static_cast<uint32_t>(QuantizationPhase::Quantize), "invalid quantization phase: ", id);
  return static_cast<QuantizationPhase>(id);
}

struct ProcessInputKernelData {
  uint32_t num_threads;
  uint32_t source_dtype_id;
  uint32_t target_dtype_id;
  uint32_t group_scale_dtype_id;
  uint32_t input_row_size;
  uint32_t hidden_size;
  uint32_t quant_group_size;
  uint32_t columns_per_task;
  uint32_t tokens_per_block;
  uint32_t layout;
  uint32_t scatter_width;
  bool use_tile_partition;
  InputQuantizationMode quant_mode;
  QuantizationPhase quantization_phase;
  bool use_m_major_input_scale;
  uint32_t output_packing;
  uint32_t finalize_tokens;
  bool separate_outputs;
  bool use_pdl;
  bool is_finalizer;
};

struct RegisteredProcessInputKernel {
  std::shared_ptr<const MappedFile> cubin;
  std::string cubin_path;
  std::string kernel_name;
  ProcessInputKernelData metadata;
};

struct ProcessInputKernelLaunchData {
  ProcessInputKernelData metadata;
  CUfunction func;
};

struct ProcessInputShape {
  int64_t num_input_rows;
  int64_t num_output_rows;
  int64_t num_work_rows;
  int64_t num_experts;
  int64_t max_tokens_per_expert;
  int64_t group_scale_stride;
};

static std::shared_mutex g_process_input_kernel_mutex;
static std::unordered_map<std::string, std::tuple<int64_t, std::string>> g_process_input_path_ids;
static std::unordered_map<int64_t, RegisteredProcessInputKernel> g_registered_process_input_kernels;
static std::unordered_map<CUcontext, std::unordered_map<int64_t, LoadedKernel>> g_loaded_process_input_kernels;

inline ProcessInputKernelData find_process_input_kernel_data(int64_t kernel_id) {
  std::shared_lock lock(g_process_input_kernel_mutex);
  auto it = g_registered_process_input_kernels.find(kernel_id);
  ASSERT_CHECK(it != g_registered_process_input_kernels.end(), "process-input kernel not found: ", kernel_id);
  return it->second.metadata;
}

inline ProcessInputKernelLaunchData get_or_load_process_input_kernel(int64_t kernel_id, CUcontext context) {
  std::shared_ptr<const MappedFile> cubin;
  std::string kernel_name;
  ProcessInputKernelData metadata;
  {
    std::shared_lock lock(g_process_input_kernel_mutex);
    auto context_it = g_loaded_process_input_kernels.find(context);
    if (context_it != g_loaded_process_input_kernels.end()) {
      auto kernel_it = context_it->second.find(kernel_id);
      if (kernel_it != context_it->second.end()) {
        auto registered_it = g_registered_process_input_kernels.find(kernel_id);
        ASSERT_CHECK(registered_it != g_registered_process_input_kernels.end(), "process-input kernel not found: ", kernel_id);
        return {registered_it->second.metadata, kernel_it->second.func};
      }
    }

    auto registered_it = g_registered_process_input_kernels.find(kernel_id);
    ASSERT_CHECK(registered_it != g_registered_process_input_kernels.end(), "process-input kernel not found: ", kernel_id);
    cubin = registered_it->second.cubin;
    kernel_name = registered_it->second.kernel_name;
    metadata = registered_it->second.metadata;
  }

  LoadedKernel kernel = {};
  check_curesult(cuModuleLoadData(&kernel.module, cubin->data()), "cuModuleLoadData");
  check_curesult(
      cuModuleGetFunction(&kernel.func, kernel.module, kernel_name.c_str()),
      "cuModuleGetFunction");

  std::unique_lock lock(g_process_input_kernel_mutex);
  auto &context_data = g_loaded_process_input_kernels[context];
  auto [it, inserted] = context_data.emplace(kernel_id, kernel);
  if (!inserted) check_curesult(cuModuleUnload(kernel.module), "cuModuleUnload");
  return {metadata, it->second.func};
}

inline std::tuple<int64_t, std::string> register_process_input_kernel(const std::string &cubin_path) {
  using Registration = std::tuple<int64_t, std::string>;
  {
    std::shared_lock lock(g_process_input_kernel_mutex);
    auto registered = g_process_input_path_ids.find(cubin_path);
    if (registered != g_process_input_path_ids.end()) return registered->second;
  }

  auto cubin = std::make_shared<MappedFile>(cubin_path);
  CubinReader reader(cubin_path);
  std::string kernel_name;
  bool is_finalizer = false;
  for (const auto &name : reader.getKernelNames()) {
    bool process_kernel = name.find("process_input_kernel") != std::string::npos;
    bool scale_kernel = name.find("finalize_group_token_scales_kernel") != std::string::npos;
    if (!process_kernel && !scale_kernel) continue;
    ASSERT_CHECK(kernel_name.empty(), "multiple process-input kernels found in ", cubin_path);
    kernel_name = name;
    is_finalizer = scale_kernel;
  }
  ASSERT_CHECK(!kernel_name.empty(), "no process-input kernel found in ", cubin_path);

  int64_t kernel_id = get_kernel_registration_id(cubin_path, kernel_name);
  ProcessInputKernelData metadata = {
      reader.getUint32("NUM_THREADS"),
      reader.getUint32("SOURCE_DTYPE_ID"),
      reader.getUint32("TARGET_DTYPE_ID"),
      reader.getUint32("GROUP_SCALE_DTYPE_ID"),
      reader.getUint32("INPUT_ROW_SIZE"),
      reader.getUint32("HIDDEN_SIZE"),
      reader.getUint32("QUANT_GROUP_SIZE"),
      reader.getUint32("COLUMNS_PER_TASK"),
      reader.getUint32("TOKENS_PER_BLOCK"),
      reader.getUint32("LAYOUT"),
      reader.getUint32("SCATTER_WIDTH"),
      reader.getBool("USE_TILE_PARTITION"),
      process_input_quantization_mode(reader.getUint32("QUANT_MODE")),
      process_input_quantization_phase(reader.getUint32("QUANTIZATION_PHASE")),
      reader.getBool("USE_M_MAJOR_INPUT_SCALE"),
      reader.getUint32("OUTPUT_PACKING"),
      reader.getUint32("FINALIZE_TOKENS_PER_BLOCK"),
      reader.getBool("SEPARATE_OUTPUTS"),
      reader.getBool("USE_PDL"),
      is_finalizer};

  Registration result = std::make_tuple(kernel_id, kernel_name);
  std::unique_lock lock(g_process_input_kernel_mutex);
  auto path_it = g_process_input_path_ids.find(cubin_path);
  if (path_it != g_process_input_path_ids.end()) return path_it->second;
  auto kernel_it = g_registered_process_input_kernels.find(kernel_id);
  bool no_collision = kernel_it == g_registered_process_input_kernels.end() || kernel_it->second.cubin_path == cubin_path;
  ASSERT_CHECK(no_collision, "process-input kernel id collision for ", cubin_path);
  if (kernel_it == g_registered_process_input_kernels.end()) {
    auto kernel = RegisteredProcessInputKernel{cubin, cubin_path, kernel_name, metadata};
    g_registered_process_input_kernels.emplace(kernel_id, kernel);
  }
  g_process_input_path_ids[cubin_path] = result;
  return result;
}

inline void check_process_input_tensor(
    const Tensor &tensor,
    const char *name,
    int64_t device,
    ScalarType dtype,
    bool allow_byte = false,
    bool require_data_alignment = true) {
  ASSERT_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  ASSERT_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  ASSERT_CHECK(tensor.get_device() == device, name, " must be on the input device");
  if (require_data_alignment) check_tensor_data_alignment(tensor, name);
  bool valid_dtype = tensor.scalar_type() == dtype;
  valid_dtype = valid_dtype || (allow_byte && tensor.scalar_type() == ScalarType::Byte);
  ASSERT_CHECK(valid_dtype, name, " has an invalid dtype");
}

inline void check_process_input_index(const Tensor &tensor, const char *name, int64_t device) {
  ScalarType dtype = tensor.scalar_type();
  ASSERT_CHECK(dtype == ScalarType::Int || dtype == ScalarType::Long, name, " must be int32 or int64");
  check_process_input_tensor(tensor, name, device, dtype, false, false);
}

inline bool has_static_tensor_scale(InputQuantizationMode mode) {
  return mode == InputQuantizationMode::StaticTensor || mode == InputQuantizationMode::StaticTensorDynamicGroup;
}

inline bool has_dynamic_token_scale(InputQuantizationMode mode) {
  return mode == InputQuantizationMode::DynamicToken || mode == InputQuantizationMode::DynamicGroupToken;
}

inline bool has_dynamic_group_scale(InputQuantizationMode mode) {
  return mode == InputQuantizationMode::DynamicGroup ||
         mode == InputQuantizationMode::StaticTensorDynamicGroup ||
         mode == InputQuantizationMode::DynamicGroupToken;
}

inline ProcessInputShape process_input_shape(
    const ProcessInputKernelData &data,
    const Tensor &inputs,
    const Tensor &outputs,
    const std::optional<Tensor> &expert_tokens,
    const std::optional<Tensor> &scatter_idx,
    const std::optional<Tensor> &num_valid_tokens) {
  ASSERT_CHECK(inputs.dim() == 2 && inputs.size(-1) == data.input_row_size, "process_input requires 2D inputs");
  ASSERT_CHECK(inputs.numel() % data.input_row_size == 0, "invalid input size");
  int64_t num_input_rows = inputs.numel() / data.input_row_size;
  ProcessInputShape shape{num_input_rows, num_input_rows, num_input_rows, 1, 1, num_input_rows};

  if (data.layout == 0) {
    ASSERT_CHECK(!expert_tokens.has_value() && !scatter_idx.has_value(), "normal layout does not use expert_tokens or scatter_idx");
  } else if (data.layout == 3) {
    ASSERT_CHECK(expert_tokens.has_value() && expert_tokens->dim() == 1, "grouped-mask requires 1D expert_tokens");
    ASSERT_CHECK(!scatter_idx.has_value() && !num_valid_tokens.has_value(), "grouped-mask layout does not use scatter metadata");
    shape.num_experts = expert_tokens->numel();
    ASSERT_CHECK(shape.num_experts > 0 && num_input_rows % shape.num_experts == 0, "invalid grouped-mask row count");
    shape.max_tokens_per_expert = num_input_rows / shape.num_experts;
    shape.num_output_rows = shape.num_experts * shape.max_tokens_per_expert;
    shape.num_work_rows = shape.num_output_rows;
    ASSERT_CHECK(expert_tokens->dim() == 1 && expert_tokens->numel() == shape.num_experts, "invalid expert_tokens");
  } else {
    ASSERT_CHECK(data.layout == 4 && inputs.dim() == 2, "invalid scatter layout");
    ASSERT_CHECK(!expert_tokens.has_value() && scatter_idx.has_value(), "scatter requires scatter_idx and does not use expert_tokens");
    ASSERT_CHECK(scatter_idx->dim() == 2 && scatter_idx->size(0) == inputs.size(0), "invalid scatter scatter_idx");
    ASSERT_CHECK(scatter_idx->size(1) == data.scatter_width, "scatter width changed after preparation");
    ASSERT_CHECK(outputs.dim() == 2, "scatter outputs must be 2D");
    ASSERT_CHECK(outputs.size(0) >= scatter_idx->numel(), "scatter output must hold scatter_idx.numel() rows");
    shape.num_output_rows = outputs.size(0);
    shape.num_work_rows = inputs.size(0);
  }
  if (!data.use_m_major_input_scale) {
    shape.group_scale_stride = shape.num_output_rows;
  } else {
    shape.group_scale_stride = CEIL_DIV(shape.num_output_rows, 4) * 4;
  }
  return shape;
}

inline std::vector<int64_t> process_input_output_shape(
    const ProcessInputKernelData &data, const Tensor &inputs, const ProcessInputShape &shape) {
  int64_t columns = data.hidden_size / data.output_packing;
  if (data.layout == 0) {
    std::vector<int64_t> output_shape;
    output_shape.reserve(inputs.dim());
    for (int64_t dimension = 0; dimension + 1 < inputs.dim(); dimension++)
      output_shape.push_back(inputs.size(dimension));
    output_shape.push_back(columns);
    return output_shape;
  }
  return {shape.num_output_rows, columns};
}

inline void check_process_input_output(
    const ProcessInputKernelData &data,
    const Tensor &inputs,
    const ProcessInputShape &shape,
    const Tensor &outputs) {
  auto expected_shape = process_input_output_shape(data, inputs, shape);
  ScalarType dtype = dtype_id_to_tensor_dtype(data.source_dtype_id);
  if (data.quant_mode != InputQuantizationMode::Disabled)
    dtype = dtype_id_to_tensor_dtype(data.target_dtype_id);
  check_process_input_tensor(outputs, "outputs", inputs.get_device(), dtype);
  ASSERT_CHECK(outputs.dim() == static_cast<int64_t>(expected_shape.size()), "invalid output rank");
  for (size_t dimension = 0; dimension < expected_shape.size(); dimension++)
    ASSERT_CHECK(outputs.size(dimension) == expected_shape[dimension], "invalid output shape");
}

inline std::optional<Tensor> prepare_process_input_group_scales(
    const ProcessInputKernelData &data,
    const Tensor &inputs,
    const ProcessInputShape &shape,
    std::optional<Tensor> scales) {
  bool used = has_dynamic_group_scale(data.quant_mode);
  if (!used) {
    ASSERT_CHECK(!scales.has_value(), "group_scales is not used by quant_mode");
    return std::nullopt;
  }
  int64_t groups = data.hidden_size / data.quant_group_size;
  bool packed_scales = get_dtype_num_bits(data.group_scale_dtype_id) == 8;
  int64_t stored_groups = packed_scales ? CEIL_DIV(groups, 4) : groups;
  int64_t rows = shape.num_output_rows;
  int64_t columns = stored_groups;
  if (data.use_m_major_input_scale) {
    rows = stored_groups;
    columns = shape.group_scale_stride;
  }
  ScalarType dtype = packed_scales ? ScalarType::Int : dtype_id_to_tensor_dtype(data.group_scale_dtype_id);
  ASSERT_CHECK(scales.has_value(), "group_scales must be allocated by prepare_process_input");
  check_process_input_tensor(*scales, "group_scales", inputs.get_device(), dtype);
  ASSERT_CHECK(scales->dim() == 2 && scales->size(0) == rows && scales->size(1) == columns, "invalid group_scales shape");
  return scales;
}

inline std::optional<Tensor> prepare_process_input_token_scales(
    const ProcessInputKernelData &data,
    const Tensor &inputs,
    const ProcessInputShape &shape,
    std::optional<Tensor> scales) {
  bool static_scale = has_static_tensor_scale(data.quant_mode);
  bool dynamic_scale = has_dynamic_token_scale(data.quant_mode);
  if (!static_scale && !dynamic_scale) {
    ASSERT_CHECK(!scales.has_value(), "token_scales is not used by quant_mode");
    return std::nullopt;
  }
  int64_t elements = static_scale ? 1 : shape.num_output_rows;
  ASSERT_CHECK(scales.has_value(), "token_scales must be allocated by prepare_process_input");
  check_process_input_tensor(*scales, "token_scales", inputs.get_device(), ScalarType::Float);
  ASSERT_CHECK(scales->numel() == elements, "invalid token_scales size");
  if (dynamic_scale) {
    int64_t rows = data.use_m_major_input_scale ? 1 : shape.num_output_rows;
    int64_t columns = data.use_m_major_input_scale ? shape.num_output_rows : 1;
    ASSERT_CHECK(scales->dim() == 2 && scales->size(0) == rows && scales->size(1) == columns, "invalid dynamic token_scales shape");
  }
  return scales;
}

inline void launch_process_input_main(
    const ProcessInputKernelData &data,
    CUfunction func,
    const Tensor &inputs,
    const Tensor &outputs,
    const std::optional<Tensor> &group_scales,
    const std::optional<Tensor> &token_scales,
    const std::optional<Tensor> &expert_tokens,
    const std::optional<Tensor> &scatter_idx,
    const std::optional<Tensor> &num_valid_tokens,
    const ProcessInputShape &shape,
    void *output_scales) {
  const void *input_ptr = inputs.data_ptr();
  void *output_ptr = outputs.data_ptr();
  const float *static_tensor_scales = has_static_tensor_scale(data.quant_mode) ? static_cast<const float *>(token_scales->data_ptr()) : nullptr;
  float *token_scales_ptr = has_dynamic_token_scale(data.quant_mode) ? static_cast<float *>(token_scales->data_ptr()) : nullptr;
  const void *expert_tokens_ptr = expert_tokens.has_value() ? expert_tokens->data_ptr() : nullptr;
  const void *scatter_idx_ptr = scatter_idx.has_value() ? scatter_idx->data_ptr() : nullptr;
  const void *num_valid_tokens_ptr = num_valid_tokens.has_value() ? num_valid_tokens->data_ptr() : nullptr;
  uint64_t num_input_rows = static_cast<uint64_t>(shape.num_input_rows);
  uint64_t num_output_rows = static_cast<uint64_t>(shape.num_output_rows);
  uint32_t max_tokens_per_expert = static_cast<uint32_t>(shape.max_tokens_per_expert);
  uint64_t group_scale_stride = static_cast<uint64_t>(shape.group_scale_stride);
  bool use_int64_expert_tokens = expert_tokens.has_value() && expert_tokens->scalar_type() == ScalarType::Long;
  bool use_int64_scatter_idx = scatter_idx.has_value() && scatter_idx->scalar_type() == ScalarType::Long;
  bool use_int64_num_valid_tokens = num_valid_tokens.has_value() && num_valid_tokens->scalar_type() == ScalarType::Long;
  void *kernel_args[] = {
      &input_ptr,
      &output_ptr,
      &static_tensor_scales,
      &output_scales,
      &token_scales_ptr,
      &expert_tokens_ptr,
      &scatter_idx_ptr,
      &num_valid_tokens_ptr,
      &num_input_rows,
      &num_output_rows,
      &max_tokens_per_expert,
      &group_scale_stride,
      &use_int64_expert_tokens,
      &use_int64_scatter_idx,
      &use_int64_num_valid_tokens};

  int64_t work_rows = shape.num_work_rows * (data.separate_outputs ? data.scatter_width : 1);
  uint64_t grid_x;
  if (!data.use_tile_partition) {
    grid_x = CEIL_DIV(work_rows, data.tokens_per_block);
  } else {
    uint64_t blocks_per_row = CEIL_DIV(data.hidden_size, data.columns_per_task);
    grid_x = work_rows * blocks_per_row;
  }
  ASSERT_CHECK(grid_x > 0 && grid_x <= 0x7FFFFFFF, "invalid process-input grid");

  CUlaunchConfig config = {};
  config.gridDimX = static_cast<uint32_t>(grid_x);
  config.gridDimY = 1;
  config.gridDimZ = 1;
  config.blockDimX = data.num_threads;
  config.blockDimY = 1;
  config.blockDimZ = 1;
  config.hStream = get_current_cuda_stream(inputs.get_device());
  CUlaunchAttribute attribute;
  if (data.use_pdl) {
    attribute.id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
    attribute.value.programmaticStreamSerializationAllowed = 1;
    config.attrs = &attribute;
    config.numAttrs = 1;
  }
  check_curesult(cuLaunchKernelEx(&config, func, kernel_args, nullptr), "cuLaunchKernelEx");
}

inline void launch_process_input_finalizer(
    const ProcessInputKernelData &data,
    CUfunction func,
    const Tensor &intermediate,
    const Tensor &group_scales,
    const Tensor &token_scales,
    const std::optional<Tensor> &expert_tokens,
    const std::optional<Tensor> &scatter_idx,
    const std::optional<Tensor> &num_valid_tokens,
    const ProcessInputShape &shape) {
  const uint16_t *input_ptr = static_cast<const uint16_t *>(intermediate.data_ptr());
  void *output_scales = group_scales.data_ptr();
  float *token_scales_ptr = static_cast<float *>(token_scales.data_ptr());
  const void *expert_tokens_ptr = expert_tokens.has_value() ? expert_tokens->data_ptr() : nullptr;
  const void *scatter_idx_ptr = scatter_idx.has_value() ? scatter_idx->data_ptr() : nullptr;
  const void *num_valid_tokens_ptr = num_valid_tokens.has_value() ? num_valid_tokens->data_ptr() : nullptr;
  uint64_t num_input_rows = static_cast<uint64_t>(shape.num_input_rows);
  uint64_t num_output_rows = static_cast<uint64_t>(shape.num_output_rows);
  uint32_t max_tokens_per_expert = static_cast<uint32_t>(shape.max_tokens_per_expert);
  uint64_t group_scale_stride = static_cast<uint64_t>(shape.group_scale_stride);
  bool use_int64_expert_tokens = expert_tokens.has_value() && expert_tokens->scalar_type() == ScalarType::Long;
  bool use_int64_scatter_idx = scatter_idx.has_value() && scatter_idx->scalar_type() == ScalarType::Long;
  bool use_int64_num_valid_tokens = num_valid_tokens.has_value() && num_valid_tokens->scalar_type() == ScalarType::Long;
  void *kernel_args[] = {
      &input_ptr,
      &output_scales,
      &token_scales_ptr,
      &expert_tokens_ptr,
      &scatter_idx_ptr,
      &num_valid_tokens_ptr,
      &num_input_rows,
      &num_output_rows,
      &max_tokens_per_expert,
      &group_scale_stride,
      &use_int64_expert_tokens,
      &use_int64_scatter_idx,
      &use_int64_num_valid_tokens};

  int64_t work_rows = shape.num_work_rows * (data.separate_outputs ? data.scatter_width : 1);
  uint64_t grid_x = CEIL_DIV(work_rows, data.finalize_tokens);
  CUlaunchConfig config = {};
  config.gridDimX = static_cast<uint32_t>(grid_x);
  config.gridDimY = 1;
  config.gridDimZ = 1;
  config.blockDimX = data.finalize_tokens * 32;
  config.blockDimY = 1;
  config.blockDimZ = 1;
  config.hStream = get_current_cuda_stream(intermediate.get_device());
  CUlaunchAttribute attribute;
  if (data.use_pdl) {
    attribute.id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
    attribute.value.programmaticStreamSerializationAllowed = 1;
    config.attrs = &attribute;
    config.numAttrs = 1;
  }
  check_curesult(cuLaunchKernelEx(&config, func, kernel_args, nullptr), "cuLaunchKernelEx");
}

inline int64_t process_input_config_index(IntArrayRef configs, int64_t rows) {
  ASSERT_CHECK(configs.size() > 0 && configs.size() % 4 == 0, "invalid process-input configs");
  for (size_t index = 0; index < configs.size(); index += 4) {
    int64_t maximum = configs[index + 1] > 0 ? configs[index + 1] : (1LL << 60);
    if (rows > configs[index] && rows <= maximum) return static_cast<int64_t>(index);
  }
  ASSERT_CHECK(false, "no process-input kernel covers M=", rows);
}

inline void launch_process_input_impl(
    Tensor configs_tensor,
    Tensor inputs,
    Tensor outputs,
    std::optional<Tensor> group_scales,
    std::optional<Tensor> token_scales,
    std::optional<Tensor> expert_tokens,
    std::optional<Tensor> scatter_idx,
    std::optional<Tensor> num_valid_tokens) {
  DeviceContextGuard context_guard(inputs.get_device());
  ASSERT_CHECK(configs_tensor.scalar_type() == ScalarType::Long, "configs must be int64");
  ASSERT_CHECK(configs_tensor.is_contiguous() && configs_tensor.get_device() < 0, "configs must be CPU");
  IntArrayRef configs(static_cast<int64_t *>(configs_tensor.data_ptr()), configs_tensor.numel());
  ASSERT_CHECK(configs.size() >= 4 && configs.size() % 4 == 0, "invalid process-input configs");
  CUcontext context = get_current_context();
  ProcessInputKernelData base = find_process_input_kernel_data(configs[2]);
  check_process_input_tensor(inputs, "inputs", inputs.get_device(), dtype_id_to_tensor_dtype(base.source_dtype_id));
  if (inputs.data_ptr() == outputs.data_ptr()) {
    ASSERT_CHECK(base.quant_mode == InputQuantizationMode::Disabled, "inplace does not support quantization");
    ASSERT_CHECK(base.input_row_size == base.hidden_size, "inplace does not support binary activation");
    ASSERT_CHECK(base.layout == 0 || base.layout == 3, "inplace supports normal and grouped-mask layouts only");
  }
  ProcessInputShape shape = process_input_shape(base, inputs, outputs, expert_tokens, scatter_idx, num_valid_tokens);
  int64_t config_index = process_input_config_index(configs, std::max<int64_t>(shape.num_work_rows, 1));
  ProcessInputKernelLaunchData primary_kernel = get_or_load_process_input_kernel(configs[config_index + 2], context);
  ProcessInputKernelData &primary = primary_kernel.metadata;
  ASSERT_CHECK(!primary.is_finalizer, "primary process-input kernel cannot be a finalizer");
  int64_t secondary_id = configs[config_index + 3];

  if (expert_tokens.has_value())
    check_process_input_index(*expert_tokens, "expert_tokens", inputs.get_device());
  if (scatter_idx.has_value())
    check_process_input_index(*scatter_idx, "scatter_idx", inputs.get_device());
  if (num_valid_tokens.has_value()) {
    check_process_input_index(*num_valid_tokens, "num_valid_tokens", inputs.get_device());
    ASSERT_CHECK(num_valid_tokens->numel() == 1, "num_valid_tokens must contain one element");
  }
  check_process_input_output(primary, inputs, shape, outputs);
  group_scales = prepare_process_input_group_scales(primary, inputs, shape, group_scales);
  token_scales = prepare_process_input_token_scales(primary, inputs, shape, token_scales);

  if (shape.num_work_rows == 0) return;

  void *public_group_scales = group_scales.has_value() ? group_scales->data_ptr() : nullptr;
  void *public_token_scales = token_scales.has_value() ? token_scales->data_ptr() : nullptr;
  if (secondary_id < 0) {
    ASSERT_CHECK(primary.quantization_phase == QuantizationPhase::Fused, "single-stage process-input kernel must use the fused phase");
    bool token_scale = primary.quant_mode == InputQuantizationMode::DynamicToken;
    void *output_scales = token_scale ? public_token_scales : public_group_scales;
    launch_process_input_main(
        primary, primary_kernel.func, inputs, outputs, group_scales, token_scales,
        expert_tokens, scatter_idx, num_valid_tokens, shape, output_scales);
  } else {
    ProcessInputKernelLaunchData secondary_kernel = get_or_load_process_input_kernel(secondary_id, context);
    ProcessInputKernelData &secondary = secondary_kernel.metadata;
    if (primary.quant_mode == InputQuantizationMode::DynamicToken) {
      ASSERT_CHECK(primary.quantization_phase == QuantizationPhase::CollectAbsmax, "invalid dynamic-token phases");
      ASSERT_CHECK(secondary.quantization_phase == QuantizationPhase::Quantize, "invalid dynamic-token phases");
      launch_process_input_main(
          primary, primary_kernel.func, inputs, outputs, group_scales, token_scales,
          expert_tokens, scatter_idx, num_valid_tokens, shape, public_token_scales);
      launch_process_input_main(
          secondary, secondary_kernel.func, inputs, outputs, group_scales, token_scales,
          expert_tokens, scatter_idx, num_valid_tokens, shape, public_token_scales);
    } else {
      bool valid_finalizer = primary.quant_mode == InputQuantizationMode::DynamicGroupToken;
      valid_finalizer = valid_finalizer && primary.quantization_phase == QuantizationPhase::Fused;
      valid_finalizer = valid_finalizer && secondary.is_finalizer;
      valid_finalizer = valid_finalizer && secondary.quantization_phase == QuantizationPhase::Fused;
      ASSERT_CHECK(valid_finalizer, "invalid process-input secondary kernel");
      int64_t groups = primary.hidden_size / primary.quant_group_size;
      Tensor intermediate = torch_empty({shape.num_output_rows * groups * 2}, ScalarType::Byte, inputs.device());
      launch_process_input_main(
          primary, primary_kernel.func, inputs, outputs, group_scales, token_scales,
          expert_tokens, scatter_idx, num_valid_tokens, shape, intermediate.data_ptr());
      launch_process_input_finalizer(
          secondary, secondary_kernel.func, intermediate, *group_scales, *token_scales,
          expert_tokens, scatter_idx, num_valid_tokens, shape);
    }
  }
}

inline void launch_process_input(
    Tensor configs_tensor,
    Tensor inputs,
    Tensor outputs,
    std::optional<Tensor> group_scales,
    std::optional<Tensor> token_scales,
    std::optional<Tensor> expert_tokens,
    std::optional<Tensor> scatter_idx,
    std::optional<Tensor> num_valid_tokens) {
  if (inputs.is_cuda()) {
    launch_process_input_impl(configs_tensor, inputs, outputs, group_scales, token_scales, expert_tokens, scatter_idx, num_valid_tokens);
  }
}

inline void launch_process_input_inplace(
    Tensor configs_tensor,
    Tensor inputs,
    std::optional<Tensor> expert_tokens,
    std::optional<Tensor> scatter_idx,
    std::optional<Tensor> num_valid_tokens) {
  if (inputs.is_cuda()) {
    launch_process_input_impl(configs_tensor, inputs, inputs, std::nullopt, std::nullopt, expert_tokens, scatter_idx, num_valid_tokens);
  }
}
