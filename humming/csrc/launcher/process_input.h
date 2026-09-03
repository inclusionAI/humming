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

enum class QuantizationMode : uint32_t {
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

inline QuantizationMode process_input_quantization_mode(uint32_t id) {
  ASSERT_CHECK(id <= static_cast<uint32_t>(QuantizationMode::DynamicGroupToken), "invalid quantization mode: ", id);
  return static_cast<QuantizationMode>(id);
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
  uint32_t tile_size;
  uint32_t tiles_per_block;
  uint32_t tokens_per_block;
  uint32_t layout;
  uint32_t layout_width;
  bool use_tile_partition;
  QuantizationMode quant_mode;
  QuantizationPhase quantization_phase;
  uint32_t scale_layout;
  uint32_t output_packing;
  uint32_t finalize_tokens;
  bool scatter_single_output;
  bool expert_layout_int64;
  bool index_int64;
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
static std::unordered_map<std::string, std::tuple<int64_t, std::string>>
    g_process_input_path_ids;
static std::unordered_map<int64_t, RegisteredProcessInputKernel>
    g_registered_process_input_kernels;
static std::unordered_map<CUcontext, std::unordered_map<int64_t, LoadedKernel>>
    g_loaded_process_input_kernels;

inline ProcessInputKernelData find_process_input_kernel_data(int64_t kernel_id) {
  std::shared_lock lock(g_process_input_kernel_mutex);
  auto it = g_registered_process_input_kernels.find(kernel_id);
  ASSERT_CHECK(it != g_registered_process_input_kernels.end(), "process-input kernel not found: ", kernel_id);
  return it->second.metadata;
}

inline ProcessInputKernelLaunchData get_or_load_process_input_kernel(
    int64_t kernel_id, CUcontext context) {
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

  int64_t kernel_id = manual_crc32(cubin_path);
  kernel_id = (kernel_id << 30) + manual_crc32(kernel_name);
  ProcessInputKernelData metadata = {
      reader.getUint32("NUM_THREADS"),
      reader.getUint32("SOURCE_DTYPE_ID"),
      reader.getUint32("TARGET_DTYPE_ID"),
      reader.getUint32("GROUP_SCALE_DTYPE_ID"),
      reader.getUint32("INPUT_ROW_SIZE"),
      reader.getUint32("HIDDEN_SIZE"),
      reader.getUint32("QUANT_GROUP_SIZE"),
      reader.getUint32("TILE_SIZE"),
      reader.getUint32("TILES_PER_BLOCK"),
      reader.getUint32("TOKENS_PER_BLOCK"),
      reader.getUint32("LAYOUT"),
      reader.getUint32("LAYOUT_WIDTH"),
      reader.getBool("USE_TILE_PARTITION"),
      process_input_quantization_mode(reader.getUint32("QUANT_MODE")),
      process_input_quantization_phase(reader.getUint32("QUANTIZATION_PHASE")),
      reader.getUint32("SCALE_LAYOUT"),
      reader.getUint32("OUTPUT_PACKING"),
      reader.getUint32("FINALIZE_TOKENS_PER_BLOCK"),
      reader.getBool("SCATTER_SINGLE_OUTPUT"),
      reader.getBool("EXPERT_LAYOUT_INT64"),
      reader.getBool("INDEX_INT64"),
      reader.getBool("USE_PDL"),
      is_finalizer};

  Registration result = std::make_tuple(kernel_id, kernel_name);
  std::unique_lock lock(g_process_input_kernel_mutex);
  auto path_it = g_process_input_path_ids.find(cubin_path);
  if (path_it != g_process_input_path_ids.end()) return path_it->second;
  auto kernel_it = g_registered_process_input_kernels.find(kernel_id);
  bool no_collision = kernel_it == g_registered_process_input_kernels.end() ||
                      kernel_it->second.cubin_path == cubin_path;
  ASSERT_CHECK(no_collision, "process-input kernel id collision for ", cubin_path);
  if (kernel_it == g_registered_process_input_kernels.end()) {
    g_registered_process_input_kernels.emplace(
        kernel_id,
        RegisteredProcessInputKernel{cubin, cubin_path, kernel_name, metadata});
  }
  g_process_input_path_ids[cubin_path] = result;
  return result;
}

inline int64_t process_input_config_index(IntArrayRef configs, int64_t rows) {
  ASSERT_CHECK(configs.size() > 0 && configs.size() % 4 == 0, "invalid process-input configs");
  for (size_t index = 0; index < configs.size(); index += 4) {
    int64_t maximum = configs[index + 1] > 0 ? configs[index + 1] : (1LL << 60);
    if (rows > configs[index] && rows <= maximum) return static_cast<int64_t>(index);
  }
  ASSERT_CHECK(false, "no process-input kernel covers M=", rows);
}

inline void check_process_input_tensor(
    const Tensor &tensor,
    const char *name,
    int64_t device,
    ScalarType dtype,
    bool allow_byte = false) {
  ASSERT_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  ASSERT_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  ASSERT_CHECK(tensor.get_device() == device, name, " must be on the input device");
  bool valid_dtype = tensor.scalar_type() == dtype;
  valid_dtype = valid_dtype || (allow_byte && tensor.scalar_type() == ScalarType::Byte);
  ASSERT_CHECK(valid_dtype, name, " has an invalid dtype");
}

inline void check_process_input_index(
    const Tensor &tensor, const char *name, int64_t device, bool int64) {
  ScalarType dtype = int64 ? ScalarType::Long : ScalarType::Int;
  check_process_input_tensor(tensor, name, device, dtype);
}

inline bool has_static_tensor_scale(QuantizationMode mode) {
  return mode == QuantizationMode::StaticTensor ||
         mode == QuantizationMode::StaticTensorDynamicGroup;
}

inline bool has_dynamic_token_scale(QuantizationMode mode) {
  return mode == QuantizationMode::DynamicToken || mode == QuantizationMode::DynamicGroupToken;
}

inline bool has_dynamic_group_scale(QuantizationMode mode) {
  return mode == QuantizationMode::DynamicGroup ||
         mode == QuantizationMode::StaticTensorDynamicGroup ||
         mode == QuantizationMode::DynamicGroupToken;
}

inline ProcessInputShape process_input_shape(
    const ProcessInputKernelData &data,
    const Tensor &inputs,
    const std::optional<Tensor> &outputs,
    const std::optional<Tensor> &expert_layout,
    const std::optional<Tensor> &indices) {
  ASSERT_CHECK(inputs.dim() >= 1 && inputs.size(-1) == data.input_row_size, "invalid input shape");
  ASSERT_CHECK(inputs.numel() % data.input_row_size == 0, "invalid input size");
  int64_t num_input_rows = inputs.numel() / data.input_row_size;
  ProcessInputShape shape{num_input_rows, num_input_rows, num_input_rows, 1, 1, num_input_rows};

  if (data.layout == 0) {
    ASSERT_CHECK(!expert_layout.has_value() && !indices.has_value(), "normal layout has no metadata");
  } else if (data.layout == 1 || data.layout == 2) {
    ASSERT_CHECK(inputs.dim() == 2 && expert_layout.has_value(), "grouped layout requires expert_layout");
    ASSERT_CHECK(expert_layout->dim() == 1 && expert_layout->numel() >= 2, "invalid expert_layout");
    shape.num_experts = expert_layout->numel() - 1;
    if (data.layout == 1) {
      ASSERT_CHECK(!indices.has_value(), "grouped layout does not use indices");
      shape.num_output_rows = inputs.size(0);
    } else {
      ASSERT_CHECK(indices.has_value() && indices->dim() == 1, "permute layout requires 1D indices");
      shape.num_output_rows = indices->numel();
    }
    shape.num_work_rows = shape.num_output_rows;
  } else if (data.layout == 3) {
    ASSERT_CHECK(inputs.dim() == 3 && expert_layout.has_value(), "grouped-padded requires expert_layout");
    ASSERT_CHECK(!indices.has_value(), "grouped-padded layout does not use indices");
    shape.num_experts = inputs.size(0);
    shape.max_tokens_per_expert = inputs.size(1);
    shape.num_output_rows = shape.num_experts * shape.max_tokens_per_expert;
    shape.num_work_rows = shape.num_output_rows;
    ASSERT_CHECK(expert_layout->dim() == 1 && expert_layout->numel() == shape.num_experts, "invalid expert_layout");
  } else {
    ASSERT_CHECK(data.layout == 4 && inputs.dim() == 2, "invalid scatter layout");
    ASSERT_CHECK(!expert_layout.has_value() && indices.has_value(), "scatter requires indices only");
    ASSERT_CHECK(indices->dim() == 2 && indices->size(0) == inputs.size(0), "invalid scatter indices");
    ASSERT_CHECK(indices->size(1) == data.layout_width, "scatter width changed after preparation");
    shape.num_output_rows = outputs.has_value() ? outputs->size(0) : indices->numel();
    shape.num_work_rows = inputs.size(0);
  }
  shape.group_scale_stride = data.scale_layout == 0 ? shape.num_output_rows : CEIL_DIV(shape.num_output_rows, 4) * 4;
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
  if (data.layout == 3)
    return {shape.num_experts, shape.max_tokens_per_expert, columns};
  return {shape.num_output_rows, columns};
}

inline Tensor prepare_process_input_output(
    const ProcessInputKernelData &data,
    const Tensor &inputs,
    const ProcessInputShape &shape,
    std::optional<Tensor> outputs,
    bool inplace) {
  auto expected_shape = process_input_output_shape(data, inputs, shape);
  ScalarType dtype = dtype_id_to_tensor_dtype(data.source_dtype_id);
  if (data.quant_mode != QuantizationMode::Disabled)
    dtype = dtype_id_to_tensor_dtype(data.target_dtype_id);
  if (inplace) {
    ASSERT_CHECK(!outputs.has_value() || outputs->data_ptr() == inputs.data_ptr(), "inplace output must alias inputs");
    outputs = inputs;
  }
  ASSERT_CHECK(outputs.has_value(), "outputs must be allocated by prepare_process_input");
  check_process_input_tensor(*outputs, "outputs", inputs.get_device(), dtype);
  ASSERT_CHECK(outputs->dim() == static_cast<int64_t>(expected_shape.size()), "invalid output rank");
  for (size_t dimension = 0; dimension < expected_shape.size(); dimension++)
    ASSERT_CHECK(outputs->size(dimension) == expected_shape[dimension], "invalid output shape");
  return *outputs;
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
  int64_t elements = shape.num_output_rows * groups;
  if (data.scale_layout == 1) elements = shape.group_scale_stride * groups;
  if (data.scale_layout == 2) elements = shape.group_scale_stride * CEIL_DIV(groups, 4) * 4;
  ScalarType dtype = dtype_id_to_tensor_dtype(data.group_scale_dtype_id);
  bool allow_byte = data.group_scale_dtype_id == 20080800;
  ASSERT_CHECK(scales.has_value(), "group_scales must be allocated by prepare_process_input");
  check_process_input_tensor(*scales, "group_scales", inputs.get_device(), dtype, allow_byte);
  ASSERT_CHECK(scales->numel() == elements, "invalid group_scales size");
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
  int64_t elements = static_scale ? shape.num_experts : shape.num_output_rows;
  ASSERT_CHECK(scales.has_value(), "token_scales must be allocated by prepare_process_input");
  check_process_input_tensor(*scales, "token_scales", inputs.get_device(), ScalarType::Float);
  ASSERT_CHECK(scales->numel() == elements, "invalid token_scales size");
  return scales;
}

inline void launch_process_input_main(
    const ProcessInputKernelData &data,
    CUfunction func,
    const Tensor &inputs,
    const Tensor &outputs,
    const std::optional<Tensor> &group_scales,
    const std::optional<Tensor> &token_scales,
    const std::optional<Tensor> &expert_layout,
    const std::optional<Tensor> &indices,
    const ProcessInputShape &shape,
    void *output_scales) {
  void *input_ptr = inputs.data_ptr();
  void *output_ptr = outputs.data_ptr();
  void *group_scale_ptr = group_scales.has_value() ? group_scales->data_ptr() : nullptr;
  void *token_scale_ptr = token_scales.has_value() ? token_scales->data_ptr() : nullptr;
  void *static_tensor_ptr = has_static_tensor_scale(data.quant_mode) ? token_scale_ptr : nullptr;
  void *dynamic_token_ptr = has_dynamic_token_scale(data.quant_mode) ? token_scale_ptr : nullptr;
  void *expert_layout_ptr = expert_layout.has_value() ? expert_layout->data_ptr() : nullptr;
  void *indices_ptr = indices.has_value() ? indices->data_ptr() : nullptr;
  uint64_t num_input_rows = static_cast<uint64_t>(shape.num_input_rows);
  uint64_t num_output_rows = static_cast<uint64_t>(shape.num_output_rows);
  uint32_t num_experts = static_cast<uint32_t>(shape.num_experts);
  uint32_t max_tokens = static_cast<uint32_t>(shape.max_tokens_per_expert);
  uint64_t scale_stride = static_cast<uint64_t>(shape.group_scale_stride);
  void *kernel_args[] = {
      &input_ptr,
      &output_ptr,
      &static_tensor_ptr,
      &output_scales,
      &dynamic_token_ptr,
      &expert_layout_ptr,
      &indices_ptr,
      &num_input_rows,
      &num_output_rows,
      &num_experts,
      &max_tokens,
      &scale_stride};

  int64_t work_rows = shape.num_work_rows * (data.scatter_single_output ? data.layout_width : 1);
  uint64_t grid_x;
  if (!data.use_tile_partition) {
    grid_x = CEIL_DIV(work_rows, data.tokens_per_block);
  } else {
    uint64_t tiles = data.hidden_size / data.tile_size;
    uint64_t blocks_per_row = CEIL_DIV(tiles, data.tiles_per_block);
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
    const std::optional<Tensor> &expert_layout,
    const std::optional<Tensor> &indices,
    const ProcessInputShape &shape) {
  void *intermediate_ptr = intermediate.data_ptr();
  void *group_scale_ptr = group_scales.data_ptr();
  void *token_scale_ptr = token_scales.data_ptr();
  void *expert_layout_ptr = expert_layout.has_value() ? expert_layout->data_ptr() : nullptr;
  void *indices_ptr = indices.has_value() ? indices->data_ptr() : nullptr;
  uint64_t num_input_rows = static_cast<uint64_t>(shape.num_input_rows);
  uint64_t num_output_rows = static_cast<uint64_t>(shape.num_output_rows);
  uint32_t num_experts = static_cast<uint32_t>(shape.num_experts);
  uint32_t max_tokens = static_cast<uint32_t>(shape.max_tokens_per_expert);
  uint64_t scale_stride = static_cast<uint64_t>(shape.group_scale_stride);
  void *kernel_args[] = {
      &intermediate_ptr,
      &group_scale_ptr,
      &token_scale_ptr,
      &expert_layout_ptr,
      &indices_ptr,
      &num_input_rows,
      &num_output_rows,
      &num_experts,
      &max_tokens,
      &scale_stride};

  int64_t work_rows = shape.num_work_rows * (data.scatter_single_output ? data.layout_width : 1);
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

inline void launch_process_input_impl(
    Tensor configs_tensor,
    Tensor inputs,
    std::optional<Tensor> outputs,
    std::optional<Tensor> group_scales,
    std::optional<Tensor> token_scales,
    std::optional<Tensor> expert_layout,
    std::optional<Tensor> indices,
    bool inplace) {
  DeviceContextGuard context_guard(inputs.get_device());
  ASSERT_CHECK(configs_tensor.scalar_type() == ScalarType::Long, "configs must be int64");
  ASSERT_CHECK(configs_tensor.is_contiguous() && configs_tensor.get_device() < 0, "configs must be CPU");
  IntArrayRef configs(static_cast<int64_t *>(configs_tensor.data_ptr()), configs_tensor.numel());
  ASSERT_CHECK(configs.size() >= 4 && configs.size() % 4 == 0, "invalid process-input configs");
  CUcontext context = get_current_context();
  ProcessInputKernelData base = find_process_input_kernel_data(configs[2]);
  check_process_input_tensor(inputs, "inputs", inputs.get_device(), dtype_id_to_tensor_dtype(base.source_dtype_id));
  ProcessInputShape shape = process_input_shape(base, inputs, outputs, expert_layout, indices);
  int64_t config_index = process_input_config_index(configs, shape.num_work_rows);
  ProcessInputKernelLaunchData primary_kernel =
      get_or_load_process_input_kernel(configs[config_index + 2], context);
  ProcessInputKernelData &primary = primary_kernel.metadata;
  ASSERT_CHECK(!primary.is_finalizer, "primary process-input kernel cannot be a finalizer");
  int64_t secondary_id = configs[config_index + 3];

  if (expert_layout.has_value())
    check_process_input_index(*expert_layout, "expert_layout", inputs.get_device(), primary.expert_layout_int64);
  if (indices.has_value())
    check_process_input_index(*indices, "indices", inputs.get_device(), primary.index_int64);
  Tensor prepared_output = prepare_process_input_output(primary, inputs, shape, outputs, inplace);
  group_scales = prepare_process_input_group_scales(primary, inputs, shape, group_scales);
  token_scales = prepare_process_input_token_scales(primary, inputs, shape, token_scales);

  void *public_group_scales = group_scales.has_value() ? group_scales->data_ptr() : nullptr;
  void *public_token_scales = token_scales.has_value() ? token_scales->data_ptr() : nullptr;
  if (secondary_id < 0) {
    ASSERT_CHECK(
        primary.quantization_phase == QuantizationPhase::Fused,
        "single-stage process-input kernel must use the fused phase");
    bool token_scale = primary.quant_mode == QuantizationMode::DynamicToken;
    void *output_scales = token_scale ? public_token_scales : public_group_scales;
    launch_process_input_main(
        primary, primary_kernel.func, inputs, prepared_output, group_scales, token_scales,
        expert_layout, indices, shape, output_scales);
  } else {
    ProcessInputKernelLaunchData secondary_kernel = get_or_load_process_input_kernel(secondary_id, context);
    ProcessInputKernelData &secondary = secondary_kernel.metadata;
    if (primary.quant_mode == QuantizationMode::DynamicToken) {
      ASSERT_CHECK(
          primary.quantization_phase == QuantizationPhase::CollectAbsmax &&
              secondary.quantization_phase == QuantizationPhase::Quantize,
          "invalid dynamic-token phases");
      launch_process_input_main(
          primary, primary_kernel.func, inputs, prepared_output, group_scales, token_scales,
          expert_layout, indices, shape, public_token_scales);
      launch_process_input_main(
          secondary, secondary_kernel.func, inputs, prepared_output, group_scales, token_scales,
          expert_layout, indices, shape, public_token_scales);
    } else {
      bool valid_finalizer = primary.quant_mode == QuantizationMode::DynamicGroupToken;
      valid_finalizer = valid_finalizer && primary.quantization_phase == QuantizationPhase::Fused;
      valid_finalizer = valid_finalizer && secondary.is_finalizer;
      valid_finalizer = valid_finalizer && secondary.quantization_phase == QuantizationPhase::Fused;
      ASSERT_CHECK(valid_finalizer, "invalid process-input secondary kernel");
      int64_t groups = primary.hidden_size / primary.quant_group_size;
      Tensor intermediate = torch_empty({shape.num_output_rows * groups * 2}, ScalarType::Byte, inputs.device());
      launch_process_input_main(
          primary, primary_kernel.func, inputs, prepared_output, group_scales, token_scales,
          expert_layout, indices, shape,
          intermediate.data_ptr());
      launch_process_input_finalizer(
          secondary, secondary_kernel.func, intermediate, *group_scales, *token_scales,
          expert_layout, indices, shape);
    }
  }
}

inline void launch_process_input(
    Tensor configs_tensor,
    Tensor inputs,
    Tensor outputs,
    std::optional<Tensor> group_scales,
    std::optional<Tensor> token_scales,
    std::optional<Tensor> expert_layout,
    std::optional<Tensor> indices) {
  if (!inputs.is_cuda()) return;
  launch_process_input_impl(
      configs_tensor, inputs, outputs, group_scales, token_scales,
      expert_layout, indices, false);
}

inline void launch_process_input_inplace(
    Tensor configs_tensor,
    Tensor inputs,
    std::optional<Tensor> expert_layout,
    std::optional<Tensor> indices) {
  if (!inputs.is_cuda()) return;
  launch_process_input_impl(
      configs_tensor, inputs, inputs, std::nullopt, std::nullopt,
      expert_layout, indices, true);
}
