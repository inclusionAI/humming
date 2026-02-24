#define USE_CUDA 1

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <map>
#include <pybind11/pybind11.h>

#include "./elf.h"
#include "./tensor.h"
#include "./torch_api.h"
#include "./utils.h"

static std::unordered_map<int64_t, KernelData> g_kernel_data;

inline int64_t find_kernel_id(IntArrayRef &configs, int64_t shape_m) {
  size_t n = configs.size();
  if (n == 1) return configs[0];
  if (n > 0 && n % 3 == 0) {
    for (size_t i = 0; i < n; i += 3) {
      int64_t min_val = configs[i];
      int64_t max_val = configs[i + 1];
      max_val = max_val > 0 ? max_val : (1 << 30);
      int64_t target = configs[i + 2];
      if (shape_m >= min_val && shape_m <= max_val) return target;
    }

    STD_TORCH_CHECK(false, "shape_m is not within any range defined in configs.");
  }

  STD_TORCH_CHECK(false, "configs length must be 1 or a non-zero multiple of 3.");
};

inline KernelData find_kernel(IntArrayRef &configs, int64_t shape_m) {
  int64_t kernel_id = find_kernel_id(configs, shape_m);
  STD_TORCH_CHECK(g_kernel_data.find(kernel_id) != g_kernel_data.end(), "kernel not existed.");
  KernelData &kernel_data = g_kernel_data[kernel_id];
  return kernel_data;
};

inline cudaStream_t get_current_cuda_stream(int64_t dev) {
#ifdef USE_TORCH_STABLE_API
  void *stream_ptr = nullptr;
  aoti_torch_get_current_cuda_stream(dev, &stream_ptr);
  return static_cast<cudaStream_t>(stream_ptr);
#else
  return at::cuda::getCurrentCUDAStream(dev);

#endif
}

inline int64_t get_num_sms(std::optional<int64_t> num_sms, int64_t dev) {
  if (num_sms.has_value()) {
    ASSERT_CHECK(num_sms.value() > 0, "error: num_sms < 1 => ", num_sms.value(), " < 1");
    return num_sms.value();
  }
  int32_t dev_sms;
  cudaDeviceGetAttribute(&dev_sms, cudaDevAttrMultiProcessorCount, dev);
  return static_cast<int64_t>(dev_sms);
}

Tensor launch_humming(
    IntArrayRef configs,
    Tensor a,
    Tensor b,
    std::optional<Tensor> c_,
    std::optional<Tensor> as_,
    std::optional<Tensor> bs_,
    std::optional<Tensor> bzp_,
    std::optional<Tensor> bias_,
    std::optional<Tensor> gs_,
    std::optional<Tensor> topk_weights_,
    std::optional<Tensor> sorted_token_ids_,
    std::optional<Tensor> expert_ids_,
    std::optional<Tensor> num_tokens_post_padded_,
    std::optional<Tensor> locks_,
    int64_t num_ctas_per_sm = 1,
    std::optional<int64_t> num_sms = std::nullopt,
    bool should_check_tensor = true) {
  
  if (a.is_meta()) {
    KernelData kernel_data = find_kernel(configs, 1);
    Tensor c = may_make_tensor_c(c_, a, kernel_data);
    return may_reshape_tensor_c(c, kernel_data);
  }

  int64_t dev = a.get_device();
  int64_t shape_m = a.size(0);
  int64_t num_experts = b.dim() == 3 ? b.size(0) : 0;
  KernelData kernel_data = find_kernel(configs, shape_m);
  int64_t actual_shape_m = shape_m * (kernel_data.is_moe_down ? kernel_data.top_k : 1);
  Tensor c = may_make_tensor_c(c_, a, kernel_data);
  a = a.contiguous();

  if (should_check_tensor) {
    check_tensor_a(a, kernel_data, dev);
    check_tensor_b(b, kernel_data, dev);
    check_tensor_c(c, kernel_data, dev, shape_m);
    check_tensor_as(as_, kernel_data, dev, shape_m);
    check_tensor_bs(bs_, kernel_data, dev, num_experts);
    check_tensor_bzp(bzp_, kernel_data, dev, num_experts);
    check_tensor_bias(bias_, kernel_data, dev, num_experts);
    check_tensor_gs(gs_, kernel_data, dev, num_experts);
    check_tensor_locks(locks_, kernel_data, dev);
    check_tensor_moe(topk_weights_, sorted_token_ids_, expert_ids_, num_tokens_post_padded_, kernel_data, dev);
  }

  void *a_ptr = a.data_ptr();
  void *b_ptr = b.data_ptr();
  void *c_ptr = c.data_ptr();
  void *as_ptr = as_.has_value() ? as_->data_ptr() : nullptr;
  void *bs_ptr = bs_.has_value() ? bs_->data_ptr() : nullptr;
  void *bzp_ptr = bzp_.has_value() ? bzp_->data_ptr() : nullptr;
  void *bias_ptr = bias_.has_value() ? bias_->data_ptr() : nullptr;
  void *gs_ptr = gs_.has_value() ? gs_->data_ptr() : nullptr;
  void *topk_weights_ptr = topk_weights_.has_value() ? topk_weights_->data_ptr() : nullptr;
  void *sorted_token_ids_ptr = sorted_token_ids_.has_value() ? sorted_token_ids_->data_ptr() : nullptr;
  void *expert_ids_ptr = expert_ids_.has_value() ? expert_ids_->data_ptr() : nullptr;
  void *num_tokens_post_padded_ptr = num_tokens_post_padded_.has_value() ? num_tokens_post_padded_->data_ptr() : nullptr;
  void *locks_ptr = locks_.has_value() ? locks_->data_ptr() : nullptr;

  auto tensor_map_a = make_tma_desc_a(a, kernel_data);
  auto tensor_map_b = make_tma_desc_b(b, kernel_data);
  auto tensor_map_c = make_tma_desc_c(c, kernel_data);
  auto tensor_map_bs = make_tma_desc_bs(bs_, kernel_data);
  auto tensor_map_bzp = make_tma_desc_bzp(bzp_, kernel_data);
  auto tensor_map_bias = make_tma_desc_bias(bias_, kernel_data);
  auto to_void_ptr = [&](void *ptr) { return ptr; };

  void *kernel_args[] = {
      kernel_data.use_tma_a ? to_void_ptr(&tensor_map_a) : to_void_ptr(&a_ptr),
      kernel_data.use_tma_b ? to_void_ptr(&tensor_map_b) : to_void_ptr(&b_ptr),
      kernel_data.use_tma_c ? to_void_ptr(&tensor_map_c) : to_void_ptr(&c_ptr),
      &as_ptr,
      kernel_data.use_tma_bs ? to_void_ptr(&tensor_map_bs) : to_void_ptr(&bs_ptr),
      kernel_data.use_tma_bzp ? to_void_ptr(&tensor_map_bzp) : to_void_ptr(&bzp_ptr),
      kernel_data.use_tma_bias ? to_void_ptr(&tensor_map_bias) : to_void_ptr(&bias_ptr),
      &gs_ptr,
      &topk_weights_ptr,
      &sorted_token_ids_ptr,
      &expert_ids_ptr,
      &num_tokens_post_padded_ptr,
      &locks_ptr,
      &actual_shape_m};

  CUlaunchConfig config = {};
  config.gridDimX = num_ctas_per_sm * get_num_sms(num_sms, dev);
  config.gridDimY = 1;
  config.gridDimZ = 1;
  config.blockDimX = kernel_data.num_threads;
  config.blockDimY = 1;
  config.blockDimZ = 1;
  config.sharedMemBytes = kernel_data.smem_size;
  config.hStream = get_current_cuda_stream(dev);

  CUfunction &func = kernel_data.func;
  constexpr auto SMEM_SIZE_ATTR = CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES;
  check_curesult(cuFuncSetAttribute(func, SMEM_SIZE_ATTR, kernel_data.smem_size), "cuFuncSetAttribute");
  check_curesult(cuLaunchKernelEx(&config, func, kernel_args, nullptr), "cuLaunchKernelEx");

  return may_reshape_tensor_c(c, kernel_data);
};

int64_t register_kernel(const std::string &cubin_path, const std::string &func_name) {
  std::string hash_str = cubin_path + "\n" + func_name;
  int64_t hash_id = manual_crc32(cubin_path);
  hash_id = (hash_id << 30) + manual_crc32(func_name);

  KernelData kernel_data;
  if (g_kernel_data.find(hash_id) != g_kernel_data.end()) {
    kernel_data = g_kernel_data[hash_id];
  } else {
    CUmodule hModule;
    CUfunction hKernel;
    check_curesult(cuModuleLoad(&hModule, cubin_path.c_str()), "cuModuleLoad");
    check_curesult(cuModuleGetFunction(&hKernel, hModule, func_name.c_str()), "cuModuleGetFunction");
    auto reader = CubinReader(cubin_path);

    g_kernel_data[hash_id] = {
        hModule,
        hKernel,

        reader.getUint32("SMEM_SIZE"),
        reader.getUint32("NUM_THREADS"),
        reader.getUint32("A_DTYPE_ID"),
        reader.getUint32("B_DTYPE_ID"),
        reader.getUint32("C_DTYPE_ID"),
        reader.getUint32("BS_DTYPE_ID"),
        reader.getUint32("PROBLEM_SHAPE_N"),
        reader.getUint32("PROBLEM_SHAPE_K"),
        reader.getUint32("BLOCK_SHAPE_M"),
        reader.getUint32("BLOCK_SHAPE_N"),
        reader.getUint32("BLOCK_SHAPE_K"),
        reader.getUint32("PAD_SHAPE_N"),
        reader.getUint32("PAD_SHAPE_K"),
        reader.getUint32("INPUT_SCALE_GROUP_SIZE"),
        reader.getUint32("WEIGHT_SCALE_GROUP_SIZE"),
        reader.getUint32("TOP_K"),

        reader.getBool("USE_STREAM_K"),
        reader.getBool("IS_MOE"),
        reader.getBool("IS_MOE_DOWN"),
        reader.getBool("IS_GLU_ACTIVATION"),
        reader.getBool("HAS_INPUT_SCALE"),
        reader.getBool("HAS_WEIGHT_SCALE"),
        reader.getBool("HAS_DYNAMIC_ZERO_POINT"),
        reader.getBool("HAS_BIAS"),
        reader.getBool("HAS_GLOBAL_SCALE"),
        reader.getBool("USE_TMA_A"),
        reader.getBool("USE_TMA_B"),
        reader.getBool("USE_TMA_C"),
        reader.getBool("USE_TMA_BS"),
        reader.getBool("USE_TMA_BZP"),
        reader.getBool("USE_TMA_BIAS")};
  };

  return hash_id;
}

COMMON_TORCH_LIBRARY(humming, m) {
  m.def(
      "launch_humming(int[] configs, Tensor a, Tensor b, Tensor(c!)? c, "
      "Tensor? as_, Tensor? bs, Tensor? bzp, Tensor? bias, Tensor? gs, "
      "Tensor? topk_weights, Tensor? sorted_token_ids, Tensor? expert_ids, Tensor? num_tokens_post_padded, "
      "Tensor(locks!)? locks, int num_ctas_per_sm = 1, int? num_sms = None, bool should_check_tensor = True) -> Tensor");
  m.def("register_kernel(str cubin_path, str func_name) -> int");
};

COMMON_TORCH_LIBRARY_IMPL(humming, CUDA, m) {
  m.impl("launch_humming", COMMON_TORCH_BOX(&launch_humming));
};

COMMON_TORCH_LIBRARY_IMPL(humming, Meta, m) {
  m.impl("launch_humming", COMMON_TORCH_BOX(&launch_humming));
};

COMMON_TORCH_LIBRARY_IMPL(humming, Undefined, m) {
  m.impl("register_kernel", COMMON_TORCH_BOX(&register_kernel));
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){};
