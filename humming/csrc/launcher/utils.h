#pragma once

#include <array>
#include <cstdint>
#include <cuda.h>
#include <string>

#include "./torch_api.h"

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

enum class MmaType : uint32_t {
  MMA = 0,
  WGMMA = 1,
  UMMA = 2,
  MXMMA = 3,
};

inline void check_curesult(const CUresult res, const char *func_name) {
  if (res != CUDA_SUCCESS) {
    const char *errName;
    const char *errStr;
    cuGetErrorName(res, &errName);
    cuGetErrorString(res, &errStr);
    ASSERT_CHECK(false, func_name, " failed with error: ", errName, " (", errStr, ")");
  }
}

class DeviceContextGuard {
public:
  explicit DeviceContextGuard(int64_t dev) {
    check_curesult(cuDeviceGet(&device_, dev), "cuDeviceGet");
    CUcontext current_context;
    check_curesult(cuCtxGetCurrent(&current_context), "cuCtxGetCurrent");
    if (current_context != nullptr) {
      CUdevice current_device;
      check_curesult(cuCtxGetDevice(&current_device), "cuCtxGetDevice");
      if (current_device == device_) return;
    }
    check_curesult(cuDevicePrimaryCtxRetain(&context_, device_), "cuDevicePrimaryCtxRetain");
    check_curesult(cuCtxPushCurrent(context_), "cuCtxPushCurrent");
    active_ = true;
  }

  ~DeviceContextGuard() {
    if (!active_) return;
    CUcontext context;
    cuCtxPopCurrent(&context);
    cuDevicePrimaryCtxRelease(device_);
  }

private:
  CUdevice device_;
  CUcontext context_;
  bool active_ = false;
};

inline CUcontext get_current_context() {
  CUcontext context;
  check_curesult(cuCtxGetCurrent(&context), "cuCtxGetCurrent");
  return context;
}

inline CUstream get_current_cuda_stream(int64_t dev) {
#if USE_TORCH_STABLE_API
  void *stream_ptr = nullptr;
  aoti_torch_get_current_cuda_stream(dev, &stream_ptr);
  return static_cast<CUstream>(stream_ptr);
#else
  return at::cuda::getCurrentCUDAStream(dev);
#endif
}

uint64_t manual_crc64(const std::string &data) {
  // CRC-64/ECMA-182: non-reflected, with zero initial value and final XOR.
  static const auto table = [] {
    std::array<uint64_t, 256> values{};
    for (uint64_t i = 0; i < values.size(); i++) {
      uint64_t crc = i << 56;
      for (int j = 0; j < 8; j++) {
        if (crc & (1ULL << 63)) crc = (crc << 1) ^ 0x42F0E1EBA9EA3693ULL;
        else crc <<= 1;
      }
      values[i] = crc;
    }
    return values;
  }();

  uint64_t crc = 0;
  for (unsigned char b : data) {
    crc = table[(crc >> 56) ^ b] ^ (crc << 8);
  }
  return crc;
}

int64_t get_kernel_registration_id(const std::string &cubin_path, const std::string &kernel_name) {
  std::string key = cubin_path;
  key.push_back('\0');
  key.append(kernel_name);
  return static_cast<int64_t>(manual_crc64(key) & 0x7FFFFFFFFFFFFFFFULL);
}

uint32_t get_dtype_num_bits(uint32_t dtype_id) {
  return (dtype_id / 10000) % 100;
};

ScalarType dtype_id_to_tensor_dtype(uint32_t dtype_id) {
  switch (dtype_id) {
    case 10080000: return ScalarType::Byte;
    case 11080000: return ScalarType::Char;
    case 21080403: return ScalarType::Float8_e4m3fn;
    case 21080502: return ScalarType::Float8_e5m2;
    case 20080800: return ScalarType::Float8_e8m0fnu;
    case 21160510: return ScalarType::Half;
    case 21160807: return ScalarType::BFloat16;
    case 21320823: return ScalarType::Float;
    default: {
      uint32_t num_bits = get_dtype_num_bits(dtype_id);
      if (num_bits == 4 || num_bits == 8) return ScalarType::Byte;
      ASSERT_CHECK(false, "invalid dtype_id: ", dtype_id)
    };
  };
};

struct KernelData {
  uint32_t smem_size;
  uint32_t num_threads;
  uint32_t a_dtype_id;
  uint32_t b_dtype_id;
  uint32_t c_dtype_id;
  uint32_t bs_dtype_id;
  uint32_t problem_shape_n;
  uint32_t problem_shape_k;
  uint32_t block_shape_m;
  uint32_t block_shape_n;
  uint32_t block_shape_k;
  uint32_t warp_shape_n;
  uint32_t pad_shape_n;
  uint32_t pad_shape_k;
  uint32_t num_experts;
  uint32_t input_scale_group_size;
  uint32_t weight_scale_group_size;
  uint32_t weight_scale_group_size_n;
  uint32_t num_ctas_per_sm;
  uint32_t umma_cta_group_size;
  uint32_t output_chunk_rows;
  uint32_t output_tile_rows;
  uint32_t output_tile_columns;
  uint32_t num_stream_k_locks_per_tile;
  uint32_t multi_cast_size_a;
  uint32_t multi_cast_size_b;
  uint32_t gemm_type_id;
  MmaType mma_type;

  bool use_stream_k;
  bool is_fp_zero_point;
  bool is_channel_weight_scale;
  bool is_group_weight_scale;
  bool is_block_weight_scale;
  bool is_tensor_weight_scale;
  bool is_channel_weight_scale_2;
  bool is_tensor_weight_scale_2;
  bool has_zero_point;
  bool has_bias;
  bool has_input_scale_2;
  bool is_tensor_input_scale;
  bool is_tensor_input_scale_2;
  bool use_m_major_input_scale;
  bool use_tma_a;
  bool use_tma_as;
  bool use_tma_as2;
  bool use_tma_b;
  bool use_tma_c;
  bool use_tma_bs;
  bool use_tma_bs2;
  bool use_tma_bzp;
  bool use_tma_bias;
  bool use_pdl;
  bool use_packed_k_layout;
  bool use_umma_ss;
  bool use_raw_weight;
  bool use_block_scaled_mma;
};

struct LoadedKernel {
  CUmodule module;
  CUfunction func;
};

struct KernelLaunchData {
  KernelData metadata;
  CUfunction func;
  int64_t num_sms;
};
