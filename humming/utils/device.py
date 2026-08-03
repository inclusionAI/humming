import atexit
import functools
import threading

import pynvml
import torch

from humming.config import ComputeConfig, LayerConfig, TuningConfig
from humming.utils.smem import estimate_smem_size_config

_nvml_lock = threading.Lock()
_nvml_initialized_by_humming = False


def _ensure_nvml_initialized() -> None:
    global _nvml_initialized_by_humming

    with _nvml_lock:
        if _nvml_initialized_by_humming:
            return
        try:
            pynvml.nvmlDeviceGetCount()
        except pynvml.NVMLError_Uninitialized:
            pynvml.nvmlInit()
            _nvml_initialized_by_humming = True


def _shutdown_nvml_if_owned() -> None:
    global _nvml_initialized_by_humming

    with _nvml_lock:
        if not _nvml_initialized_by_humming:
            return
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError_Uninitialized:
            pass
        finally:
            _nvml_initialized_by_humming = False


atexit.register(_shutdown_nvml_if_owned)


def _device_index(device: int | torch.device | None = None) -> int:
    if device is None:
        return torch.cuda.current_device()
    if isinstance(device, int):
        return device
    device = torch.device(device)
    if device.index is None:
        return torch.cuda.current_device()
    return device.index


@functools.lru_cache(maxsize=None)
def _get_device_capability(device_index: int) -> tuple[int, int]:
    return torch.cuda.get_device_capability(device_index)


def get_device_capability(device: int | torch.device | None = None) -> tuple[int, int]:
    return _get_device_capability(_device_index(device))


@functools.lru_cache
def _get_device_smem_limits(device_index: int) -> tuple[int, int]:
    properties = torch.cuda.get_device_properties(device_index)
    per_block = getattr(
        properties,
        "shared_memory_per_block_optin",
        properties.shared_memory_per_block,
    )
    per_sm = properties.shared_memory_per_multiprocessor
    return per_block, per_sm


def get_device_smem_limits(device: int | torch.device | None = None) -> tuple[int, int]:
    return _get_device_smem_limits(_device_index(device))


def fits_device_smem(
    layer_config: LayerConfig,
    compute_config: ComputeConfig,
    tuning_config: TuningConfig,
    device: int | torch.device | None = None,
) -> bool:
    estimated = estimate_smem_size_config(
        layer_config,
        compute_config,
        tuning_config,
    )
    per_block, per_sm = get_device_smem_limits(device)
    return estimated <= per_block and estimated * tuning_config.num_ctas_per_sm <= per_sm


def get_device_name(gpu_index=0):
    _ensure_nvml_initialized()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    return pynvml.nvmlDeviceGetName(handle)


@functools.lru_cache(maxsize=None)
def calculate_gpu_bandwidth(gpu_index: int = 0):
    _ensure_nvml_initialized()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
    if (major, minor) == (12, 1):
        # NVIDIA GB10 (DGX Spark, sm_121): unified LPDDR5X memory with no
        # discrete memory-clock domain, so NVML_CLOCK_MEM raises
        # NVML_ERROR_NOT_SUPPORTED (and the reported bus width is 0).
        # Use the platform's known memory bandwidth instead.
        return 273.0
    gpu_name = pynvml.nvmlDeviceGetName(handle)
    try:
        bus_width = pynvml.nvmlDeviceGetMemoryBusWidth(handle)
    except pynvml.NVMLError_FunctionNotFound:
        # nvidia driver 470 + cuda-compat supports cuda 12
        # but doesn't support nvmlDeviceGetMemoryBusWidth.
        # so we hardcode bus width for some old devices.
        if "A100" in gpu_name or "A800" in gpu_name:
            bus_width = 5120
        elif "A10" in gpu_name:
            bus_width = 384
        elif "T4" in gpu_name:
            bus_width = 256
        else:
            raise
    mem_clock_mhz = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_MEM)
    return (mem_clock_mhz * 2 * bus_width) / 8 / 1000


@functools.lru_cache(maxsize=None)
def estimate_tensorcore_max_tops(gpu_index: int = 0):
    _ensure_nvml_initialized()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
    sm_version = major * 10 + minor
    max_clock_mhz = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM)
    sm_count = get_device_num_sms(gpu_index)

    ops_map = {
        75: 1024,
        80: 2048,
        86: 1024,
        87: 2048,
        89: 1024,
        90: 4096,
        100: 8192,
        103: 8192,
        120: 1024,
        121: 1024,
    }
    ops_per_clock = ops_map[sm_version]

    # 1. This function returns the dense FP16 Tensor Core performance (FP16 accumulator).
    #    Note that on certain architectures (such as SM75/SM86/SM89),
    #    the performance of the FP32 accumulator is only half that of the FP16 accumulator.
    # 2. Due to power limiting (power walls), most GPUs cannot reach the max clock speed
    #    that read by nvml. The actual achievable peak frequency must be determined
    #    through real-world benchmarking. We only estimate the value.
    factor = 0.9 if sm_version != 80 else 1.0
    return (sm_count * ops_per_clock * max_clock_mhz) / 1e6 * factor


def estimate_compute_bound_threshold(weight_nbytes, shape_n, shape_k, dtype, use_f16_accum):
    # total_memory_size = weight_nbytes + shape_k * shape_m * dtype.num_bits / 8
    # total_compute_ops = shape_n * shape_k * shape_m * 2
    # given (total_memory_size / max_bandwidth) = (total_compute_ops / max_tops), solve shape_m
    device_index = torch.cuda.current_device()
    max_bandwidth = calculate_gpu_bandwidth(device_index)
    max_tops = estimate_tensorcore_max_tops(device_index)
    num_bits = 16
    if dtype in ["float8e4m3", "float8e5m2", "int8"]:
        max_tops = max_tops * 2
        num_bits = 8
    elif dtype in ["int4", "float4e2m1"]:
        max_tops = max_tops * 4
        num_bits = 4
    sm_version_tuple = get_device_capability(device_index)
    if sm_version_tuple in [(7, 5), (8, 6), (8, 9)] and "float" in dtype and not use_f16_accum:
        max_tops = max_tops / 2

    left_bias = weight_nbytes / max_bandwidth
    left_factor = shape_k * num_bits / 8 / max_bandwidth
    right_factor = shape_n * shape_k * 2 / max_tops

    return left_bias / (right_factor - left_factor) * 1e3


@functools.lru_cache(maxsize=None)
def _get_device_num_sms(device_index: int) -> int:
    dev_props = torch.cuda.get_device_properties(device_index)
    return dev_props.multi_processor_count


def get_device_num_sms(device: int | torch.device | None = None) -> int:
    return _get_device_num_sms(_device_index(device))
