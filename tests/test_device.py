"""CPU-only tests for humming.utils.device NVML fallbacks.

These tests mock pynvml so they run without a GPU. They cover integrated GPUs
with unified memory (e.g. NVIDIA GB10 / DGX Spark, sm_121), where NVML does not
expose a memory-clock domain: nvmlDeviceGetMaxClockInfo(NVML_CLOCK_MEM) raises
NVML_ERROR_NOT_SUPPORTED and nvmlDeviceGetMemoryBusWidth returns 0. Previously
this crashed engine startup via get_default_tuning_configs -> ... ->
calculate_gpu_bandwidth.
"""

import pynvml
import pytest

from humming.utils import device


def _install_fake_nvml(
    monkeypatch,
    *,
    name="NVIDIA GB10",
    bus_width=0,
    sm_clock=3003,
    mem_clock_supported=False,
    compute_capability=(12, 1),
):
    """Patch device.pynvml with a fake NVML that mimics a given GPU."""
    monkeypatch.setattr(device.pynvml, "nvmlInit", lambda: None)
    monkeypatch.setattr(device.pynvml, "nvmlShutdown", lambda: None)
    monkeypatch.setattr(device.pynvml, "nvmlDeviceGetHandleByIndex", lambda idx: object())
    monkeypatch.setattr(device.pynvml, "nvmlDeviceGetName", lambda h: name)
    monkeypatch.setattr(device.pynvml, "nvmlDeviceGetMemoryBusWidth", lambda h: bus_width)
    monkeypatch.setattr(
        device.pynvml,
        "nvmlDeviceGetCudaComputeCapability",
        lambda h: compute_capability,
    )

    def fake_max_clock(handle, clock_type):
        if clock_type == pynvml.NVML_CLOCK_MEM:
            if not mem_clock_supported:
                raise pynvml.NVMLError(pynvml.NVML_ERROR_NOT_SUPPORTED)
            return 1500
        return sm_clock

    monkeypatch.setattr(device.pynvml, "nvmlDeviceGetMaxClockInfo", fake_max_clock)


def test_calculate_gpu_bandwidth_discrete_gpu(monkeypatch):
    # A discrete GPU that reports both bus width and memory clock.
    _install_fake_nvml(
        monkeypatch,
        name="NVIDIA H100",
        bus_width=5120,
        mem_clock_supported=True,
    )
    bw = device.calculate_gpu_bandwidth()
    # (mem_clock_mhz * 2 * bus_width) / 8 / 1000 = (1500 * 2 * 5120) / 8 / 1000
    assert bw == pytest.approx((1500 * 2 * 5120) / 8 / 1000)


def test_calculate_gpu_bandwidth_mem_clock_not_supported(monkeypatch):
    # GB10 / DGX Spark: NVML_CLOCK_MEM unsupported and bus width 0.
    _install_fake_nvml(monkeypatch, bus_width=0, mem_clock_supported=False)
    monkeypatch.delenv("HUMMING_GPU_BANDWIDTH_GBPS", raising=False)
    with pytest.warns(UserWarning, match="memory bandwidth"):
        bw = device.calculate_gpu_bandwidth()
    assert bw == device._FALLBACK_GPU_BANDWIDTH_GBPS


def test_calculate_gpu_bandwidth_zero_bus_width_supported_clock(monkeypatch):
    # Even if the memory clock query somehow succeeds, a zero bus width still
    # yields no usable bandwidth and must fall back rather than return 0.
    _install_fake_nvml(monkeypatch, bus_width=0, mem_clock_supported=True)
    monkeypatch.delenv("HUMMING_GPU_BANDWIDTH_GBPS", raising=False)
    with pytest.warns(UserWarning, match="memory bandwidth"):
        bw = device.calculate_gpu_bandwidth()
    assert bw == device._FALLBACK_GPU_BANDWIDTH_GBPS


def test_calculate_gpu_bandwidth_env_override(monkeypatch):
    _install_fake_nvml(monkeypatch, bus_width=0, mem_clock_supported=False)
    monkeypatch.setenv("HUMMING_GPU_BANDWIDTH_GBPS", "273.0")
    bw = device.calculate_gpu_bandwidth()
    assert bw == pytest.approx(273.0)


def test_estimate_tensorcore_max_tops_gb10(monkeypatch):
    # sm_121 IS in ops_map; SM clock is supported on GB10. Should not raise.
    _install_fake_nvml(monkeypatch, sm_clock=3003, compute_capability=(12, 1))
    monkeypatch.setattr(device, "get_device_num_sms", lambda idx=0: 48)
    tops = device.estimate_tensorcore_max_tops()
    # sm_count * ops_per_clock(1024) * max_clock(3003) / 1e6 * 0.9
    assert tops == pytest.approx(48 * 1024 * 3003 / 1e6 * 0.9)


def test_estimate_tensorcore_max_tops_unknown_sm(monkeypatch):
    # A future/unknown SM version must not raise KeyError.
    _install_fake_nvml(monkeypatch, sm_clock=3003, compute_capability=(13, 5))
    monkeypatch.setattr(device, "get_device_num_sms", lambda idx=0: 48)
    with pytest.warns(UserWarning, match="Unknown SM version"):
        tops = device.estimate_tensorcore_max_tops()
    assert tops == pytest.approx(48 * 1024 * 3003 / 1e6 * 0.9)
