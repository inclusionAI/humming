from typing import Any

import torch

from humming.utils.device import _get_device_info_extension


def get_device_index(device: int | torch.device | None = None) -> int:
    if device is None:
        return torch.cuda.current_device()
    if isinstance(device, int):
        return device
    device = torch.device(device)
    if device.index is None:
        return torch.cuda.current_device()
    return device.index


class DeviceInfo:
    __slots__ = ("_index", "_value")

    def __init__(self, index: int | torch.device | None = None) -> None:
        if isinstance(index, torch.device):
            if index.type != "cuda":
                raise ValueError(f"expected a CUDA device, got {index}")
            index = index.index if index.index is not None else torch.cuda.current_device()
        self._index = index
        self._value = None
        if index != -1:
            self._value = _get_device_info_extension()._DeviceInfo(index)
            self._index = self._value.index

    def _get_value(self) -> Any:
        if self._value is None:
            self._value = _get_device_info_extension()._DeviceInfo(self._index)
        return self._value

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get_value(), name)

    def __repr__(self) -> str:
        return f"DeviceInfo(index={self._index})"

    def print(self) -> None:
        self._get_value().print()

    @property
    def index(self) -> int:
        return self._get_value().index

    @property
    def name(self) -> str:
        return self._get_value().name

    @property
    def sm_count(self) -> int:
        return self._get_value().sm_count

    @property
    def max_threads_per_sm(self) -> int:
        return self._get_value().max_threads_per_sm

    @property
    def max_registers_per_sm(self) -> int:
        return self._get_value().max_registers_per_sm

    @property
    def sm_major(self) -> int:
        return self._get_value().sm_major

    @property
    def sm_minor(self) -> int:
        return self._get_value().sm_minor

    @property
    def sm_version(self) -> int:
        return self._get_value().sm_version

    @property
    def l2_cache_size(self) -> int:
        return self._get_value().l2_cache_size

    @property
    def l2_cache_size_mb(self) -> float:
        return self._get_value().l2_cache_size_mb

    @property
    def l1_cache_size(self) -> int:
        return self._get_value().l1_cache_size

    @property
    def l1_cache_size_kb(self) -> float:
        return self._get_value().l1_cache_size_kb

    @property
    def max_smem_size(self) -> int:
        return self._get_value().max_smem_size

    @property
    def max_smem_size_kb(self) -> float:
        return self._get_value().max_smem_size_kb

    @property
    def memory_clock_khz(self) -> int:
        return self._get_value().memory_clock_khz

    @property
    def memory_bus_width(self) -> int:
        return self._get_value().memory_bus_width

    @property
    def sm_clock_khz(self) -> int:
        return self._get_value().sm_clock_khz

    @property
    def memory_bandwidth_gbps(self) -> float:
        return self._get_value().memory_bandwidth_gbps

    @property
    def tensorcore_tops(self) -> dict[str, float]:
        return self._get_value().tensorcore_tops


current_device = DeviceInfo(-1)
