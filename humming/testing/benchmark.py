import enum
import importlib.metadata
import json
from pathlib import Path

from humming import ops
from humming.utils.device import calculate_gpu_bandwidth, get_device_name


def save_benchmark_result(result, args, packages: list[str] | None = None) -> None:
    kwargs = dict(vars(args))
    output_file = kwargs.pop("output_file", None)
    if output_file is None:
        return
    kwargs.pop("shape_m_list", None)
    for key, value in list(kwargs.items()):
        if isinstance(value, enum.Enum):
            kwargs[key] = value.value
    if "num_experts" in kwargs and kwargs["num_experts"] is None:
        del kwargs["num_experts"]
        del kwargs["top_k"]
        del kwargs["is_moe_down"]

    versions = {}
    packages = list(packages or [])
    packages.insert(0, "torch")
    for package in packages:
        versions[package] = importlib.metadata.version(package)

    result_new = {}
    for values in result.copy():
        values = values.copy()
        shape_m = values.pop("shape_m")
        result_new[shape_m] = values

    dtype = kwargs.get("dtype", kwargs.get("a_dtype"))
    use_f16_accum = kwargs.get("use_f16_accum", False)

    data = {
        "problem": vars(args),
        "device": {
            "device_name": get_device_name(),
            "memory_gbps": calculate_gpu_bandwidth(),
            "compute_tops": ops.tops_bench(dtype, use_f16_accum=use_f16_accum),
        },
        "packages": versions,
        "result": result_new,
    }

    Path(output_file).parent.mkdir(exist_ok=True, parents=True)
    with open(output_file, "w") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
