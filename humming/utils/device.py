import importlib.util
import os
import subprocess
import sys
import sysconfig
from pathlib import Path
from types import ModuleType

from filelock import FileLock

import humming.utils.jit as jit_utils
from humming.utils.cuda import find_all_cuda_paths

_extension: ModuleType | None = None


def _get_cuda_include_paths() -> list[str]:
    environments = [
        environment
        for environment in find_all_cuda_paths()
        if any((Path(path) / "cuda.h").is_file() for path in environment["include_paths"])
    ]
    if not environments:
        raise RuntimeError("CUDA headers were not found; install a CUDA runtime development package")
    environment = max(
        environments,
        key=lambda value: (value["major"] or 0, value["minor"] or 0),
    )
    return environment["include_paths"]


def build_device_info_extension(
    output_path: str | Path,
    compiler: str = "g++",
    cuda_library_dir: str | Path | None = None,
) -> str:
    source_path = Path(__file__).parents[1] / "csrc" / "device_info.cpp"
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp")
    command = [
        compiler,
        "-O3",
        "-std=c++17",
        "-fPIC",
        "-shared",
        str(source_path),
        f"-I{sysconfig.get_paths()['include']}",
        *[f"-I{path}" for path in _get_cuda_include_paths()],
    ]
    if cuda_library_dir is not None:
        command.append(f"-L{cuda_library_dir}")
    command.extend(["-Wl,--no-as-needed", "-lcuda", "-Wl,--as-needed", "-o", str(temporary_path)])
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to build the device info extension:\nCMD: {' '.join(command)}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    os.replace(temporary_path, output_path)
    return output_path.as_posix()


def _load_device_info_extension(path: str | Path) -> ModuleType:
    module_name = "humming._device_info"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load device info extension from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _get_device_info_extension() -> ModuleType:
    global _extension
    if _extension is not None:
        return _extension

    source_path = Path(__file__).parents[1] / "csrc" / "device_info.cpp"
    extension_name = "_device_info.abi3.so"
    precompiled_path = jit_utils.get_precompiled_artifact_path(source_path, extension_name)
    if precompiled_path is not None:
        _extension = _load_device_info_extension(precompiled_path)
        return _extension

    signature = jit_utils.get_native_platform_signature()
    source_hash = jit_utils.hash_path_content(str(source_path), releative=True)
    build_hash = jit_utils.hash_to_hex(source_hash + signature)
    build_dir = Path(jit_utils.get_humming_cache_dir()) / "device_info" / build_hash
    extension_path = build_dir / extension_name
    build_dir.mkdir(parents=True, exist_ok=True)

    lock_path = jit_utils.get_humming_lock_filename("device_info_" + build_hash)
    with FileLock(lock_path):
        if not extension_path.is_file():
            compiler = os.environ.get("CXX") or "g++"
            build_device_info_extension(extension_path, compiler=compiler)
    _extension = _load_device_info_extension(extension_path)
    return _extension
