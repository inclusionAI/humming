import os
import functools
import struct
import re
import hashlib
import importlib
import sys
from pathlib import Path
import subprocess
from elftools.elf.elffile import ELFFile


def read_symbol_value(filename, symbol_name, default_value=None):
    with open(filename, "rb") as f:
        elffile = ELFFile(f)

        symbol_table = elffile.get_section_by_name(".symtab")
        symbol = symbol_table.get_symbol_by_name(symbol_name)

        if symbol is None:
            return default_value

        symbol = symbol[0]
        section = elffile.get_section(symbol["st_shndx"])
        offset = symbol["st_value"]
        size = symbol["st_size"]

        raw_data = section.data()[offset : offset + size]
        symbol_value = struct.unpack("<i", raw_data)[0]

    return symbol_value


def find_kernel_name_in_cubin(filename, func_keyword):
    with open(filename, "rb") as f:
        elffile = ELFFile(f)
        symbol_table = elffile.get_section_by_name(".symtab")

        func_symbol_names = []
        for symbol in symbol_table.iter_symbols():
            if symbol["st_info"]["type"] == "STT_FUNC" and func_keyword in symbol.name:
                func_symbol_names.append(symbol.name)

        assert len(func_symbol_names) == 1

    return func_symbol_names[0]


def hash_to_hex(s: str) -> str:
    md5 = hashlib.md5()
    md5.update(s.encode("utf-8"))
    return md5.hexdigest()[0:16]


@functools.lru_cache(maxsize=1)
def get_cuda_include_path():
    cuda_include_path = os.getenv("CUDA_INCLUDE_PATH")
    if cuda_include_path is not None:
        return cuda_include_path
    cuda_home_path = os.getenv("CUDA_HOME")
    if cuda_home_path is not None:
        return os.path.join(cuda_home_path, "include")
    return "/usr/local/cuda/include/"


@functools.lru_cache(maxsize=8)
def get_cuda_command_path(name):
    if "/" in name:
        return name
    cuda_command_path = os.getenv(f"CUDA_{name.upper()}_PATH")
    if cuda_command_path is not None:
        return cuda_command_path
    cuda_home_path = os.getenv("CUDA_HOME")
    if cuda_home_path is not None:
        return os.path.join(cuda_home_path, "bin/" + name)

    cuda_command_path = f"/usr/local/cuda/bin/{name}"
    if os.path.exists(cuda_command_path):
        return cuda_command_path

    return name


@functools.lru_cache(maxsize=1)
def get_cuda_nvcc_version(nvcc_path):
    result = subprocess.run([nvcc_path, "--version"], stdout=subprocess.PIPE, text=True).stdout
    result = re.findall("release (\\d+\\.\\d+)", result)
    if result is None:
        raise RuntimeError(f"Invalid NVCC: {nvcc_path}")
    return result[0]


@functools.lru_cache(maxsize=1)
def get_humming_module_cache_dir():
    tmp_dir = os.getenv("HUMMING_MODULE_DIR")
    if tmp_dir is not None:
        return tmp_dir
    return os.path.join(os.path.expanduser("~"), ".humming/cache/module/")


@functools.lru_cache(maxsize=1)
def get_humming_cache_dir():
    cache_dir = os.getenv("HUMMING_CACHE_DIR")
    if cache_dir is not None:
        return cache_dir
    return os.path.join(os.path.expanduser("~"), ".humming/cache/")


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def make_humming_module(func_name, result):
    dirname = get_humming_module_cache_dir()
    if dirname not in sys.path:
        sys.path.append(dirname)

    content = f"def {func_name}():\n    return {result}\n"
    module_name = "humming_module_" + hash_to_hex(content)
    filename = module_name + ".py"
    Path(dirname).mkdir(exist_ok=True, parents=True)
    if (Path(dirname) / filename).exists():
        return importlib.import_module(module_name)

    with open(Path(dirname) / filename, "w") as f:
        f.write(content)
        f.flush()

    importlib.invalidate_caches()

    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        print(sys.path)
        raise
