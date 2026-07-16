import functools
import glob
import hashlib
import importlib
import os
import re
import struct
import subprocess
import sys
import threading
from pathlib import Path

from elftools.elf.elffile import ELFFile
from filelock import FileLock

import humming.utils.jit as jit_utils


def popen_and_reap(cmd, **kwargs):
    proc = subprocess.Popen(cmd, **kwargs)
    threading.Thread(target=proc.wait, name="humming-bg-build-reaper", daemon=True).start()
    return proc


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


def find_kernel_name_in_cubin_cached(filename, func_keyword):
    # Pure-python ELF symbol iteration takes seconds per cubin and holds the
    # GIL, so warm-cache kernel loading is slower than cold compilation when
    # many kernels load concurrently. Persist the resolved name next to the
    # cubin and reuse it on later runs.
    cache_filename = filename + ".name"
    try:
        with open(cache_filename) as f:
            cached = f.read().strip()
    except OSError:
        cached = ""
    if cached and re.match(f"^_Z\\d+{func_keyword}", cached):
        return cached

    kernel_name = find_kernel_name_in_cubin(filename, func_keyword)
    tmp_filename = f"{cache_filename}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        with open(tmp_filename, "w") as f:
            f.write(kernel_name)
        os.replace(tmp_filename, cache_filename)
    except OSError:
        try:
            os.remove(tmp_filename)
        except OSError:
            pass
    return kernel_name


def find_kernel_name_in_cubin(filename, func_keyword):
    with open(filename, "rb") as f:
        elffile = ELFFile(f)
        symbol_table = elffile.get_section_by_name(".symtab")

        func_symbol_names = []
        for symbol in symbol_table.iter_symbols():
            if symbol["st_info"]["type"] != "STT_FUNC":
                continue
            if re.findall(f"^_Z\\d+{func_keyword}", symbol.name):
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
    re_result = re.findall("release (\\d+\\.\\d+)", result)
    if not re_result:
        raise RuntimeError(f"Invalid NVCC: {nvcc_path}")
    return re_result[0]


@functools.lru_cache(maxsize=1)
def get_humming_tmp_dir() -> str:
    tmp_dir = os.getenv("HUMMING_TMP_DIR")
    if tmp_dir is not None:
        return tmp_dir
    dirname = os.path.join(os.path.expanduser("~"), ".humming/tmp/")
    Path(dirname).mkdir(exist_ok=True, parents=True)
    return dirname


@functools.lru_cache(maxsize=1)
def get_humming_cache_dir() -> str:
    cache_dir = os.getenv("HUMMING_CACHE_DIR")
    if cache_dir is not None:
        return cache_dir
    return os.path.join(os.path.expanduser("~"), ".humming/cache/")


@functools.lru_cache(maxsize=1)
def get_humming_module_dir() -> str:
    tmp_dirname = get_humming_tmp_dir()
    dirname = os.path.join(tmp_dirname, "module/")
    Path(dirname).mkdir(exist_ok=True, parents=True)
    return dirname


@functools.lru_cache(maxsize=1)
def get_humming_lock_dir() -> str:
    tmp_dirname = get_humming_tmp_dir()
    dirname = os.path.join(tmp_dirname, "lock/")
    Path(dirname).mkdir(exist_ok=True, parents=True)
    return dirname


def hash_path_content(path: str, releative: bool = False, text_only: bool = True) -> str:
    data = {}

    assert os.path.exists(path)
    if os.path.isfile(path):
        filename = path.split("/")[-1] if releative else path
        with open(path, "rb") as f:
            data[filename] = str(f.read())
    else:
        pattern = os.path.join(path, "**/*")
        for fullname in sorted(glob.glob(pattern, recursive=True)):
            filename = os.path.relpath(fullname, path) if releative else fullname
            if not os.path.isfile(fullname):
                continue
            try:
                with open(fullname, "r") as f:
                    data[filename] = f.read()
            except UnicodeDecodeError:
                if text_only:
                    continue
                with open(fullname, "rb") as f:
                    data[filename] = str(f.read())

    return hash_to_hex(str(data))


@functools.lru_cache(maxsize=1)
def get_humming_lock_filename(name: str) -> str:
    if not name.endswith(".lock"):
        name = name + ".lock"
    lock_dirname = get_humming_lock_dir()
    return os.path.join(lock_dirname, name)


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def make_humming_module(func_name, result):
    dirname = get_humming_module_dir()
    if dirname not in sys.path:
        sys.path.append(dirname)

    content = f"def {func_name}():\n    return {result}\n"
    hash_hex = hash_to_hex(content)
    module_name = "humming_module_" + hash_hex

    lock_filename = jit_utils.get_humming_lock_filename(hash_hex)
    with FileLock(lock_filename):
        filename = module_name + ".py"
        if not (Path(dirname) / filename).exists():
            with open(Path(dirname) / filename, "w") as f:
                f.write(content)
                f.flush()

        importlib.invalidate_caches()
        return importlib.import_module(module_name)
