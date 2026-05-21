import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

import torch
import torch.utils.cpp_extension
from filelock import FileLock

import humming.utils.jit as jit_utils
from humming import dtypes
from humming.utils.cuda import filter_cuda_paths

_libs = {}
_launcher_inited = False


def register_op(
    name: str,
    impl_func: Callable,
    fake_impl_func: Callable | None = None,
    mutates_args: list[str] | None = None,
):
    mutates_args = [] if mutates_args is None else mutates_args
    schema_str = torch.library.infer_schema(impl_func, mutates_args=mutates_args)
    lib_name, op_name = name.split("::")

    if lib_name not in _libs:
        _lib = torch.library.Library(lib_name, "FRAGMENT")
        _libs[lib_name] = _lib

    _lib = _libs[lib_name]
    _lib.define(op_name + schema_str)
    _lib.impl(op_name, impl_func, dispatch_key="CUDA")
    if fake_impl_func is not None:
        _lib._register_fake(op_name, fake_impl_func)


def get_humming_launcher_build_dir(use_torch_stable_api: bool):
    import humming

    dirname = os.path.dirname(humming.__file__)
    launcher_code_hash = jit_utils.hash_path_content(
        path=os.path.join(dirname, "csrc/launcher/"),
        releative=True,
    )

    cache_dir = jit_utils.get_humming_cache_dir()
    py_version = f"py{sys.version_info.major}{sys.version_info.minor}"
    torch_major, torch_minor = torch.__version__.split(".")[:2]
    torch_version = f"torch{torch_major}{torch_minor}"
    abi_tag = "stable" if use_torch_stable_api else "nostable"
    version = f"{py_version}_{torch_version}_{abi_tag}"

    launcher_build_dir = os.path.join(cache_dir, f"launcher/{version}/{launcher_code_hash}")
    Path(launcher_build_dir).mkdir(exist_ok=True, parents=True)
    return launcher_build_dir


def _resolve_use_torch_stable_api() -> bool:
    """Decide whether to compile the launcher with the torch stable C ABI.

    Defaults to enabled for torch >= 2.11, which is the first release whose
    stable ABI registers all ScalarType cases humming relies on, including
    Float8_e8m0fnu (added in pytorch/pytorch#173669, gated by
    TORCH_FEATURE_VERSION >= TORCH_VERSION_2_11_0). On torch 2.10.x the
    stable ABI is missing Float8_e8m0fnu (ScalarType 44), so passing a
    UE8M0 weight scale through ``torch.ops.humming.launch_kernel`` fails
    with ``RuntimeError: Not yet supported ScalarType 44``.

    The default can be overridden via the ``HUMMING_USE_TORCH_STABLE_API``
    environment variable for users who want to force one path or the other
    (e.g. to test the stable ABI on an older torch with a custom build).
    """
    from packaging.version import Version

    override = os.environ.get("HUMMING_USE_TORCH_STABLE_API")
    if override is not None:
        return override.strip().lower() in ("1", "true", "yes", "on")
    return Version(torch.__version__) >= Version("2.11")


def init_humming_launcher():
    global _launcher_inited
    if _launcher_inited:
        return

    USE_TORCH_STABLE_API = _resolve_use_torch_stable_api()
    lock_filename = jit_utils.get_humming_lock_filename("launcher")
    with FileLock(lock_filename):
        import humming

        build_dir = get_humming_launcher_build_dir(USE_TORCH_STABLE_API)
        torch_lock_file = os.path.join(build_dir, "lock")
        if os.path.exists(torch_lock_file):
            os.unlink(torch_lock_file)

        dirname = os.path.dirname(humming.__file__)
        filename = os.path.join(dirname, "csrc/launcher/launcher.cpp")

        cuda_env = filter_cuda_paths(
            required_headers=["cuda.h", "crt/host_defines.h", "cuda/std/cstdint"],
        )

        torch.utils.cpp_extension.load(
            name="humming_launcher",
            sources=[filename],
            extra_include_paths=list(cuda_env["include_paths"]),
            extra_ldflags=["-lcuda", "-lc10_cuda", "-ltorch_cuda"],
            extra_cflags=["-O3", f"-DUSE_TORCH_STABLE_API={int(USE_TORCH_STABLE_API)}"],
            build_directory=build_dir,
        )

        _launcher_inited = True


def build_humming_launcher_in_bg():
    if os.getenv("HUMMING_DISABLE_PARALLEL_BUILD", "0") == "1":
        return None
    cmd = "import humming.ops.utils; humming.ops.utils.init_humming_launcher()"
    env = os.environ.copy()
    env["HUMMING_DISABLE_PARALLEL_BUILD"] = "1"
    subprocess.Popen(
        [sys.executable, "-c", cmd],
        env=env,
        stderr=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
    )


build_humming_launcher_in_bg()
