import os
from pathlib import Path

import pytest
import torch

import humming.jit.compiler as compiler_module
import humming.utils.jit as jit_utils
from humming.jit.compiler import Compiler


class _FakeCompiler(Compiler):
    compile_calls = 0

    @classmethod
    def signature(cls):
        return "fake-compiler"

    @classmethod
    def get_flags(cls, sm_version, disable_fast_math=False):
        return [f"sm={sm_version}"]

    @classmethod
    def _compile(cls, source_path, cache_dirname, sm_version, kernel_expr, flags):
        cls.compile_calls += 1
        (Path(cache_dirname) / "kernel_tmp.cubin").write_bytes(b"fake-cubin")
        return 0, "stdout", "stderr"


def test_compiler_publishes_cache_while_holding_lock(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()
    lock_active = False

    class TrackingLock:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            nonlocal lock_active
            assert not lock_active
            lock_active = True

        def __exit__(self, exc_type, exc, traceback):
            nonlocal lock_active
            lock_active = False

    real_replace = os.replace

    def checked_replace(source, destination):
        assert lock_active, "the final cubin must be published before releasing the cache lock"
        return real_replace(source, destination)

    monkeypatch.setattr(jit_utils, "get_humming_cache_dir", lambda: cache_dir.as_posix())
    monkeypatch.setattr(
        jit_utils,
        "get_humming_lock_filename",
        lambda name: (lock_dir / f"{name}.lock").as_posix(),
    )
    monkeypatch.setattr(Compiler, "cuh_last_update_time", staticmethod(lambda: "headers"))
    monkeypatch.setattr(compiler_module, "FileLock", TrackingLock)
    monkeypatch.setattr(compiler_module.os, "replace", checked_replace)
    _FakeCompiler.compile_calls = 0

    first = _FakeCompiler.compile("source", "90a", "kernel")
    second = _FakeCompiler.compile("source", "90a", "kernel")

    assert first == second
    assert Path(first).read_bytes() == b"fake-cubin"
    assert _FakeCompiler.compile_calls == 1
    assert not list(cache_dir.rglob("kernel_tmp.cubin"))


def test_precompiled_manifest_uses_content_hashes(tmp_path, monkeypatch):
    arch = jit_utils.get_native_arch()
    assert arch is not None

    package_dir = tmp_path / "humming"
    native_dir = package_dir / "_native" / arch
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    native_dir.mkdir(parents=True)
    source = source_dir / "helper.cpp"
    artifact = native_dir / "helper"
    source.write_text("version one")
    artifact.write_bytes(b"native version one")

    jit_utils.write_precompiled_artifact_manifest(native_dir, {artifact.name: source})
    monkeypatch.setattr(jit_utils, "__file__", (package_dir / "utils/jit.py").as_posix())

    assert jit_utils.get_precompiled_artifact_path(source, artifact.name) == artifact

    source_stat = source.stat()
    os.utime(
        source,
        ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns + 1_000_000_000),
    )
    assert jit_utils.get_precompiled_artifact_path(source, artifact.name) == artifact

    source_stat = source.stat()
    source.write_text("version two")
    os.utime(source, ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns))
    assert jit_utils.get_precompiled_artifact_path(source, artifact.name) is None


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
@pytest.mark.parametrize("tuning_source", ("heuristic", "batch_invariant", "heuristic+batch_invariant"))
def test_launcher_runs_device_local_kernels(monkeypatch, tuning_source):
    from humming import dtypes
    from humming.config import ComputeConfig, GemmType, LayerConfig
    from humming.kernel.humming import HummingKernel
    from humming.testing import KernelTestCase, KernelTestRunner, assert_kernel_test_shape_coverage

    monkeypatch.setenv("HUMMING_TEST_TUNING_SOURCE", tuning_source)
    compile_batches = []
    compile_many = HummingKernel.compile_many

    def record_compile_batch(kernel_specs, device):
        compile_batches.append([config["use_batch_invariant"] for _, config in kernel_specs])
        return compile_many(kernel_specs, device)

    monkeypatch.setattr(HummingKernel, "compile_many", record_compile_batch)

    def make_case():
        return KernelTestCase(
            name="multi-device-smoke",
            layer_config=LayerConfig(
                shape_n=64,
                shape_k=32,
                a_dtype=dtypes.bfloat16,
                b_dtype=dtypes.uint4,
                c_dtype=dtypes.bfloat16,
                bs_dtype=dtypes.bfloat16,
            ),
            compute_config=ComputeConfig(gemm_type=GemmType.DENSE),
            seed=2026,
        )

    for device_index in range(2):
        with torch.cuda.device(device_index):
            runner = KernelTestRunner(make_case())
            results = runner.run((1,))
            torch.cuda.synchronize(device_index)
            assert_kernel_test_shape_coverage(results, (1,))
            assert len(compile_batches) == device_index + 1
            expected_modes = [source == "batch_invariant" for source in tuning_source.split("+")]
            assert compile_batches[-1] == expected_modes
            assert not runner.compute_config.use_batch_invariant
            for result in results:
                torch.testing.assert_close(
                    result.outputs,
                    result.outputs_ref,
                    rtol=runner.test_case.rtol,
                    atol=runner.test_case.atol,
                )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_kernel_runtime_instances_are_context_local():
    from humming import dtypes
    from humming.kernel.process_input import ProcessInputKernel
    from humming.ops import process_input

    def make_kernel():
        return ProcessInputKernel(
            input_dtype=dtypes.float32,
            hidden_size=32,
            quant_group_size=32,
            hadamard_block_size=32,
            threads_per_task=32,
            values_per_thread=1,
            quant_mode="none",
            use_tile_partition=True,
        )

    values = torch.randn((2, 32), dtype=torch.float32)
    input0 = values.to("cuda:0")
    input1 = values.to("cuda:1")

    with torch.cuda.device(0):
        output0 = process_input(input0, hadamard_block_size=32)[0]
        kernel0 = make_kernel()
    with torch.cuda.device(1):
        output1 = process_input(input1, hadamard_block_size=32)[0]
        kernel1 = make_kernel()

    assert kernel0 is not kernel1
    torch.testing.assert_close(output0.cpu(), output1.cpu(), rtol=0, atol=0)


def test_nvrtc_signature_uses_subprocess_library(monkeypatch):
    import ctypes

    loaded_paths = []

    class VersionFunction:
        def __call__(self, major, minor):
            ctypes.cast(major, ctypes.POINTER(ctypes.c_int))[0] = 13
            ctypes.cast(minor, ctypes.POINTER(ctypes.c_int))[0] = 2
            return 0

    class Library:
        nvrtcVersion = VersionFunction()

    def load_library(path):
        loaded_paths.append(path)
        return Library()

    monkeypatch.setattr(compiler_module, "get_nvrtc_library_path", lambda: "/toolkit/libnvrtc.so")
    monkeypatch.setattr(compiler_module.ctypes, "CDLL", load_library)
    assert compiler_module.NVRTCCompiler.signature() == "nvrtc+13.2"
    assert loaded_paths == ["/toolkit/libnvrtc.so"]


@pytest.mark.parametrize("limit", (1, 3, 32))
def test_compile_many_obeys_worker_limit(monkeypatch, limit):
    from contextlib import nullcontext

    from humming.jit import runtime

    worker_counts = []

    class RecordingExecutor:
        def __init__(self, max_workers):
            worker_counts.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def map(self, function, specs):
            return map(function, specs)

    monkeypatch.setattr(runtime, "get_parallel_build_workers", lambda: limit)
    monkeypatch.setattr(runtime, "ThreadPoolExecutor", RecordingExecutor)
    monkeypatch.setattr(runtime.torch.cuda, "device", lambda device: nullcontext())
    specs = [(lambda value: value, {"value": value}) for value in range(5)]
    assert runtime.KernelRuntime.compile_many(specs, device=0) == list(range(5))
    assert worker_counts == ([] if limit == 1 else [min(limit, len(specs))])
    assert runtime.KernelRuntime.compile_many([], device=0) == []
