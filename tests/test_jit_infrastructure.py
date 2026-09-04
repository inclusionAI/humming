import os
from pathlib import Path

import pytest
import torch

os.environ.setdefault("HUMMING_DISABLE_PARALLEL_BUILD", "1")

import humming.jit.compiler as compiler_module  # noqa: E402
import humming.utils.jit as jit_utils  # noqa: E402
from humming.jit.compiler import Compiler  # noqa: E402


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
def test_launcher_runs_device_local_kernels():
    from humming import dtypes
    from humming.config import ComputeConfig, GemmType, LayerConfig
    from humming.testing import KernelTestCase, KernelTestRunner

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
            result = runner.run((1,))[0]
            torch.cuda.synchronize(device_index)
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
            source_dtype=dtypes.float32,
            target_dtype=dtypes.float32,
            hidden_size=32,
            quant_group_size=32,
            hadamard_block_size=32,
            threads_per_task=32,
            values_per_thread=1,
            quant_mode="none",
            use_tile_partition=True,
            tile_size=32,
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
