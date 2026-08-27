import dataclasses
import os
from pathlib import Path

import pytest
import torch

os.environ.setdefault("HUMMING_DISABLE_PARALLEL_BUILD", "1")

import humming.jit.compiler as compiler_module  # noqa: E402
import humming.utils.jit as jit_utils  # noqa: E402
from humming.jit.compiler import Compiler  # noqa: E402
from humming.jit.runtime import KernelRuntime  # noqa: E402


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


def test_kernel_runtime_instances_are_separated_by_context(monkeypatch):
    current_context = [object()]
    monkeypatch.setattr(
        KernelRuntime,
        "current_context",
        staticmethod(lambda: current_context[0]),
    )

    @dataclasses.dataclass(kw_only=True)
    class DummyKernel(KernelRuntime):
        value: int

        def init_sm_version(self):
            self.sm_version = 90
            self.sm_version_str = "90a"

        def init_kernel(self):
            pass

    first_kernel = DummyKernel(value=1)
    current_context[0] = object()
    second_kernel = DummyKernel(value=1)

    assert first_kernel is not second_kernel


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_launcher_loads_the_same_kernel_in_each_cuda_context():
    from humming import dtypes
    from humming.config import ComputeConfig, GemmType, LayerConfig
    from humming.testing import KernelTestCase, KernelTestRunner

    case = KernelTestCase(
        name="multi-context-smoke",
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

    with torch.cuda.device(0):
        runner = KernelTestRunner(case)
        kernel_config = runner.prepare_kernels((1,))[1][0][0]
        inputs_orig = torch.randn((1, 32), dtype=torch.bfloat16, device="cuda:0")
        _, inputs, input_scale = runner.prepare_inputs(inputs_orig)
        launch_tensors = {
            "inputs": inputs,
            "input_scale": input_scale,
            "sorted_ids": None,
            "expert_ids": None,
            "num_tokens_padded": None,
            "expert_layout": None,
        }
        output0 = runner._launch_kernel(1, launch_tensors, kernel_config)
        torch.cuda.synchronize(0)

        with torch.cuda.device(1):
            kernel_config = runner.prepare_kernels((1,))[1][0][0]
            runner.kernel_tensors = {
                key: value.cpu().to("cuda:1") if isinstance(value, torch.Tensor) else value
                for key, value in runner.kernel_tensors.items()
            }
            launch_tensors = {
                key: value.cpu().to("cuda:1") if isinstance(value, torch.Tensor) else value
                for key, value in launch_tensors.items()
            }

        with torch.cuda.device(1):
            output1 = runner._launch_kernel(1, launch_tensors, kernel_config)
            torch.cuda.synchronize(1)

        torch.testing.assert_close(output0.cpu(), output1.cpu(), rtol=0, atol=0)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_kernel_runtime_instances_are_context_local():
    from humming import dtypes
    from humming.kernel.process_input import ProcessInputKernel
    from humming.ops import hadamard_transform

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
        output0 = hadamard_transform(input0, block_size=32)
        kernel0 = make_kernel()
    with torch.cuda.device(1):
        output1 = hadamard_transform(input1, block_size=32)
        kernel1 = make_kernel()

    assert kernel0 is not kernel1
    torch.testing.assert_close(output0.cpu(), output1.cpu(), rtol=0, atol=0)
