import json
import os

import pytest

from humming.tune import measure
from humming.tune.measure import Measurer, MeasureRequest


def _config(block_m):
    return {
        "block_shape": (block_m, 128, 128),
        "warp_shape": (block_m, 32, 128),
        "use_stream_k": False,
        "use_f16_accum": False,
        "num_sms": 78,
        "num_stages": 3,
        "num_ctas_per_sm": 1,
    }


def _key(config):
    return json.dumps(config, sort_keys=True)


LAYER_ARGS = {
    "shape_n": 7168,
    "shape_k": 2048,
    "num_experts": 32,
    "a_dtype": "float8e4m3",
    "b_dtype": "int4",
    "bs_dtype": "bfloat16",
    "c_dtype": "bfloat16",
    "weight_scale_group_size": 128,
    "input_scale_group_size": 0,
    "top_k": 8,
    "is_moe_down": False,
    "balanced": True,
    "expert_max_tokens": None,
}


class _FakeWorker:
    """Scripted stand-in for _BenchWorker so the pure orchestration logic
    (two-stage selection, blacklist skip, correctness gate, worker-death
    escalation) can be tested deterministically without a GPU."""

    #: set by each test before Measurer construction
    script = {}

    def __init__(self, args_dict, num_spares, device_index=None):
        self.respawn_count = 0

    def request(self, msg, timeout):
        op = msg["op"]
        # set_ref carries no config; look it up under the None key.
        entry = self.script.get((op, _key(msg.get("config"))), {})
        if entry.get("die"):
            raise RuntimeError("worker died exitcode=43")
        if op == "bench":
            return {"op": op, "ms": entry.get("ms", 1.0), "error": entry.get("error", "")}
        if op == "set_ref":
            return {"op": op, "error": entry.get("error", "")}
        if op == "compare":
            return {"op": op, "passed": entry.get("passed", True), "error": entry.get("error", "")}
        raise AssertionError(op)

    def stop(self):
        pass


@pytest.fixture
def fake_worker(monkeypatch):
    monkeypatch.setattr(measure, "_BenchWorker", _FakeWorker)
    _FakeWorker.script = {}
    return _FakeWorker


def _by_config(timings, config):
    key = _key(config)
    return next(t for t in timings if _key(t.config) == key)


def test_picks_fastest_correct_config(fake_worker, tmp_path):
    fast, mid, base = _config(8), _config(16), _config(32)
    fake_worker.script = {
        ("bench", _key(fast)): {"ms": 0.10},
        ("bench", _key(mid)): {"ms": 0.20},
        ("bench", _key(base)): {"ms": 0.30},
        ("set_ref", _key(base)): {},
        ("compare", _key(fast)): {"passed": True},
    }
    with Measurer(LAYER_ARGS, "grouped_masked", progress_log_path=str(tmp_path / "p.log")) as m:
        timings = m.measure(
            MeasureRequest(shape_m=64, configs=[fast, mid, base], baseline_config=base)
        )

    assert {_key(t.config) for t in timings} == {_key(fast), _key(mid), _key(base)}
    winner = _by_config(timings, fast)
    assert winner.correctness_ok is True
    assert winner.fine_ms == pytest.approx(0.10)


def test_skips_fastest_when_correctness_fails(fake_worker, tmp_path):
    wrong, good, base = _config(8), _config(16), _config(32)
    fake_worker.script = {
        ("bench", _key(wrong)): {"ms": 0.10},
        ("bench", _key(good)): {"ms": 0.20},
        ("bench", _key(base)): {"ms": 0.30},
        ("set_ref", _key(base)): {},
        ("compare", _key(wrong)): {"passed": False},
        ("compare", _key(good)): {"passed": True},
    }
    with Measurer(LAYER_ARGS, "grouped_masked", progress_log_path=str(tmp_path / "p.log")) as m:
        timings = m.measure(
            MeasureRequest(shape_m=64, configs=[wrong, good, base], baseline_config=base)
        )

    assert _by_config(timings, wrong).correctness_ok is False
    assert _by_config(timings, good).correctness_ok is True


def test_refines_next_coarse_batch_when_correctness_pool_is_exhausted(
    fake_worker, tmp_path, caplog
):
    wrong1, wrong2 = _config(8), _config(16)
    good, batchmate, untouched, base = (
        _config(20),
        _config(24),
        _config(28),
        _config(32),
    )
    fake_worker.script = {
        ("bench", _key(wrong1)): {"ms": 0.10},
        ("bench", _key(wrong2)): {"ms": 0.20},
        ("bench", _key(good)): {"ms": 0.30},
        ("bench", _key(batchmate)): {"ms": 0.40},
        ("bench", _key(untouched)): {"ms": 0.50},
        ("bench", _key(base)): {"ms": 0.60},
        ("set_ref", _key(base)): {},
        ("compare", _key(wrong1)): {"passed": False},
        ("compare", _key(wrong2)): {"passed": False},
        ("compare", _key(good)): {"passed": True},
    }
    caplog.set_level("INFO")

    with Measurer(
        LAYER_ARGS,
        "grouped_masked",
        progress_log_path=str(tmp_path / "p.log"),
        topk=2,
    ) as m:
        timings = m.measure(
            MeasureRequest(
                shape_m=144,
                configs=[wrong1, wrong2, good, batchmate, untouched, base],
                baseline_config=base,
            )
        )

    assert _by_config(timings, wrong1).correctness_ok is False
    assert _by_config(timings, wrong2).correctness_ok is False
    assert _by_config(timings, good).correctness_ok is True
    assert _by_config(timings, good).fine_ms == pytest.approx(0.30)
    assert _by_config(timings, batchmate).fine_ms == pytest.approx(0.40)
    assert _by_config(timings, untouched).fine_ms is None
    assert (
        "correctness pool exhausted at shape_m=144, refining next 2 candidates (round 1)"
        in caplog.text
    )


def test_does_not_compare_candidates_slower_than_baseline(fake_worker, tmp_path):
    slower, base = _config(16), _config(32)
    fake_worker.script = {
        ("bench", _key(slower)): {"ms": 0.20},
        ("bench", _key(base)): {"ms": 0.10},
        ("set_ref", _key(base)): {},
        ("compare", _key(slower)): {"passed": False},
    }

    with Measurer(
        LAYER_ARGS,
        "grouped_masked",
        progress_log_path=str(tmp_path / "p.log"),
    ) as m:
        timings = m.measure(
            MeasureRequest(shape_m=64, configs=[slower, base], baseline_config=base)
        )

    assert _by_config(timings, slower).correctness_ok is None
    assert _by_config(timings, base).correctness_ok is True


def test_blacklisted_config_from_prior_run_is_skipped(fake_worker, tmp_path):
    poisoned, good = _config(8), _config(16)
    log = tmp_path / "p.log"
    # a prior run died mid-flight on `poisoned` at m=64: an S line with no closing E
    log.write_text(f"S\t64\t{_key(poisoned)}\n", encoding="utf-8")
    fake_worker.script = {
        ("bench", _key(good)): {"ms": 0.20},
        ("set_ref", _key(good)): {},
    }
    with Measurer(LAYER_ARGS, "grouped_masked", progress_log_path=str(log)) as m:
        timings = m.measure(
            MeasureRequest(shape_m=64, configs=[poisoned, good], baseline_config=good)
        )

    t = _by_config(timings, poisoned)
    assert t.fail_reason == "blacklisted"
    assert t.coarse_ms == float("inf")


def test_worker_death_twice_marks_globally_broken(fake_worker, tmp_path):
    killer, base = _config(8), _config(32)
    fake_worker.script = {
        ("bench", _key(killer)): {"die": True},
        ("bench", _key(base)): {"ms": 0.30},
        ("set_ref", _key(base)): {},
    }
    with Measurer(LAYER_ARGS, "grouped_masked", progress_log_path=str(tmp_path / "p.log")) as m:
        t1 = m.measure(MeasureRequest(shape_m=64, configs=[killer, base], baseline_config=base))
        t2 = m.measure(MeasureRequest(shape_m=128, configs=[killer, base], baseline_config=base))

    assert _by_config(t1, killer).fail_reason == "worker_died"
    # second death at a different m escalates to globally-broken → skipped, not re-run
    assert _by_config(t2, killer).fail_reason == "worker_died"


def test_rejects_unsupported_gemm_type(tmp_path):
    with pytest.raises(ValueError, match="unsupported gemm_type"):
        Measurer(LAYER_ARGS, "grouped_contiguous", progress_log_path=str(tmp_path / "p.log"))


def test_bench_worker_pins_spawn_device_and_restores_environment(monkeypatch):
    spawn_environments = []

    class FakeProcess:
        def start(self):
            spawn_environments.append(os.environ.get("CUDA_VISIBLE_DEVICES"))

    class FakeContext:
        @staticmethod
        def Queue():
            return object()

        @staticmethod
        def Process(**kwargs):
            return FakeProcess()

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "original")
    worker = measure._BenchWorker(LAYER_ARGS, num_spares=0, device_index=3)
    worker._ctx = FakeContext()

    worker._spawn_handle()

    assert spawn_environments == ["3"]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "original"


def test_bench_worker_without_device_keeps_spawn_environment(monkeypatch):
    spawn_environments = []

    class FakeProcess:
        def start(self):
            spawn_environments.append(os.environ.get("CUDA_VISIBLE_DEVICES"))

    class FakeContext:
        @staticmethod
        def Queue():
            return object()

        @staticmethod
        def Process(**kwargs):
            return FakeProcess()

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "original")
    worker = measure._BenchWorker(LAYER_ARGS, num_spares=0)
    worker._ctx = FakeContext()

    worker._spawn_handle()

    assert spawn_environments == ["original"]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "original"


def test_baseline_outside_config_set_still_prunes(fake_worker, tmp_path):
    """measured.py verifies bucket interiors with configs=[candidate] and the
    heuristic only as baseline_config: the baseline fine time must survive in
    a side channel so slower candidates are still pruned."""
    cand, base = _config(16), _config(32)
    fake_worker.script = {
        ("bench", _key(cand)): {"ms": 0.20},
        ("bench", _key(base)): {"ms": 0.10},
    }
    with Measurer(
        LAYER_ARGS, "grouped_masked", progress_log_path=str(tmp_path / "p.log")
    ) as m:
        timings = m.measure(
            MeasureRequest(shape_m=64, configs=[cand], baseline_config=base)
        )

    assert len(timings) == 1
    assert timings[0].correctness_ok is None  # pruned: slower than the baseline


def test_baseline_outside_config_set_still_gates(fake_worker, tmp_path):
    cand, base = _config(16), _config(32)
    fake_worker.script = {
        ("bench", _key(cand)): {"ms": 0.10},
        ("bench", _key(base)): {"ms": 0.20},
        ("compare", _key(cand)): {"passed": True},
    }
    with Measurer(
        LAYER_ARGS, "grouped_masked", progress_log_path=str(tmp_path / "p.log")
    ) as m:
        timings = m.measure(
            MeasureRequest(shape_m=64, configs=[cand], baseline_config=base)
        )

    assert timings[0].correctness_ok is True


def test_failed_baseline_reopens_slower_candidates(fake_worker, tmp_path):
    """When the baseline itself fails the golden gate, candidates that were
    pruned for being slower than it must be reconsidered."""
    slower, base = _config(16), _config(32)
    fake_worker.script = {
        ("bench", _key(slower)): {"ms": 0.20},
        ("bench", _key(base)): {"ms": 0.10},
        ("compare", _key(base)): {"passed": False},
        ("compare", _key(slower)): {"passed": True},
    }
    with Measurer(
        LAYER_ARGS, "grouped_masked", progress_log_path=str(tmp_path / "p.log")
    ) as m:
        timings = m.measure(
            MeasureRequest(shape_m=64, configs=[slower, base], baseline_config=base)
        )

    assert _by_config(timings, base).correctness_ok is False
    assert _by_config(timings, slower).correctness_ok is True


def test_baseline_compare_death_is_poisoned(fake_worker, tmp_path):
    base = _config(32)
    fake_worker.script = {
        ("bench", _key(base)): {"ms": 0.10},
        ("compare", _key(base)): {"die": True},
    }
    with Measurer(
        LAYER_ARGS, "grouped_masked", progress_log_path=str(tmp_path / "p.log")
    ) as m:
        timings = m.measure(
            MeasureRequest(shape_m=64, configs=[base], baseline_config=base)
        )
        assert _by_config(timings, base).fail_reason == "worker_died"
        # the poison log must carry the death so a rerun blacklists it
        assert (str(64), _key(base)) in m._poison_keys
