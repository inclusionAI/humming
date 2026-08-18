from copy import deepcopy
from types import SimpleNamespace

from humming.config import GemmType
from humming.tune import measured

_LAST_BUCKET_MAX = 1 << 30
_META = SimpleNamespace(input_scale_group_size=0)
_CACHED_HEURISTIC_TABLE = [
    [0, 128, {"name": "cached-small"}],
    [128, _LAST_BUCKET_MAX, {"name": "cached-large"}],
]


def _config(name):
    return {"name": name, "use_f16_accum": False}


def _per_m(*entries):
    return [{"shape_m": shape_m, "config": config} for shape_m, config in entries]


def _patch_heuristics(monkeypatch, table):
    def fake_get_heuristics_config(
        meta,
        shape_m,
        use_f16_accum,
        use_batch_invariant,
        use_m_major_input_scale,
        gemm_type,
    ):
        assert meta is _META
        assert shape_m is None
        assert use_f16_accum is False
        assert use_batch_invariant is False
        assert use_m_major_input_scale is False
        assert gemm_type == GemmType.DENSE
        return table

    monkeypatch.setattr(measured, "get_heuristics_config", fake_get_heuristics_config)


def _make_table(monkeypatch, per_m, heuristics):
    _patch_heuristics(monkeypatch, heuristics)
    return measured._make_table(
        per_m,
        _META,
        GemmType.DENSE,
        use_f16_accum=False,
        use_batch_invariant=False,
        use_m_major_input_scale=False,
    )


def _lookup(table, shape_m):
    matches = [config for lo, hi, config in table if lo < shape_m <= hi]
    assert len(matches) == 1
    return matches[0]


def test_appends_heuristic_tail_after_last_measured_shape(monkeypatch):
    measured_small = _config("measured-small")
    measured_large = _config("measured-large")
    heuristics = [
        [0, 128, _config("heuristic-small")],
        [128, _LAST_BUCKET_MAX, _config("heuristic-large")],
    ]

    table = _make_table(
        monkeypatch,
        _per_m((64, measured_small), (256, measured_large)),
        heuristics,
    )

    assert table == [
        [0, 64, measured_small],
        [64, 256, measured_large],
        [256, _LAST_BUCKET_MAX, _config("heuristic-large")],
    ]


def test_merges_matching_measured_and_heuristic_seam(monkeypatch):
    measured_small = _config("measured-small")
    seam_config = _config("seam")
    heuristics = [
        [0, 512, seam_config],
        [512, _LAST_BUCKET_MAX, _config("heuristic-large")],
    ]

    table = _make_table(
        monkeypatch,
        _per_m((64, measured_small), (256, seam_config)),
        heuristics,
    )

    assert table == [
        [0, 64, measured_small],
        [64, 512, seam_config],
        [512, _LAST_BUCKET_MAX, _config("heuristic-large")],
    ]


def test_table_has_seamless_unique_coverage(monkeypatch):
    table = _make_table(
        monkeypatch,
        _per_m((64, _config("measured-small")), (256, _config("measured-large"))),
        [
            [0, 128, _config("heuristic-small")],
            [128, 512, _config("heuristic-medium")],
            [512, _LAST_BUCKET_MAX, _config("heuristic-large")],
        ],
    )

    assert table[0][0] == 0
    assert table[-1][1] == _LAST_BUCKET_MAX
    assert all(left[0] < right[0] for left, right in zip(table, table[1:], strict=False))
    assert all(left[1] == right[0] for left, right in zip(table, table[1:], strict=False))
    for shape_m in (1, 64, 65, 256, 257, 512, 4096):
        _lookup(table, shape_m)


def test_measured_range_lookup_is_unchanged(monkeypatch):
    measured_small = _config("measured-small")
    measured_large = _config("measured-large")
    per_m = _per_m(
        (64, measured_small),
        (128, measured_small),
        (256, measured_large),
    )
    old_table = [
        [0, 128, measured_small],
        [128, _LAST_BUCKET_MAX, measured_large],
    ]
    new_table = _make_table(
        monkeypatch,
        per_m,
        [
            [0, 512, _config("heuristic-medium")],
            [512, _LAST_BUCKET_MAX, _config("heuristic-large")],
        ],
    )

    for shape_m in range(1, 257):
        assert _lookup(new_table, shape_m) == _lookup(old_table, shape_m)


def test_does_not_mutate_cached_heuristic_table(monkeypatch):
    frozen = deepcopy(_CACHED_HEURISTIC_TABLE)

    _make_table(
        monkeypatch,
        _per_m((64, _config("measured-small")), (256, _config("measured-large"))),
        _CACHED_HEURISTIC_TABLE,
    )

    assert _CACHED_HEURISTIC_TABLE == frozen


def _timing(config, fine_ms, correctness_ok):
    return measured.measure.ConfigTiming(
        config=config,
        coarse_ms=fine_ms,
        fine_ms=fine_ms,
        correctness_ok=correctness_ok,
        fail_reason=None,
    )


def _run_with_fake_measurer(monkeypatch, initial_timings, reverify_results):
    baseline = _config("baseline")
    requests = []

    class FakeMeasurer:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def measure(self, request):
            requests.append(request)
            if len(requests) == 1:
                return initial_timings
            config = request.configs[0]
            assert request.baseline_config == baseline
            return [_timing(config, 1.0, reverify_results[len(requests) - 2])]

    monkeypatch.setattr(
        measured.cache,
        "current_fingerprint",
        lambda device: {"sm_version": 90, "gpu_name": "NVIDIA H20", "num_sms": 78},
    )
    monkeypatch.setattr(
        measured,
        "get_search_space",
        lambda sm_version, gpu_name: SimpleNamespace(
            enumerate=lambda meta, gemm_type, num_sms, fast: [
                timing.config for timing in initial_timings
            ]
        ),
    )
    monkeypatch.setattr(measured, "_make_layer_args", lambda *args: {})
    monkeypatch.setattr(measured.measure, "Measurer", FakeMeasurer)
    monkeypatch.setattr(measured, "_make_table", lambda per_m, *args: [])
    monkeypatch.setattr(measured, "get_heuristics_config", lambda *args, **kwargs: baseline)

    _, per_m = measured._run_search(
        _META,
        GemmType.DENSE,
        [4096],
        progress_log_path="unused",
        verify_bucket_interiors=False,
    )
    return per_m[0], requests


def test_reverify_rejects_winner_and_uses_next_best(monkeypatch, caplog):
    winner = _config("winner")
    runner_up = _config("runner-up")

    result, requests = _run_with_fake_measurer(
        monkeypatch,
        [_timing(winner, 1.0, True), _timing(runner_up, 2.0, True)],
        [False, True],
    )

    assert result["config"] == runner_up
    assert result["fine_ms"] == 2.0
    assert result["used_fallback"] is False
    assert result["same_as_heuristic"] is False
    assert [request.configs for request in requests[1:]] == [[winner], [runner_up]]
    assert "config=" in caplog.text
    assert "shape_m=4096 reason=reverify-failed" in caplog.text


def test_reverify_stops_after_two_failures_and_falls_back(monkeypatch, caplog):
    winner = _config("winner")
    runner_up = _config("runner-up")
    third = _config("third")

    result, requests = _run_with_fake_measurer(
        monkeypatch,
        [
            _timing(winner, 1.0, True),
            _timing(runner_up, 2.0, True),
            _timing(third, 3.0, True),
        ],
        [False, False],
    )

    assert result["config"] == _config("baseline")
    assert result["fine_ms"] is None
    assert result["used_fallback"] is True
    assert result["same_as_heuristic"] is True
    assert [request.configs for request in requests[1:]] == [[winner], [runner_up]]
    assert caplog.text.count("reason=reverify-failed") == 2


def test_tune_and_save_persists_summary_without_configs(monkeypatch, caplog):
    table = [[0, _LAST_BUCKET_MAX, _config("winner")]]
    per_m = [
        {
            "shape_m": 64,
            "config": _config("baseline"),
            "fine_ms": 1.0,
            "heuristic_ms": 1.0,
            "speedup": 1.0,
            "used_fallback": False,
            "same_as_heuristic": True,
        },
        {
            "shape_m": 128,
            "config": _config("winner"),
            "fine_ms": 0.4,
            "heuristic_ms": 1.0,
            "speedup": 2.5,
            "used_fallback": False,
            "same_as_heuristic": False,
        },
    ]
    saved = {}

    monkeypatch.setattr(measured, "_run_search", lambda *args, **kwargs: (table, per_m))
    monkeypatch.setattr(measured.cache, "current_fingerprint", lambda device: {"gpu": "x"})

    def fake_save_table(*args, **kwargs):
        saved.update(kwargs)
        return "/tmp/measured.json"

    monkeypatch.setattr(measured.cache, "save_table", fake_save_table)
    caplog.set_level("INFO")

    returned_table, path = measured.tune_and_save(
        _META,
        GemmType.DENSE,
        [64, 128],
    )

    assert returned_table == table
    assert path == "/tmp/measured.json"
    assert [item["shape_m"] for item in saved["per_m"]] == [64, 128]
    assert all("config" not in item for item in saved["per_m"])
    assert (
        "tune summary: 2 points, 1 same-as-heuristic, max speedup 2.50@m=128"
        in caplog.text
    )


def _run_with_bucket_verification(
    monkeypatch,
    shape_m_list,
    winner_names,
    heuristics,
    checkpoint_results=None,
):
    checkpoint_results = checkpoint_results or {}
    grid_points = set(shape_m_list)
    requests = []
    candidate_names = list(dict.fromkeys(winner_names.values()))

    class FakeMeasurer:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def measure(self, request):
            requests.append(request)
            if request.shape_m in grid_points:
                winner_name = winner_names[request.shape_m]
                winner = next(
                    config for config in request.configs if config["name"] == winner_name
                )
                return [_timing(winner, 1.0, True)]

            config = request.configs[0]
            correctness_ok = checkpoint_results.get(request.shape_m, True)
            return [_timing(config, 1.0, correctness_ok)]

    def fake_get_heuristics_config(
        meta,
        shape_m,
        use_f16_accum,
        use_batch_invariant,
        use_m_major_input_scale,
        gemm_type,
    ):
        assert meta is _META
        assert use_f16_accum is False
        assert use_batch_invariant is False
        assert use_m_major_input_scale is False
        assert gemm_type == GemmType.DENSE
        if shape_m is None:
            return heuristics
        return _lookup(heuristics, shape_m)

    monkeypatch.setattr(
        measured.cache,
        "current_fingerprint",
        lambda device: {"sm_version": 90, "gpu_name": "NVIDIA H20", "num_sms": 78},
    )
    monkeypatch.setattr(
        measured,
        "get_search_space",
        lambda sm_version, gpu_name: SimpleNamespace(
            enumerate=lambda meta, gemm_type, num_sms, fast: [
                {"name": name} for name in candidate_names
            ]
        ),
    )
    monkeypatch.setattr(measured, "_make_layer_args", lambda *args: {})
    monkeypatch.setattr(measured.measure, "Measurer", FakeMeasurer)
    monkeypatch.setattr(measured, "get_heuristics_config", fake_get_heuristics_config)

    table, per_m = measured._run_search(
        _META,
        GemmType.DENSE,
        shape_m_list,
        progress_log_path="unused",
        reverify_winners=False,
    )
    return table, per_m, requests


def test_bucket_midpoint_failure_splices_only_that_bucket(monkeypatch):
    heuristics = [
        [0, 10, {"name": "heuristic-low"}],
        [10, 14, {"name": "heuristic-mid"}],
        [14, _LAST_BUCKET_MAX, {"name": "heuristic-high"}],
    ]
    table, _, requests = _run_with_bucket_verification(
        monkeypatch,
        [8, 16, 32],
        {8: "measured-low", 16: "bad", 32: "measured-top"},
        heuristics,
        checkpoint_results={12: False},
    )

    assert table == [
        [0, 8, _config("measured-low")],
        [8, 10, _config("heuristic-low")],
        [10, 14, _config("heuristic-mid")],
        [14, 16, _config("heuristic-high")],
        [16, 32, _config("measured-top")],
        [32, _LAST_BUCKET_MAX, _config("heuristic-high")],
    ]
    failed_requests = [request for request in requests if request.shape_m == 12]
    assert len(failed_requests) == 1
    assert failed_requests[0].baseline_config == _config("heuristic-mid")


def test_matching_checkpoint_heuristics_skip_all_verification_calls(monkeypatch):
    table, _, requests = _run_with_bucket_verification(
        monkeypatch,
        [1, 8, 24],
        {1: "narrow", 8: "shared", 24: "shared"},
        [
            [0, 1, {"name": "narrow"}],
            [1, _LAST_BUCKET_MAX, {"name": "shared"}],
        ],
    )

    assert [request.shape_m for request in requests] == [1, 8, 24]
    assert table == [
        [0, 1, _config("narrow")],
        [1, _LAST_BUCKET_MAX, _config("shared")],
    ]


def test_bucket_replacement_preserves_table_invariants(monkeypatch):
    table, _, _ = _run_with_bucket_verification(
        monkeypatch,
        [8, 16, 32],
        {8: "measured-low", 16: "bad", 32: "merged"},
        [
            [0, 10, {"name": "heuristic-low"}],
            [10, 14, {"name": "heuristic-mid"}],
            [14, _LAST_BUCKET_MAX, {"name": "merged"}],
        ],
        checkpoint_results={12: False},
    )

    assert table[0][0] == 0
    assert table[-1][1] == _LAST_BUCKET_MAX
    assert all(lo < hi for lo, hi, _config_value in table)
    assert all(left[0] < right[0] for left, right in zip(table, table[1:], strict=False))
    assert all(left[1] == right[0] for left, right in zip(table, table[1:], strict=False))
    assert all(
        measured._config_key(left[2]) != measured._config_key(right[2])
        for left, right in zip(table, table[1:], strict=False)
    )


def test_passing_top_measured_bucket_is_retained(monkeypatch):
    table, _, requests = _run_with_bucket_verification(
        monkeypatch,
        [8, 16, 32],
        {8: "baseline", 16: "baseline", 32: "top"},
        [[0, _LAST_BUCKET_MAX, {"name": "baseline"}]],
    )

    assert table == [
        [0, 16, _config("baseline")],
        [16, 32, _config("top")],
        [32, _LAST_BUCKET_MAX, _config("baseline")],
    ]
    interior_requests = [request for request in requests if request.shape_m not in {8, 16, 32}]
    assert [request.shape_m for request in interior_requests] == [24, 17]
    assert all(request.baseline_config == _config("baseline") for request in interior_requests)
