import json
from pathlib import Path

import pytest

from humming.config import GemmType
from humming.tune.cache import (
    load_saved_shape_m_list,
    load_table,
    make_cache_key,
    save_table,
)


class _FakeMeta:
    def __init__(self, text: str = "meta-a"):
        self.text = text

    def to_str(self) -> str:
        return self.text


FINGERPRINT = {
    "gpu_name": "NVIDIA H20",
    "sm_version": 90,
    "num_sms": 78,
    "humming_version": "0.0.test",
    "torch_cuda_version": "12.4",
}


def _flags(**overrides):
    flags = {
        "use_f16_accum": False,
        "use_batch_invariant": False,
        "use_m_major_input_scale": False,
    }
    flags.update(overrides)
    return flags


def _table():
    return [
        [
            1,
            64,
            {
                "block_shape": (16, 128, 128),
                "warp_shape": (16, 32, 64),
                "num_stages": 3,
                "num_ctas_per_sm": 1,
                "use_f16_accum": False,
            },
        ],
        [
            65,
            128,
            {
                "block_shape": (32, 128, 128),
                "warp_shape": (32, 32, 64),
                "num_stages": 4,
                "num_ctas_per_sm": 2,
                "use_f16_accum": True,
            },
        ],
    ]


def test_round_trip_restores_nested_config_tuples(tmp_path):
    meta = _FakeMeta()
    flags = _flags()
    table = _table()

    path = save_table(meta, GemmType.DENSE, flags, FINGERPRINT, table, cache_dir=str(tmp_path))
    loaded = load_table(meta, GemmType.DENSE, flags, FINGERPRINT, cache_dir=str(tmp_path))

    assert Path(path).parent == tmp_path
    assert loaded == table
    cfg = loaded[0][2]
    assert type(cfg["block_shape"]) is tuple
    assert type(cfg["warp_shape"]) is tuple


def test_saved_shape_m_list_round_trip(tmp_path):
    meta = _FakeMeta()
    flags = _flags()
    shape_m_list = [8, 64, 512]

    path = save_table(
        meta,
        GemmType.DENSE,
        flags,
        FINGERPRINT,
        _table(),
        cache_dir=str(tmp_path),
        shape_m_list=shape_m_list,
    )

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    assert payload["_shape_m_list"] == shape_m_list
    assert (
        load_saved_shape_m_list(
            meta,
            GemmType.DENSE,
            flags,
            FINGERPRINT,
            cache_dir=str(tmp_path),
        )
        == shape_m_list
    )


def test_per_m_metadata_is_saved_without_configs(tmp_path):
    meta = _FakeMeta()
    flags = _flags()
    per_m = [
        {
            "shape_m": 64,
            "fine_ms": 0.5,
            "heuristic_ms": 1.0,
            "speedup": 2.0,
            "used_fallback": False,
            "same_as_heuristic": False,
        }
    ]

    path = save_table(
        meta,
        GemmType.DENSE,
        flags,
        FINGERPRINT,
        _table(),
        cache_dir=str(tmp_path),
        per_m=per_m,
    )

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    assert payload["_per_m"] == per_m
    assert "config" not in payload["_per_m"][0]


def test_old_payload_has_no_saved_shape_m_list(tmp_path):
    meta = _FakeMeta()
    flags = _flags()
    path = Path(
        save_table(
            meta,
            GemmType.DENSE,
            flags,
            FINGERPRINT,
            _table(),
            cache_dir=str(tmp_path),
            shape_m_list=[64, 256],
        )
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["_shape_m_list"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert (
        load_saved_shape_m_list(
            meta,
            GemmType.DENSE,
            flags,
            FINGERPRINT,
            cache_dir=str(tmp_path),
        )
        is None
    )
    assert load_table(
        meta,
        GemmType.DENSE,
        flags,
        FINGERPRINT,
        cache_dir=str(tmp_path),
    ) == _table()


@pytest.mark.parametrize(
    "field,value",
    [
        ("gpu_name", "NVIDIA H200"),
        ("sm_version", 80),
        ("num_sms", 132),
        ("humming_version", "0.0.other"),
        ("torch_cuda_version", "12.8"),
    ],
)
def test_fingerprint_mismatch_invalidates_whole_file(tmp_path, field, value):
    meta = _FakeMeta()
    flags = _flags()
    path = save_table(meta, GemmType.DENSE, flags, FINGERPRINT, _table(), cache_dir=str(tmp_path))
    fingerprint = dict(FINGERPRINT)
    fingerprint[field] = value

    assert load_table(meta, GemmType.DENSE, flags, fingerprint, cache_dir=str(tmp_path)) is None
    assert Path(path).exists()


def test_semantic_key_uses_meta_gemm_type_and_sorted_flags():
    meta = _FakeMeta()
    flags = _flags()
    reordered_flags = {
        "use_m_major_input_scale": False,
        "use_f16_accum": False,
        "use_batch_invariant": False,
    }
    key = make_cache_key(meta, GemmType.DENSE, flags)

    assert make_cache_key(meta, GemmType.DENSE, reordered_flags) == key
    assert make_cache_key(_FakeMeta("meta-b"), GemmType.DENSE, flags) != key
    assert make_cache_key(meta, GemmType.GROUPED_MASKED, flags) != key
    assert make_cache_key(meta, GemmType.DENSE, _flags(use_f16_accum=True)) != key


def test_key_ignores_sublayer_name():
    # An offline-tuned table (sublayer_name="") must resolve to the same key as
    # the serving layer, which labels its sublayers "w13"/"w2".
    flags = _flags()
    base = json.dumps({"shape_n": 512, "shape_k": 7168, "sublayer_name": ""}, sort_keys=True)
    named = json.dumps({"shape_n": 512, "shape_k": 7168, "sublayer_name": "w13"}, sort_keys=True)
    diff = json.dumps({"shape_n": 7168, "shape_k": 256, "sublayer_name": "w2"}, sort_keys=True)
    assert make_cache_key(_FakeMeta(base), GemmType.INDEXED, flags) == make_cache_key(
        _FakeMeta(named), GemmType.INDEXED, flags
    )
    assert make_cache_key(_FakeMeta(base), GemmType.INDEXED, flags) != make_cache_key(
        _FakeMeta(diff), GemmType.INDEXED, flags
    )


def test_corrupted_json_returns_miss(tmp_path):
    meta = _FakeMeta()
    flags = _flags()
    filename = tmp_path / (make_cache_key(meta, GemmType.DENSE, flags) + ".json")
    filename.write_text("{", encoding="utf-8")

    assert load_table(meta, GemmType.DENSE, flags, FINGERPRINT, cache_dir=str(tmp_path)) is None


def test_atomic_save_leaves_no_tmp_files(tmp_path):
    path = save_table(
        _FakeMeta(),
        GemmType.DENSE,
        _flags(),
        FINGERPRINT,
        _table(),
        cache_dir=str(tmp_path),
    )

    assert Path(path).exists()
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob(".*.tmp"))


def test_meta_str_mismatch_returns_miss(tmp_path):
    meta = _FakeMeta()
    flags = _flags()
    path = Path(
        save_table(meta, GemmType.DENSE, flags, FINGERPRINT, _table(), cache_dir=str(tmp_path))
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["_meta_str"] = "different-meta"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert load_table(meta, GemmType.DENSE, flags, FINGERPRINT, cache_dir=str(tmp_path)) is None


@pytest.mark.parametrize(
    "bad_table",
    [
        [[0, 64]],  # row too short
        [{"lo": 0, "hi": 64}],  # row is a dict
        [[0, 64, "not-a-config"]],  # config is not a dict
        [["lo", 64, {}]],  # bounds are not numeric
        [[0, 1e999, {}]],  # Infinity bound overflows int()
    ],
)
def test_malformed_table_structure_returns_miss(tmp_path, bad_table):
    meta = _FakeMeta()
    flags = _flags()
    path = save_table(
        meta, GemmType.DENSE, flags, FINGERPRINT, _table(), cache_dir=str(tmp_path)
    )
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    payload["table"] = bad_table
    Path(path).write_text(json.dumps(payload), encoding="utf-8")

    assert (
        load_table(meta, GemmType.DENSE, flags, FINGERPRINT, cache_dir=str(tmp_path))
        is None
    )
