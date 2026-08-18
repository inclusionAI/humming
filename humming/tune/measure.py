import dataclasses
import json
import logging
import math
import multiprocessing
import os
import queue
import statistics
import time

from humming.tune._worker import worker_main as _worker_main

_COARSE_TIMEOUT_SECONDS = 90
_FINE_TIMEOUT_SECONDS = 240
_WORKER_INIT_TIMEOUT_SECONDS = 600
_COARSE_WARMUP_MS = 25
_FINE_WARMUP_MS = 100
_FINE_TRIALS = 3
_CORRECTNESS_REFINE_ROUNDS = 3
_GOLDEN_PROGRESS_KEY = "__golden__"
_LOGGER = logging.getLogger(__name__)
# grouped_contiguous is intentionally excluded: the worker's input generation and
# correctness comparison only cover these three layouts. Accepting it would silently
# mis-measure. Add it here once _worker.py grows a contiguous code path.
_GEMM_TYPE_VALUES = {"dense", "indexed", "grouped_masked"}
# Derived flags (use_fused_e8m0_scale, is_*_weight_scale, ...) are intentionally
# absent: LayerConfig.__post_init__ re-derives them purely from the carried
# dtype/group-size fields, so they survive the worker round-trip without being
# passed. Do not "fix" a missing flag by adding it here — verify the derivation
# inputs are present instead.
_LAYER_ARG_NAMES = (
    "shape_n",
    "shape_k",
    "num_experts",
    "a_dtype",
    "b_dtype",
    "bs_dtype",
    "c_dtype",
    "weight_scale_group_size",
    "input_scale_group_size",
    "top_k",
    "is_moe_down",
    "balanced",
    "expert_max_tokens",
)


@dataclasses.dataclass(kw_only=True)
class MeasureRequest:
    shape_m: int
    configs: list[dict]
    baseline_config: dict


@dataclasses.dataclass(kw_only=True)
class ConfigTiming:
    config: dict
    coarse_ms: float
    fine_ms: float | None
    correctness_ok: bool | None
    fail_reason: str | None


class _ProgressLog:
    def __init__(self, progress_log_path):
        dirname = os.path.dirname(progress_log_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        self._progress_log = open(progress_log_path, "a")

    def enter(self, shape_m, key):
        self._progress_log.write(f"S\t{shape_m}\t{key}\n")
        self._progress_log.flush()

    def exit(self):
        self._progress_log.write("E\n")
        self._progress_log.flush()

    def close(self):
        self._progress_log.close()


class _BenchWorker:
    def __init__(self, args_dict: dict, num_spares: int, device_index=None):
        self._args_dict = dict(args_dict)
        self._num_spares = num_spares
        self._device_index = device_index
        self._ctx = multiprocessing.get_context("spawn")
        self.respawn_count = -1
        self._active = None
        self._spares = []

    def _spawn_handle(self):
        cmd_q = self._ctx.Queue()
        res_q = self._ctx.Queue()
        proc = self._ctx.Process(
            target=_worker_main,
            args=(self._args_dict, cmd_q, res_q),
            daemon=True,
        )
        if self._device_index is None:
            proc.start()
        else:
            env_name = "CUDA_VISIBLE_DEVICES"
            had_value = env_name in os.environ
            previous_value = os.environ.get(env_name)
            os.environ[env_name] = str(self._device_index)
            try:
                proc.start()
            finally:
                if had_value:
                    os.environ[env_name] = previous_value
                else:
                    os.environ.pop(env_name, None)
        return {"proc": proc, "cmd_q": cmd_q, "res_q": res_q, "ready": False}

    def _wait_result(self, handle, timeout):
        deadline = time.monotonic() + timeout
        while True:
            try:
                wait_s = min(5.0, max(0.1, deadline - time.monotonic()))
                return handle["res_q"].get(timeout=wait_s)
            except queue.Empty:
                if not handle["proc"].is_alive():
                    raise RuntimeError(
                        f"worker died exitcode={handle['proc'].exitcode}"
                    ) from None
                if time.monotonic() > deadline:
                    self._kill_handle(handle)
                    raise RuntimeError("worker timeout (hung kernel)") from None

    def _wait_ready(self, handle, timeout=_WORKER_INIT_TIMEOUT_SECONDS):
        if not handle["ready"]:
            msg = self._wait_result(handle, timeout)
            assert msg["op"] == "ready", f"{msg=}"
            handle["ready"] = True
        return handle

    def _refill_spares(self):
        import threading

        while len(self._spares) < self._num_spares:
            spare = self._spawn_handle()

            def warm(handle=spare):
                try:
                    self._wait_ready(handle)
                except Exception:
                    pass

            thread = threading.Thread(target=warm, daemon=True)
            thread.start()
            self._spares.append((spare, thread))

    def ensure_started(self):
        if self._active is not None and self._active["proc"].is_alive():
            return
        self.respawn_count += 1
        handle = None
        while handle is None and self._spares:
            spare, thread = self._spares.pop(0)
            thread.join()
            if spare["proc"].is_alive() and spare["ready"]:
                handle = spare
            else:
                self._kill_handle(spare)
        if handle is None:
            handle = self._wait_ready(self._spawn_handle())
        self._active = handle
        self._refill_spares()

    def request(self, msg, timeout):
        self.ensure_started()
        self._active["cmd_q"].put(msg)
        return self._wait_result(self._active, timeout)

    @staticmethod
    def _kill_handle(handle):
        if handle is not None and handle["proc"].is_alive():
            handle["proc"].kill()
            handle["proc"].join(10)

    def stop(self):
        handles = [self._active] + [spare for spare, _ in self._spares]
        for handle in handles:
            if handle is None:
                continue
            try:
                if handle["proc"].is_alive() and handle["ready"]:
                    handle["cmd_q"].put({"op": "stop"})
                    handle["proc"].join(10)
            finally:
                self._kill_handle(handle)


def _load_poison_keys(progress_log_path):
    poison = set()
    if not os.path.exists(progress_log_path):
        return poison
    in_flight = None
    with open(progress_log_path) as f:
        for line in f:
            if line.startswith("S\t"):
                if in_flight is not None:
                    poison.add(in_flight)
                parts = line.rstrip("\n").split("\t", 2)
                in_flight = (parts[1], parts[2]) if len(parts) == 3 else None
            elif line.startswith("E"):
                in_flight = None
    if in_flight is not None:
        poison.add(in_flight)
    return poison


def _make_config_key(config):
    return json.dumps(config, sort_keys=True)


def _normalize_layer_args(layer_args) -> dict:
    if isinstance(layer_args, dict):
        args = dict(layer_args)
    else:
        args = vars(layer_args).copy()
    missing = [name for name in _LAYER_ARG_NAMES if name not in args]
    if missing:
        raise ValueError(f"layer_args missing required fields: {missing}")
    return args


def _format_bench_error(error):
    if not error:
        return None
    return f"bench_error: {error}"


def _get_gemm_type_value(gemm_type):
    value = gemm_type if isinstance(gemm_type, str) else getattr(gemm_type, "value", gemm_type)
    if value not in _GEMM_TYPE_VALUES:
        raise ValueError(f"unsupported gemm_type: {gemm_type}")
    return value


def _dedupe_records(records):
    seen = set()
    result = []
    for record in records:
        if record["key"] in seen:
            continue
        seen.add(record["key"])
        result.append(record)
    return result


def _make_records(configs, baseline_config):
    records = []
    records_by_key = {}
    ordered_keys = []

    def add_config(config, returned):
        config = dict(config)
        key = _make_config_key(config)
        if key not in records_by_key:
            records_by_key[key] = {
                "config": config,
                "key": key,
                "returned": returned,
            }
            records.append(records_by_key[key])
        if returned:
            ordered_keys.append(key)

    for config in configs:
        add_config(config, True)
    add_config(baseline_config, False)
    return records, records_by_key, ordered_keys


def _mark_worker_died(poison_keys, death_counts, globally_broken_keys, shape_m, record):
    poison_keys.add((str(shape_m), record["key"]))
    death_counts[record["key"]] = death_counts.get(record["key"], 0) + 1
    if death_counts[record["key"]] >= 2:
        globally_broken_keys.add(record["key"])


def _bench_once(worker, progress, record, shape_m, warmup, rep, timeout):
    progress.enter(shape_m, record["key"])
    try:
        res = worker.request(
            {
                "op": "bench",
                "m": shape_m,
                "config": record["config"],
                "warmup": warmup,
                "rep": rep,
            },
            timeout,
        )
    except RuntimeError:
        return float("inf"), "worker_died"
    progress.exit()
    return res["ms"], _format_bench_error(res["error"])


def _fine_bench(worker, progress, record, shape_m, fine_rep):
    timings = []
    for _ in range(_FINE_TRIALS):
        ms, fail_reason = _bench_once(
            worker,
            progress,
            record,
            shape_m,
            warmup=_FINE_WARMUP_MS,
            rep=fine_rep,
            timeout=_FINE_TIMEOUT_SECONDS,
        )
        if fail_reason is not None:
            return float("inf"), fail_reason
        timings.append(ms)
    return statistics.median(timings), None


def _set_ref_output(worker, progress, shape_m):
    progress.enter(shape_m, _GOLDEN_PROGRESS_KEY)
    try:
        res = worker.request(
            {"op": "set_ref", "m": shape_m},
            _FINE_TIMEOUT_SECONDS,
        )
    except RuntimeError:
        return "worker_died"
    progress.exit()
    return _format_bench_error(res["error"])


def _compare_against_ref(worker, progress, record, shape_m):
    progress.enter(shape_m, record["key"])
    try:
        res = worker.request(
            {"op": "compare", "m": shape_m, "config": record["config"]},
            _FINE_TIMEOUT_SECONDS,
        )
    except RuntimeError:
        return False, "worker_died"
    progress.exit()
    return res["passed"], _format_bench_error(res["error"])


class Measurer:
    def __init__(
        self,
        layer_args,
        gemm_type,
        *,
        progress_log_path,
        coarse_rep_ms=25,
        fine_rep_ms=500,
        topk=8,
        num_spares=2,
        device_index=None,
    ):
        if coarse_rep_ms <= 0 or fine_rep_ms <= 0:
            raise ValueError("coarse_rep_ms and fine_rep_ms must be positive")
        if topk <= 0:
            raise ValueError("topk must be positive")
        if num_spares < 0:
            raise ValueError("num_spares must be non-negative")

        self._gemm_type = _get_gemm_type_value(gemm_type)
        self._layer_args = _normalize_layer_args(layer_args)
        self._layer_args["gemm_type"] = self._gemm_type
        self._coarse_rep_ms = coarse_rep_ms
        self._fine_rep_ms = fine_rep_ms
        self._topk = topk
        self._poison_keys = _load_poison_keys(progress_log_path)
        self._death_counts = {}
        self._globally_broken_keys = set()
        self._progress = _ProgressLog(progress_log_path)
        self._worker = _BenchWorker(
            self._layer_args, num_spares, device_index=device_index
        )
        self._closed = False

    def _fine_bench_records(self, records, shape_m, results_by_key):
        fine_results = []
        worker_died = False
        for record in records:
            ms, fail_reason = _fine_bench(
                self._worker,
                self._progress,
                record,
                shape_m,
                self._fine_rep_ms,
            )
            if record["returned"]:
                timing = results_by_key[record["key"]]
                timing.fine_ms = ms
                timing.fail_reason = fail_reason
            if fail_reason == "worker_died":
                worker_died = True
                _mark_worker_died(
                    self._poison_keys,
                    self._death_counts,
                    self._globally_broken_keys,
                    shape_m,
                    record,
                )
            fine_results.append(
                {"record": record, "ms": ms, "fail_reason": fail_reason}
            )
        return fine_results, worker_died

    def _set_reference(self, shape_m):
        ref_fail_reason = _set_ref_output(
            self._worker,
            self._progress,
            shape_m,
        )
        if ref_fail_reason is None:
            return True
        _LOGGER.warning(
            "golden reference failed at shape_m=%s: %s", shape_m, ref_fail_reason
        )
        return False

    def _compare_fine_candidates(
        self,
        fine_results,
        baseline_record,
        shape_m,
        results_by_key,
        baseline_ms,
    ):
        baseline_key = baseline_record["key"]
        winner_key = None
        for item in sorted(fine_results, key=lambda result: result["ms"]):
            if not math.isfinite(item["ms"]):
                continue
            record = item["record"]
            # Only candidates that beat the baseline are worth a correctness
            # check; the baseline itself is verified once at the end when no
            # faster candidate survives.
            if record["key"] == baseline_key or item["ms"] >= baseline_ms:
                continue
            already = results_by_key.get(record["key"])
            if already is not None and already.correctness_ok is False:
                continue
            passed, fail_reason = _compare_against_ref(
                self._worker,
                self._progress,
                record,
                shape_m,
            )
            if record["returned"]:
                timing = results_by_key[record["key"]]
                timing.correctness_ok = passed
                timing.fail_reason = fail_reason
            _LOGGER.info(
                "golden correctness shape_m=%s baseline=%s passed=%s config=%s error=%s",
                shape_m,
                False,
                passed,
                record["key"],
                fail_reason or "",
            )
            if fail_reason == "worker_died":
                _mark_worker_died(
                    self._poison_keys,
                    self._death_counts,
                    self._globally_broken_keys,
                    shape_m,
                    record,
                )
                if not self._set_reference(shape_m):
                    return False, False, False
            if passed:
                winner_key = record["key"]
                break
        if winner_key is None:
            return False, True, False
        return True, True, False

    def _verify_baseline(self, baseline_record, shape_m, results_by_key, baseline_ms):
        """Returns True/False for the baseline verdict, None if not verifiable."""
        if not math.isfinite(baseline_ms):
            return None
        timing = results_by_key.get(baseline_record["key"])
        passed, fail_reason = _compare_against_ref(
            self._worker,
            self._progress,
            baseline_record,
            shape_m,
        )
        if timing is not None:
            timing.correctness_ok = passed
            timing.fail_reason = fail_reason
        _LOGGER.info(
            "golden correctness shape_m=%s baseline=%s passed=%s config=%s error=%s",
            shape_m,
            True,
            passed,
            baseline_record["key"],
            fail_reason or "",
        )
        if fail_reason == "worker_died":
            _mark_worker_died(
                self._poison_keys,
                self._death_counts,
                self._globally_broken_keys,
                shape_m,
                baseline_record,
            )
            if not self._set_reference(shape_m):
                return None
        return passed

    def measure(self, req: MeasureRequest) -> list[ConfigTiming]:
        assert isinstance(req, MeasureRequest), f"{req=}"
        records, records_by_key, ordered_keys = _make_records(req.configs, req.baseline_config)
        baseline_key = _make_config_key(req.baseline_config)
        baseline_record = records_by_key[baseline_key]
        results_by_key = {
            record["key"]: ConfigTiming(
                config=dict(record["config"]),
                coarse_ms=float("inf"),
                fine_ms=None,
                correctness_ok=None,
                fail_reason=None,
            )
            for record in records
            if record["returned"]
        }

        coarse_results = []
        for record in records:
            if not record["returned"]:
                continue
            timing = results_by_key[record["key"]]
            if record["key"] in self._globally_broken_keys:
                timing.fail_reason = "worker_died"
                coarse_results.append({"record": record, "ms": float("inf")})
                continue
            if (str(req.shape_m), record["key"]) in self._poison_keys:
                timing.fail_reason = "blacklisted"
                coarse_results.append({"record": record, "ms": float("inf")})
                continue

            ms, fail_reason = _bench_once(
                self._worker,
                self._progress,
                record,
                req.shape_m,
                warmup=_COARSE_WARMUP_MS,
                rep=self._coarse_rep_ms,
                timeout=_COARSE_TIMEOUT_SECONDS,
            )
            timing.coarse_ms = ms
            timing.fail_reason = fail_reason
            if fail_reason == "worker_died":
                _mark_worker_died(
                    self._poison_keys,
                    self._death_counts,
                    self._globally_broken_keys,
                    req.shape_m,
                    record,
                )
            coarse_results.append({"record": record, "ms": ms})

        coarse_candidates = [
            item["record"]
            for item in sorted(coarse_results, key=lambda item: item["ms"])
            if math.isfinite(item["ms"])
            and item["record"]["key"] not in self._globally_broken_keys
        ]
        fine_records = coarse_candidates[: self._topk]
        if (
            baseline_record["key"] not in self._globally_broken_keys
            and (str(req.shape_m), baseline_record["key"]) not in self._poison_keys
        ):
            fine_records.append(baseline_record)
        fine_records = _dedupe_records(fine_records)

        fine_results, _ = self._fine_bench_records(
            fine_records, req.shape_m, results_by_key
        )
        # The baseline may not be in the returned config set; keep its fine
        # timing in a side channel so slower-than-baseline pruning and the
        # final baseline verification never lose it.
        baseline_fine_ms = next(
            (
                item["ms"]
                for item in fine_results
                if item["record"]["key"] == baseline_key
            ),
            float("inf"),
        )
        ref_ok = bool(fine_results) and self._set_reference(req.shape_m)

        passed = False
        baseline_wins = False
        if ref_ok:
            passed, ref_ok, baseline_wins = self._compare_fine_candidates(
                fine_results,
                baseline_record,
                req.shape_m,
                results_by_key,
                baseline_fine_ms,
            )

        refined_keys = {record["key"] for record in fine_records}
        remaining_records = [
            record
            for record in coarse_candidates
            if record["key"] not in refined_keys
        ]
        for round_index in range(1, _CORRECTNESS_REFINE_ROUNDS + 1):
            if passed or baseline_wins or not ref_ok or not remaining_records:
                break
            batch = remaining_records[: self._topk]
            remaining_records = remaining_records[self._topk :]
            _LOGGER.info(
                "correctness pool exhausted at shape_m=%s, "
                "refining next %s candidates (round %s)",
                req.shape_m,
                len(batch),
                round_index,
            )
            batch_results, worker_died = self._fine_bench_records(
                batch, req.shape_m, results_by_key
            )
            if worker_died:
                ref_ok = self._set_reference(req.shape_m)
                if not ref_ok:
                    break
            passed, ref_ok, baseline_wins = self._compare_fine_candidates(
                batch_results,
                baseline_record,
                req.shape_m,
                results_by_key,
                baseline_fine_ms,
            )

        if ref_ok and not passed:
            baseline_ok = self._verify_baseline(
                baseline_record, req.shape_m, results_by_key, baseline_fine_ms
            )
            if baseline_ok is False:
                # The baseline itself fails the golden gate, so it cannot serve
                # as the pruning bar: reconsider the round-0 candidates that
                # were skipped for being slower than it.
                self._compare_fine_candidates(
                    fine_results,
                    baseline_record,
                    req.shape_m,
                    results_by_key,
                    float("inf"),
                )

        return [results_by_key[key] for key in ordered_keys]

    def close(self):
        if self._closed:
            return
        self._worker.stop()
        self._progress.close()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


__all__ = ["MeasureRequest", "ConfigTiming", "Measurer"]
