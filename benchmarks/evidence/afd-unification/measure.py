# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Paired release measurements: pool preparation, native replay, and supervised CLI."""

import argparse
import copy
import hashlib
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
BASE = Path("/private/tmp/afd-unification-baseline")
OUTPUT = Path(__file__).parent
ENV = {
    **os.environ,
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
}


def worker(request):
    from aisimulate import _runtime
    from aisimulate.compiler import prediction_to_replay_spec
    from aisimulate.config import CorePredictionConfig
    from aisimulate.runner import EngineReplayRunnerFactory
    from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

    fingerprint = hashlib.sha256(Path(_runtime.__file__).read_bytes()).hexdigest()
    assert fingerprint == request["binary_sha256"]
    raw = request["config"]
    if request.get("pool_only"):
        role = request["pool_only"]
        tp = raw["engine"]["workers"]["aggregated"]["afd"]["tp_a"]
        config = dict(
            model=raw["engine"]["model"],
            system="h200_sxm",
            backend="sglang",
            backend_version="0.5.14",
            worker_type="aggregated",
            tp=tp if role == "a" else 8,
            moe_tp_size=tp if role == "a" else 1,
            moe_ep_size=1 if role == "a" else 8,
            estimation_mode="op_level",
        )
        if request["arm"] == "candidate" or "DeepSeek" in raw["engine"]["model"]:
            config.update(afd_pool_role=role, afd_routing_location="a")
        start = time.perf_counter_ns()
        model = RustForwardPassPerfModel.best_available(ForwardPassPerfModelConfig(**config))
        elapsed = (time.perf_counter_ns() - start) / 1e6
        model.close()
        return {
            "binary_sha256": fingerprint,
            "cold_pool_construction_ms": elapsed,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }
    start = time.perf_counter_ns()
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
    python_preparation_ms = (time.perf_counter_ns() - start) / 1e6
    runner = EngineReplayRunnerFactory(determinism="canonical_v1").create(0)
    try:

        def call():
            start = time.perf_counter_ns()
            result = runner.run(spec)
            elapsed = (time.perf_counter_ns() - start) / 1e6
            expected = raw["traffic"]["stop"]["requests"]
            assert result.metrics["completed_requests"] == expected
            assert result.metrics["total_output_tokens"] == expected * raw["traffic"]["source"]["output_tokens"]
            return elapsed, result

        first_ms, result = call()  # Includes native pool construction and evidence aggregation.
        for _ in range(10):
            call()
        samples = [call()[0] for _ in range(30)]
    finally:
        runner.close()
    pools = {}
    for role in ["a", "f"]:
        config = dict(
            model=raw["engine"]["model"],
            system="h200_sxm",
            backend="sglang",
            backend_version="0.5.14",
            worker_type="aggregated",
            tp=raw["engine"]["workers"]["aggregated"]["afd"]["tp_a"] if role == "a" else 8,
            moe_tp_size=raw["engine"]["workers"]["aggregated"]["afd"]["tp_a"] if role == "a" else 1,
            moe_ep_size=1 if role == "a" else 8,
            estimation_mode="op_level",
        )
        # Match the actual pool construction used by each revision, not a surrogate graph.
        if request["arm"] == "candidate" or "DeepSeek" in raw["engine"]["model"]:
            config.update(afd_pool_role=role, afd_routing_location="a")
        start = time.perf_counter_ns()
        model = RustForwardPassPerfModel.best_available(ForwardPassPerfModelConfig(**config))
        pools[role] = (time.perf_counter_ns() - start) / 1e6
        model.close()
    return {
        "binary_sha256": fingerprint,
        "python_preparation_ms": python_preparation_ms,
        "cached_pool_construction_ms": pools,
        "first_native_call_ms": first_ms,
        "warm_native_call_ms": samples,
        "warm_median_ms": statistics.median(samples),
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "counts": {k: result.metrics[k] for k in ["completed_requests", "total_input_tokens", "total_output_tokens"]},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(json.load(sys.stdin))))
        return
    fixture = json.loads((ROOT / "crates/core/parity_tests/perfmodel/goldens/afd_execution_plan.json").read_text())
    qwen = copy.deepcopy(fixture["cases"]["qwen_tp2"]["config"])
    qwen["engine"]["workers"]["aggregated"]["afd"]["tp_a"] = 1
    deepseek = fixture["cases"]["deepseek_v32_tp2"]["config"]
    cases = {"qwen_tp1_control": qwen, "deepseek_tp2": deepseek}
    hashes = {
        arm: hashlib.sha256((src / "python/aisimulate/src/aisimulate/_runtime.abi3.so").read_bytes()).hexdigest()
        for arm, src in [("baseline", BASE), ("candidate", ROOT)]
    }
    result = {
        "baseline_revision": "23d21fa4f62ff47e742c48c26d3ed202f97308d2",
        "candidate_revision": "23d21fa4+working-tree",
        "platform": platform.platform(),
        "cpu_affinity": "unavailable on macOS",
        "features": ["extension-module"],
        "warmup": 10,
        "iterations": 30,
        "rounds": 3,
        "rows": {},
    }
    with tempfile.TemporaryDirectory(prefix="afd-measure-") as temp:
        temp = Path(temp)
        for name, raw in cases.items():
            row = {"config": raw, "rounds": [], "cli": []}
            for r in range(3):
                measurements = {}
                for arm in ["baseline", "candidate"] if r % 2 == 0 else ["candidate", "baseline"]:
                    source = BASE if arm == "baseline" else ROOT
                    request = {"arm": arm, "config": raw, "binary_sha256": hashes[arm]}
                    p = subprocess.run(
                        [sys.executable, str(Path(__file__).resolve()), "--worker"],
                        input=json.dumps(request),
                        text=True,
                        capture_output=True,
                        check=True,
                        env={**ENV, "PYTHONPATH": str(source / "python/aisimulate/src")},
                    )
                    measurements[arm] = json.loads(p.stdout)
                    measurements[arm]["cold_pools"] = {}
                    for role in ["a", "f"]:
                        cold = subprocess.run(
                            [sys.executable, str(Path(__file__).resolve()), "--worker"],
                            input=json.dumps({**request, "pool_only": role}),
                            text=True,
                            capture_output=True,
                            check=True,
                            env={**ENV, "PYTHONPATH": str(source / "python/aisimulate/src")},
                        )
                        measurements[arm]["cold_pools"][role] = json.loads(cold.stdout)

                assert measurements["baseline"]["counts"] == measurements["candidate"]["counts"]
                row["rounds"].append(measurements)
                print(name, "native round", r + 1, flush=True)
            # Full supervised CLI: small control and 1,000-request scaling row, 3 paired rounds.
            for count in [4, 1000]:
                config = copy.deepcopy(raw)
                config["traffic"]["stop"]["requests"] = count
                path = temp / (name + str(count) + ".yaml")
                path.write_text(yaml.safe_dump(config))
                for r in range(3):
                    pair = {"requests": count, "round": r}
                    for arm in ["baseline", "candidate"] if r % 2 == 0 else ["candidate", "baseline"]:
                        source = BASE if arm == "baseline" else ROOT
                        output = temp / f"{name}-{count}-{r}-{arm}"
                        start = time.perf_counter_ns()
                        subprocess.run(
                            [
                                str(ROOT / "python/aisimulate/.venv/bin/aisimulate"),
                                "predict",
                                "--stack",
                                "engine",
                                "--config",
                                str(path),
                                "--output-dir",
                                str(output),
                                "--format",
                                "json",
                            ],
                            capture_output=True,
                            text=True,
                            check=True,
                            env={**ENV, "PYTHONPATH": str(source / "python/aisimulate/src")},
                        )
                        elapsed = (time.perf_counter_ns() - start) / 1e6
                        report = json.loads((output / "prediction.json").read_text())
                        summary = report.get("summary", report)
                        assert summary["completed_requests"] == summary["num_requests"] == count
                        pair[arm] = {
                            "cli_ms": elapsed,
                            "wall_time_ms": summary["wall_time_ms"],
                            "counts": {
                                k: summary[k]
                                for k in ["completed_requests", "total_input_tokens", "total_output_tokens"]
                            },
                        }
                    assert pair["baseline"]["counts"] == pair["candidate"]["counts"]
                    row["cli"].append(pair)
                    print(name, count, "CLI round", r + 1, flush=True)
            result["rows"][name] = row
            (OUTPUT / "performance.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
