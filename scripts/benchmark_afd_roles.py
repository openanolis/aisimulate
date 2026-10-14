#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Paired fresh-process preparation and complete engine-replay measurements."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

HOST_METRICS = {"wall_time_ms", "processed_tokens_per_s", "processed_output_tokens_per_s"}


def configuration(case, legacy):
    mode, phase, size = case.split(":")
    large = size == "large"
    topology = {"n_a_nodes": 1, "n_f_nodes": 1, "tp_a": 1, "a_batch_size": 18, "num_microbatches": 3}
    mono = {"kv_cache": {"capacity": {"type": "fixed", "blocks": 65536}}}
    engine = {
        "model": "Qwen/Qwen3-30B-A3B",
        "hardware": "h200_sxm",
        "backend": "sglang",
        "backend_version": "0.5.14",
        "context_length": 2048,
    }
    if mode == "ordinary":
        engine.update(mode="aggregated", workers={"aggregated": mono})
        if phase == "fixed":
            mono["timing"] = {"type": "fixed", "prefill_ms": 1, "decode_ms": 1}
    elif legacy:
        engine.update(mode="afd", afd={**topology, "phase": phase, "combined_with_pd": phase != "both"})
        if phase != "both":
            engine["workers"] = {"decode" if phase == "prefill" else "prefill": mono}
    else:
        worker = {"type": "afd", "afd": topology}
        if phase == "both":
            engine.update(mode="aggregated", workers={"aggregated": worker})
        else:
            workers = {role: worker if role == phase or phase == "dual" else mono for role in ("prefill", "decode")}
            engine.update(mode="disaggregated", workers=workers)
    return {
        "engine": engine,
        "traffic": {
            "source": {
                "type": "synthetic",
                "input_tokens": 1024 if large else 32,
                "output_tokens": 128 if large else 8,
            },
            "load": {"type": "concurrency", "concurrency": 32 if large else 4},
            "stop": {"requests": 1000 if large else 32},
        },
    }


def measure(case, legacy, preparation_only=False):
    from aisimulate import _runtime
    from aisimulate.compiler import prediction_to_replay_spec
    from aisimulate.config import CorePredictionConfig, CoreRecommendationConfig
    from aisimulate.recommend import recommendation_to_sweeper
    from aisimulate.runner import EngineReplayRunnerFactory, _materialize_engine_execution_spec
    from aisimulate.sweeper.replay import ReplayOutputRequirements
    from aisimulate.sweeper.search_space import enumerate_branches

    raw = configuration(case, legacy)
    if case.startswith("search:"):
        raw = configuration("afd:decode:small", legacy)
        worker = raw["engine"]["afd"] if legacy else raw["engine"]["workers"]["decode"]["afd"]
        del worker["n_a_nodes"], worker["n_f_nodes"]
        mono = raw["engine"]["workers"]["prefill"]
        mono["kv_cache"] = {"capacity": {"type": "default"}}
        mono["parallelism"] = {
            "preset": [
                {"replicas": 1, "tensor": 1, "pipeline": 1, "attention_data": 1, "moe_tensor": 1, "moe_expert": 1}
            ]
        }
        raw["optimization"] = {"constraints": {"max_candidate_gpus": 64}}
        config = CoreRecommendationConfig.model_validate(raw)
        times, counts = [], []
        for _ in range(6):
            start = time.perf_counter()
            smart = recommendation_to_sweeper(config)
            branches = enumerate_branches(smart, max_seq_len=2048)
            times.append((time.perf_counter() - start) * 1000)
            counts.append(sum(len(branch.parallel_configs) for branch in branches))
        return {
            "candidate_counts": counts,
            "search_cold_ms": times[0],
            "search_warm_ms": times[1:],
            "binary_sha256": hashlib.sha256(Path(_runtime.__file__).read_bytes()).hexdigest(),
        }
    config = CorePredictionConfig.model_validate(raw)
    started = time.perf_counter()
    spec = prediction_to_replay_spec(config)
    compile_ms = (time.perf_counter() - started) * 1000
    if preparation_only:
        started = time.perf_counter()
        execution = _materialize_engine_execution_spec(spec, trace_block_size=512, record_per_request=False)
        execution = execution.get("spec", execution)
        execution["requests"] = []
        payload = json.dumps(execution, allow_nan=False)
        native = json.loads(_runtime.run_replay_json(payload))
        preparation_ms = (time.perf_counter() - started) * 1000
        return {
            "compile_ms": compile_ms,
            "zero_request_preparation_ms": preparation_ms,
            "native_wall_ms": native["wall_time_ms"],
            "prepared_role_counts": 1 if raw["engine"]["mode"] == "aggregated" or case.split(":")[1] == "both" else 2,
            "binary_sha256": hashlib.sha256(Path(_runtime.__file__).read_bytes()).hexdigest(),
        }
    runner = EngineReplayRunnerFactory().create(0)
    started = time.perf_counter()
    report = runner.run(
        spec, output_requirements=ReplayOutputRequirements(include_raw_report=True, capture_per_request=True)
    )
    total_ms = (time.perf_counter() - started) * 1000
    runner.close()
    native = report.metadata["native_report"]
    records = native["per_request"]
    assert all(
        r["terminal_status"] == "completed" and r["output_length"] == r["requested_output_length"] for r in records
    )
    metrics = {k: v for k, v in report.metrics.items() if k not in HOST_METRICS}
    return {
        "compile_ms": compile_ms,
        "complete_runner_ms": total_ms,
        "native_wall_ms": report.metrics["wall_time_ms"],
        "metrics": metrics,
        "binary_sha256": hashlib.sha256(Path(_runtime.__file__).read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker")
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--preparation-only", action="store_true")
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--candidate-source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pairs", type=int, default=5)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(measure(args.worker, args.legacy, args.preparation_only), sort_keys=True, allow_nan=False))
        return
    cases = [f"afd:{phase}:{size}" for phase in ("both", "prefill", "decode") for size in ("small", "large")]
    cases += ["ordinary:fixed:large", "ordinary:op_level:large"]
    if args.preparation_only:
        cases = [case for case in cases if case.endswith(":large")]
    else:
        cases.append("search:decode:small")
    result = {
        "host": platform.platform(),
        "python": platform.python_version(),
        "pairs": args.pairs,
        "preparation_only": args.preparation_only,
        "cpu_affinity": "unavailable on macOS",
        "rows": {},
    }
    for case in cases:
        pairs = []
        for index in range(args.pairs):
            pair = {}
            for side in ("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline"):
                source = args.baseline_source if side == "baseline" else args.candidate_source
                env = {
                    **os.environ,
                    "PYTHONPATH": str(source),
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                }
                cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", case]
                if side == "baseline":
                    cmd.append("--legacy")
                if args.preparation_only:
                    cmd.append("--preparation-only")
                started = time.perf_counter()
                child = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
                row = json.loads(child.stdout)
                row["process_elapsed_ms"] = (time.perf_counter() - started) * 1000
                pair[side] = row
            if "metrics" in pair["baseline"]:
                base_metrics, candidate_metrics = pair["baseline"]["metrics"], pair["candidate"]["metrics"]
                assert base_metrics.keys() == candidate_metrics.keys(), case
                deltas = {}
                for name, value in base_metrics.items():
                    candidate_value = candidate_metrics[name]
                    if isinstance(value, float) and isinstance(candidate_value, float):
                        assert math.isclose(value, candidate_value, rel_tol=1e-9, abs_tol=1e-9), (
                            case,
                            name,
                            value,
                            candidate_value,
                        )
                        if value != candidate_value:
                            deltas[name] = abs(value - candidate_value)
                    else:
                        assert value == candidate_value, (case, name)
                pair["simulation_rounding_deltas"] = deltas
            elif args.preparation_only:
                assert pair["baseline"]["prepared_role_counts"] == pair["candidate"]["prepared_role_counts"], case
            else:
                assert pair["baseline"]["candidate_counts"] == pair["candidate"]["candidate_counts"], case
            pairs.append(pair)
        result["rows"][case] = pairs
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
        print(case, "PASS", flush=True)


if __name__ == "__main__":
    main()
