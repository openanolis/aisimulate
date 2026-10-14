#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Paired full-CLI/native-replay timings against a staged AFD installation.

The baseline directory must contain its Python package and matching release
extension. This measures host execution separately from simulated latency.
It does not claim replay-loop-only timing or measured serving accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "python/aisimulate/src"
CLI = ROOT / "python/aisimulate/.venv/bin/aisimulate"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def installation(source: Path) -> tuple[dict[str, str], dict]:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(source), str(SOURCE)))}
    result = subprocess.run(
        [sys.executable, "-c", "import aisimulate._runtime as r; print(r.__file__)"],
        env=env,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    runtime = Path(result.stdout.strip())
    if runtime.parent != source / "aisimulate":
        raise ValueError(f"loaded unexpected runtime: {runtime}")
    return env, {"source": str(source), "runtime": str(runtime), "runtime_sha256": sha256(runtime)}


def measure(config: Path, env: dict[str, str], expected_requests: int) -> dict:
    with tempfile.TemporaryDirectory(prefix="afd-pipeline-timing-") as directory:
        started = time.perf_counter()
        process = subprocess.run(
            [str(CLI), "predict", "--config", str(config), "--output-dir", directory, "--format", "json"],
            env=env,
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        elapsed = (time.perf_counter() - started) * 1000
        if process.returncode:
            raise RuntimeError(f"prediction failed ({process.returncode}):\n{process.stderr}")
        prediction = json.loads((Path(directory) / "prediction.json").read_text())
        summary = prediction.get("summary", prediction)
        if summary["completed_requests"] != expected_requests:
            raise ValueError("incomplete execution")
        return {
            "cli_ms": elapsed,
            "native_replay_ms": summary["wall_time_ms"],
            "completed_requests": summary["completed_requests"],
            "total_input_tokens": summary["total_input_tokens"],
            "total_output_tokens": summary["total_output_tokens"],
            "simulated_duration_ms": summary["duration_ms"],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=12)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--rows", nargs="+", help="Optional row names to measure")
    args = parser.parse_args()
    if args.pairs < 2 or args.warmups < 0:
        parser.error("pairs must be >=2 and warmups >=0")
    base_env, base_identity = installation(args.baseline_source.resolve())
    candidate_env, candidate_identity = installation(SOURCE)
    result = {
        "host": {"platform": platform.platform(), "machine": platform.machine(), "python": sys.version},
        "baseline": base_identity,
        "candidate": candidate_identity,
        "scope": {
            "cli_ms": "fresh CLI process, including startup, construction, replay and output",
            "native_replay_ms": (
                "Replayer::run_inner through collector finalization; excludes model construction and output"
            ),
        },
        "warmups_per_arm": args.warmups,
        "pairs": args.pairs,
        "rows": [],
    }
    seed = yaml.safe_load((ROOT / "crates/core/parity_tests/perfmodel/configs/afd-predict-pure.yaml").read_text())
    with tempfile.TemporaryDirectory(prefix="afd-pipeline-inputs-") as directory:
        inputs = Path(directory)
        for name, count, layout, m, trace in [
            ("small_fixed_control", 16, "four_stage", 3, False),
            ("trace_four_stage", 1024, "four_stage", 4, True),
            ("trace_two_stage", 1024, "two_stage", 2, True),
        ]:
            if args.rows and name not in args.rows:
                continue
            config = json.loads(json.dumps(seed))
            config["engine"]["context_length"] = 2048
            config["engine"]["workers"]["aggregated"]["afd"]["num_microbatches"] = m
            config["traffic"]["load"]["concurrency"] = 18 if trace else 8
            config["traffic"]["stop"] = {"requests": count}
            row = {"name": name, "requests": count, "microbatches": m, "layout": layout, "samples": []}
            if trace:
                # Trace length bounds the run; request-count stops are only
                # valid for generated synthetic traffic.
                config["traffic"]["stop"] = {}
                trace_path = inputs / f"{name}.jsonl"
                trace_path.write_text(
                    "".join(
                        json.dumps(
                            {
                                "request_id": f"request-{i}",
                                "timestamp": i,
                                "input_length": 512 + (i % 3) * 128,
                                "output_length": 96 + (i % 3) * 16,
                                "hash_ids": [10000 + 2 * i, 10001 + 2 * i],
                            }
                        )
                        + "\n"
                        for i in range(count)
                    )
                )
                row["trace_sha256"] = sha256(trace_path)
                config["traffic"]["source"] = {"type": "trace", "format": "mooncake", "paths": [str(trace_path)]}
            else:
                config["traffic"]["source"]["output_tokens"] = 8
            # Both revisions must execute the same resource layout.
            config["engine"]["workers"]["aggregated"]["afd"]["pipeline_layout"] = layout
            base_config = inputs / f"{name}-baseline.yaml"
            base_config.write_text(yaml.safe_dump(config))
            candidate_config = inputs / f"{name}-candidate.yaml"
            candidate_config.write_text(yaml.safe_dump(config))
            row["baseline_config"] = yaml.safe_load(base_config.read_text())
            row["candidate_config"] = config
            for _ in range(args.warmups):
                measure(base_config, base_env, count)
                measure(candidate_config, candidate_env, count)
            for pair in range(args.pairs):
                order = ("baseline", "candidate") if pair % 2 == 0 else ("candidate", "baseline")
                sample = {"pair": pair, "order": order}
                for arm in order:
                    sample[arm] = measure(
                        base_config if arm == "baseline" else candidate_config,
                        base_env if arm == "baseline" else candidate_env,
                        count,
                    )
                for field in ("completed_requests", "total_input_tokens", "total_output_tokens"):
                    if sample["baseline"][field] != sample["candidate"][field]:
                        raise ValueError(f"{name}: accounting differs: {field}")
                row["samples"].append(sample)
                print(f"{name}: pair {pair + 1}/{args.pairs}", flush=True)
            row["timings"] = {}
            for metric in ("cli_ms", "native_replay_ms"):
                ratios = [sample["candidate"][metric] / sample["baseline"][metric] for sample in row["samples"]]
                row["timings"][metric] = {
                    "baseline_median": statistics.median(sample["baseline"][metric] for sample in row["samples"]),
                    "candidate_median": statistics.median(sample["candidate"][metric] for sample in row["samples"]),
                    "paired_ratio_median": statistics.median(ratios),
                    "paired_ratio_range": [min(ratios), max(ratios)],
                }
            result["rows"].append(row)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps({"name": name, "timings": row["timings"]}), flush=True)


if __name__ == "__main__":
    main()
