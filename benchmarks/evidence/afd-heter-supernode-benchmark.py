#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Matched preparation timings and deterministic replay checks for the AFD port.

Stage the baseline Python packages with its matching release runtime first. Both
arms use the current role-based input contract; no legacy-mode conversion is used.
The existing benchmark_afd_pipeline.py separately measures full supervised CLI time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import runpy
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CASES = ("afd:both:large", "afd:prefill:large", "afd:decode:large", "afd:dual:large", "ordinary:fixed:large")
HOST_METRICS = {"wall_time_ms", "processed_tokens_per_s", "processed_output_tokens_per_s"}


def worker(case, preparation):
    existing = runpy.run_path(str(ROOT / "scripts/benchmark_afd_roles.py"))
    if preparation:
        result = existing["measure"](case, False, True)
        result["config"] = existing["configuration"](case, False)
        return result
    from aisimulate import _runtime
    from aisimulate.compiler import prediction_to_replay_spec
    from aisimulate.config import CorePredictionConfig
    from aisimulate.runner import EngineReplayRunnerFactory
    from aisimulate.sweeper.replay import ReplayOutputRequirements

    config = existing["configuration"](case, False)
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(config))
    report = (
        EngineReplayRunnerFactory(determinism="canonical_v1")
        .create(0)
        .run(spec, output_requirements=ReplayOutputRequirements(include_raw_report=True, capture_per_request=True))
    )
    records = report.metadata["native_report"]["per_request"]
    assert len(records) == report.metrics["completed_requests"] == config["traffic"]["stop"]["requests"]
    assert all(
        r["terminal_status"] == "completed" and r["output_length"] == r["requested_output_length"] for r in records
    )
    if config["engine"]["mode"] == "disaggregated":
        assert all(r["admission_count"] == 2 and r["source_released_ms"] is not None for r in records)
    canonical = {"metrics": {k: v for k, v in report.metrics.items() if k not in HOST_METRICS}, "per_request": records}
    encoded = json.dumps(canonical, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    return {
        "semantic_sha256": hashlib.sha256(encoded).hexdigest(),
        "completed_requests": len(records),
        "total_input_tokens": report.metrics["total_input_tokens"],
        "total_output_tokens": report.metrics["total_output_tokens"],
        "binary_sha256": hashlib.sha256(Path(_runtime.__file__).read_bytes()).hexdigest(),
        "config": config,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker")
    parser.add_argument("--preparation", action="store_true")
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pairs", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=2)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker, args.preparation), allow_nan=False))
        return
    sources = {"baseline": args.baseline_source.resolve(), "candidate": ROOT / "python/aisimulate/src"}
    result = {
        "baseline_revision": "7836ce9a9b1a56cfe6f3244ae12870060022d286",
        "candidate_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "host": platform.platform(),
        "python": sys.version,
        "toolchain": subprocess.check_output(["rustc", "-Vv"], text=True),
        "scope": "compile + zero-request native preparation, including model construction and memory preflight",
        "semantic_scope": (
            "synthetic 1000-request AFD role matrix and ordinary fixed-timing control; "
            "not a long-corpus qualification"
        ),
        "excluded_metrics": sorted(HOST_METRICS),
        "pairs": args.pairs,
        "warmups_per_arm": args.warmups,
        "cpu_affinity": "not available on macOS; paired alternating order",
        "rows": [],
    }

    def invoke(case, arm, preparation):
        env = {
            **os.environ,
            "PYTHONPATH": str(sources[arm]),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", case]
        if preparation:
            command.append("--preparation")
        process = subprocess.run(command, env=env, cwd=ROOT, capture_output=True, text=True)
        if process.returncode:
            raise RuntimeError(f"{case}/{arm}: {process.stderr}")
        return json.loads(process.stdout)

    for case in CASES:
        semantic = {arm: [invoke(case, arm, False) for _ in range(2)] for arm in sources}
        digests = {item["semantic_sha256"] for items in semantic.values() for item in items}
        if len(digests) != 1:
            raise ValueError(f"{case}: nondeterminism or unexplained semantic change: {semantic}")
        row = {"case": case, "semantic_repetitions": semantic, "samples": []}
        for _ in range(args.warmups):
            for arm in sources:
                invoke(case, arm, True)
        for index in range(args.pairs):
            order = ["baseline", "candidate"] if index % 2 == 0 else ["candidate", "baseline"]
            sample = {"order": order}
            for arm in order:
                sample[arm] = invoke(case, arm, True)
            row["samples"].append(sample)
        row["timings"] = {}
        for metric in ("compile_ms", "zero_request_preparation_ms"):
            ratios = [s["candidate"][metric] / s["baseline"][metric] for s in row["samples"]]
            row["timings"][metric] = {
                "baseline_median_ms": statistics.median(s["baseline"][metric] for s in row["samples"]),
                "candidate_median_ms": statistics.median(s["candidate"][metric] for s in row["samples"]),
                "paired_ratio_median": statistics.median(ratios),
                "paired_ratio_range": [min(ratios), max(ratios)],
            }
        result["rows"].append(row)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(case, row["timings"], flush=True)


if __name__ == "__main__":
    main()
