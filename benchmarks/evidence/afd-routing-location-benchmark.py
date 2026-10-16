#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Matched release-build forward and native AFD CLI measurements.

Baseline source must contain the baseline Python packages and rebuilt extension.
The controller alternates revision order and checkpoints completed pairs.
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

ROOT = Path(__file__).resolve().parents[2]
MODELS = ("Qwen/Qwen3-32B", "Qwen/Qwen3-30B-A3B")


def run(command, env):
    completed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    return json.loads(completed.stdout)


def measure_model(model):
    import aisimulate._runtime as runtime
    from aisimulate_core._native import RustForwardPassPerfModel

    config = dict(
        model=model,
        system="h200_sxm",
        backend="sglang",
        backend_version="0.5.14",
        worker_type="aggregated",
        tp=8,
        moe_tp_size=8,
        moe_ep_size=1,
        estimation_mode="op_level",
        fallback_policy="deny",
        database_mode="SILICON",
    )
    if hasattr(runtime, "AFD_DEFAULT_ROUTING_LOCATION"):
        config["afd_routing_location"] = "a"
    started = time.perf_counter_ns()
    forward = RustForwardPassPerfModel.best_available(json.dumps(config))
    build_ms = (time.perf_counter_ns() - started) / 1e6
    started = time.perf_counter_ns()
    predicted_ms = forward.predict_prefill_latency(4, 128)
    first_ms = (time.perf_counter_ns() - started) / 1e6
    for _ in range(10):
        forward.predict_prefill_latency(4, 128)
    samples = []
    for _ in range(100):
        started = time.perf_counter_ns()
        forward.predict_prefill_latency(4, 128)
        samples.append((time.perf_counter_ns() - started) / 1e6)

    recipe = yaml.safe_load((ROOT / "crates/core/parity_tests/perfmodel/configs/afd-predict-pure.yaml").read_text())
    recipe["engine"].update(model=model, context_length=4096)
    recipe["engine"]["afd"].update(tp_a=4, a_batch_size=4, num_microbatches=2)
    recipe["traffic"]["source"].update(input_tokens=128, output_tokens=16)
    recipe["traffic"]["load"]["concurrency"] = 2
    recipe["traffic"]["stop"]["requests"] = 8
    with tempfile.TemporaryDirectory(prefix="afd-routing-measure-") as directory:
        path = Path(directory) / "config.yaml"
        path.write_text(yaml.safe_dump(recipe))
        started = time.perf_counter_ns()
        subprocess.run(
            [
                sys.executable,
                "-m",
                "aisimulate",
                "predict",
                "--stack",
                "engine",
                "--config",
                str(path),
                "--output-dir",
                str(Path(directory) / "out"),
                "--format",
                "json",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        cli_ms = (time.perf_counter_ns() - started) / 1e6
        prediction = json.loads((Path(directory) / "out/prediction.json").read_text())
    return dict(
        model=model,
        extension=runtime.__file__,
        extension_sha256=hashlib.sha256(Path(runtime.__file__).read_bytes()).hexdigest(),
        forward_build_ms=build_ms,
        forward_first_ms=first_ms,
        forward_warm_median_ms=statistics.median(samples),
        forward_prediction_ms=predicted_ms,
        cli_wall_ms=cli_ms,
        prediction=prediction,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure-model", choices=MODELS)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    if args.measure_model:
        print(json.dumps(measure_model(args.measure_model)))
        return
    if args.baseline_source is None or args.output is None:
        parser.error("--baseline-source and --output are required")
    base_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    source_diff = subprocess.check_output(
        ["git", "diff", "--binary", "HEAD", "--", "crates/core/src", "python/aisimulate/src"], cwd=ROOT
    )
    identity = dict(base_revision=base_revision, candidate_source_diff_sha256=hashlib.sha256(source_diff).hexdigest())
    revisions = dict(
        base=base_revision, candidate=base_revision + "+worktree:" + identity["candidate_source_diff_sha256"]
    )
    sources = dict(base=args.baseline_source.resolve(), candidate=ROOT / "python/aisimulate/src")
    report = dict(
        schema_version=1,
        identity=identity,
        host=platform.platform(),
        python=platform.python_version(),
        build=(
            "cargo rustc --release --features extension-module --offline -- "
            "-C link-arg=-undefined -C link-arg=dynamic_lookup"
        ),
        scopes=dict(
            forward_build=(
                "Canonical best_available construction, including compile and data loading; Python startup excluded."
            ),
            forward_first=(
                "First canonical prefill query after construction; includes first-use native/table preparation."
            ),
            forward_warm="100 cached canonical prefill queries after 10 warmups; Python wrapper included.",
            gate=(
                "Repository forward_perf_gate worker protocol 2: setup, "
                "primed unseen query, 10 warmups, 100 warm queries."
            ),
            cli=(
                "Full supervised predict --stack engine command including startup, "
                "model construction, replay and output serialization."
            ),
            native_wall=(
                "prediction.wall_time_ms: native run through collector finalization; "
                "excludes timing-model construction."
            ),
        ),
        rounds=[],
    )
    cases = [
        dict(
            case_id=f"{model.split('/')[-1]}-{phase}",
            model_id=model,
            model_path=model,
            system_name="h200_sxm",
            backend_name="sglang",
            backend_version="0.5.14",
            database_mode="SILICON",
            phase=phase,
            batch_size=4,
            isl=128,
            osl=16,
            prefix=0,
            stride=32,
            tp_size=8,
            pp_size=1,
            attention_dp_size=1,
            moe_tp_size=8,
            moe_ep_size=1,
        )
        for model in MODELS
        for phase in ("context", "generation")
    ]
    with tempfile.TemporaryDirectory(prefix="afd-routing-gate-") as directory:
        for round_index in range(args.rounds):
            pair = {}
            order = ("base", "candidate") if round_index % 2 == 0 else ("candidate", "base")
            for revision in order:
                env = {**os.environ, "PYTHONPATH": str(sources[revision])}
                request = Path(directory) / "request.json"
                request.write_text(
                    json.dumps(
                        dict(protocol_version=2, revision=revisions[revision], cases=cases, warmup=10, iterations=100)
                    )
                )
                gate = run(
                    [
                        sys.executable,
                        str(ROOT / "python/aisimulate/tools/forward_perf_gate/worker.py"),
                        "--request",
                        str(request),
                    ],
                    env,
                )
                if any(result["status"] != "OK" for result in gate["results"]):
                    raise RuntimeError(json.dumps(gate))
                models = [
                    run([sys.executable, str(Path(__file__).resolve()), "--measure-model", model], env)
                    for model in MODELS
                ]
                pair[revision] = dict(gate=gate, models=models)
            report["rounds"].append(pair)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(f"completed pair {round_index + 1}/{args.rounds}", flush=True)


if __name__ == "__main__":
    main()
