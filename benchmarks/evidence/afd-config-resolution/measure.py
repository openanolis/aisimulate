# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure one pinned release binary; run alternating baseline/candidate processes.

Usage from the repository root: python measure.py BINARY [--legacy-input]
The baseline needs the former redundant timing fields reconstructed from the
same rank-owned inputs. Both arms execute the same canonical workload.
"""

import hashlib
import importlib.util
import json
import resource
import statistics
import sys
import time
from pathlib import Path

binary = sys.argv[1]
spec = importlib.util.spec_from_file_location("aisimulate._runtime", binary)
module = importlib.util.module_from_spec(spec)
sys.modules["aisimulate._runtime"] = module
spec.loader.exec_module(module)
import aisimulate

aisimulate._runtime = module
import runpy

from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config import CorePredictionConfig
from aisimulate.runner import _materialize_engine_execution_spec

fixtures = runpy.run_path("tests/sweeper/test_afd_search_integration.py")
results = []
for name, count in [("control", 4), ("replay", 1000)]:
    raw = fixtures["prediction"]("aggregated", ("aggregated",))
    raw["traffic"]["source"].update(input_tokens=512, output_tokens=32)
    raw["traffic"]["stop"]["requests"] = count
    raw["traffic"]["load"]["concurrency"] = 16
    wire = _materialize_engine_execution_spec(
        prediction_to_replay_spec(CorePredictionConfig.model_validate(raw)),
        trace_block_size=64,
        record_per_request=True,
    )
    wire["determinism"] = "canonical_v1"
    if "--legacy-input" in sys.argv[2:]:
        rank = wire["spec"]["engine"]["rank"]
        timing = rank["timing_model"]["config"]
        timing.update(
            **rank["afd"], backend=rank["backend"], nextn=rank.get("aic_nextn", 0), max_seq_len=rank["max_model_len"]
        )
    payload = json.dumps(wire)
    start = time.perf_counter_ns()
    encoded = module.run_replay_json(payload)
    first = (time.perf_counter_ns() - start) / 1e6
    report = json.loads(encoded)
    assert report["completed_requests"] == count
    for _ in range(3):
        module.run_replay_json(payload)
    samples = []
    for _ in range(15):
        start = time.perf_counter_ns()
        result = module.run_replay_json(payload)
        samples.append((time.perf_counter_ns() - start) / 1e6)
    clean = {
        k: v
        for k, v in report.items()
        if k not in {"wall_time_ms", "processed_tokens_per_s", "processed_output_tokens_per_s"}
    }
    results.append(
        {
            "name": name,
            "requests": count,
            "first_native_ms": first,
            "warm_native_ms": samples,
            "median_warm_native_ms": statistics.median(samples),
            "canonical_sha256": hashlib.sha256(json.dumps(clean, sort_keys=True).encode()).hexdigest(),
            "duration_ms": report["duration_ms"],
            "total_input_tokens": report["total_input_tokens"],
            "total_output_tokens": report["total_output_tokens"],
        }
    )
print(
    json.dumps(
        {
            "binary_sha256": hashlib.sha256(Path(binary).read_bytes()).hexdigest(),
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "cases": results,
        }
    )
)
