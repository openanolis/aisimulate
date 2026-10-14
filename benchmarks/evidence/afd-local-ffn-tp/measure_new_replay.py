# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Complete candidate-only CLI/native replay scaling on new real model layouts."""

import hashlib
import json
import os
import platform
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
CASES = json.loads((ROOT / "crates/core/parity_tests/perfmodel/goldens/afd_execution_plan.json").read_text())["cases"]
CLI = ROOT / "python/aisimulate/.venv/bin/aisimulate"
result = {
    "host": platform.platform(),
    "scope": "fresh full CLI + native replay interval; new layouts have no supported baseline",
    "samples_per_row": 3,
    "rows": [],
}
with tempfile.TemporaryDirectory(prefix="afd-new-scale-") as directory:
    temp = Path(directory)
    for name in ["qwen_tp2", "deepseek_v32_tp4_trace", "mimo_tp4_trace"]:
        for count in [16, 1024]:
            trace = temp / "requests.jsonl"
            trace.write_text(
                "".join(
                    json.dumps(
                        {
                            "request_id": str(i),
                            "timestamp": i,
                            "input_length": 512 + (i % 3) * 128,
                            "output_length": 32 + (i % 3) * 8,
                            "hash_ids": [42, 43, 10000 + i],
                        }
                    )
                    + "\n"
                    for i in range(count)
                )
            )
            raw = json.loads(json.dumps(CASES[name]["config"]))
            raw["traffic"]["source"] = {"type": "trace", "format": "mooncake", "paths": [str(trace)]}
            raw["traffic"]["stop"] = {}
            raw["traffic"]["load"]["concurrency"] = 32
            config = temp / "config.yaml"
            config.write_text(yaml.safe_dump(raw))
            row = {
                "name": name,
                "requests": count,
                "config": raw,
                "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
                "samples": [],
            }
            for _ in range(3):
                started = time.perf_counter()
                p = subprocess.run(
                    [
                        str(CLI),
                        "predict",
                        "--stack",
                        "engine",
                        "--config",
                        str(config),
                        "--output-dir",
                        str(temp / "output"),
                        "--overwrite",
                        "--format",
                        "json",
                    ],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    check=True,
                    env={
                        **os.environ,
                        "OMP_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                        "RAYON_NUM_THREADS": "1",
                        "PYTHONHASHSEED": "0",
                    },
                )
                cli_ms = (time.perf_counter() - started) * 1000
                report = json.loads((temp / "output/prediction.json").read_text())
                s = report.get("summary", report)
                assert s["completed_requests"] == count
                assert s["total_input_tokens"] == sum(512 + (i % 3) * 128 for i in range(count))
                assert s["total_output_tokens"] == sum(32 + (i % 3) * 8 for i in range(count))
                row["samples"].append(
                    {
                        "cli_ms": cli_ms,
                        **{
                            k: s[k]
                            for k in [
                                "wall_time_ms",
                                "duration_ms",
                                "completed_requests",
                                "total_input_tokens",
                                "total_output_tokens",
                                "committed_prefill_tokens",
                                "power_coverage",
                            ]
                        },
                    }
                )
            row["median_native_replay_ms"] = statistics.median(s["wall_time_ms"] for s in row["samples"])
            row["median_cli_ms"] = statistics.median(s["cli_ms"] for s in row["samples"])
            result["rows"].append(row)
            Path(__file__).with_name("new-replay-scaling.json").write_text(json.dumps(result, indent=2) + "\n")
            print(name, count, round(row["median_native_replay_ms"], 2), round(row["median_cli_ms"], 2), flush=True)
