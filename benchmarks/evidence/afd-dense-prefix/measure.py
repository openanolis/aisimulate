# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare complete CLI and native replay cost using the existing AFD harness."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_afd_pipeline import installation, measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=6)
    args = parser.parse_args()
    output = Path(__file__).with_name("performance.json")
    os.environ.update(
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", RAYON_NUM_THREADS="1", PYTHONHASHSEED="0"
    )
    baseline_env, baseline = installation(args.baseline_source.resolve())
    candidate_env, candidate = installation(ROOT / "python/aisimulate/src")
    changed = subprocess.check_output(["git", "diff", "--name-only"], cwd=ROOT, text=True).splitlines()
    source_hashes = {
        p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
        for p in changed
        if not p.startswith("benchmarks/evidence/")
    }
    result = {
        "baseline_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "candidate": "working tree at baseline_revision, identified by source hashes and release binary SHA256",
        "candidate_source_sha256": source_hashes,
        "baseline_binary": {**baseline, "size_bytes": Path(baseline["runtime"]).stat().st_size},
        "candidate_binary": {**candidate, "size_bytes": Path(candidate["runtime"]).stat().st_size},
        "build": {
            "profile": "release",
            "features": ["default", "python", "extension-module"],
            "rustflags": "-C link-arg=-undefined -C link-arg=dynamic_lookup",
            "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
        },
        "host": {"platform": platform.platform(), "python": sys.version},
        "scope": {
            "cli_ms": "fresh CLI process including supervision, construction, simulation, finalization and output",
            "native_replay_ms": (
                "Replayer::run_inner through collector finalization; excludes provider construction and output"
            ),
        },
        "warmups_per_arm": 1,
        "pairs": args.pairs,
        "rows": [],
    }
    cases = json.loads((ROOT / "crates/core/parity_tests/perfmodel/goldens/afd_execution_plan.json").read_text())[
        "cases"
    ]
    with tempfile.TemporaryDirectory(prefix="afd-dense-prefix-measure-") as temp:
        for name, seed, layout, count, concurrency in [
            ("pure_control", "qwen_tp2", "four_stage", 16, 8),
            ("dense_four_stage", "deepseek_v32_tp2", "four_stage", 128, 32),
            ("dense_two_stage", "deepseek_v32_tp2", "two_stage", 128, 32),
        ]:
            config = copy.deepcopy(cases[seed]["config"])
            config["engine"]["workers"]["aggregated"]["afd"].update(
                a_batch_size=8, num_microbatches=3, pipeline_layout=layout
            )
            config["traffic"]["load"]["concurrency"] = concurrency
            config["traffic"]["source"]["output_tokens"] = 32
            config["traffic"]["stop"] = {"requests": count}
            path = Path(temp) / f"{name}.yaml"
            path.write_text(yaml.safe_dump(config))
            row = {"name": name, "config": config, "samples": []}
            for env in [baseline_env, candidate_env]:
                measure(path, env, count)
            for pair in range(args.pairs):
                order = ["baseline", "candidate"] if pair % 2 == 0 else ["candidate", "baseline"]
                sample = {"pair": pair, "order": order}
                for arm in order:
                    sample[arm] = measure(path, baseline_env if arm == "baseline" else candidate_env, count)
                for field in ["completed_requests", "total_input_tokens", "total_output_tokens"]:
                    assert sample["baseline"][field] == sample["candidate"][field], (name, field)
                if name == "pure_control":
                    assert sample["baseline"]["simulated_duration_ms"] == sample["candidate"]["simulated_duration_ms"]
                row["samples"].append(sample)
                print(f"{name}: {pair + 1}/{args.pairs}", flush=True)
            row["timings"] = {}
            for metric in ["cli_ms", "native_replay_ms"]:
                ratios = [s["candidate"][metric] / s["baseline"][metric] for s in row["samples"]]
                row["timings"][metric] = {
                    "baseline_median": statistics.median(s["baseline"][metric] for s in row["samples"]),
                    "candidate_median": statistics.median(s["candidate"][metric] for s in row["samples"]),
                    "paired_ratio_median": statistics.median(ratios),
                    "paired_ratio_range": [min(ratios), max(ratios)],
                }
            result["rows"].append(row)
            output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(row["timings"]), flush=True)


if __name__ == "__main__":
    main()
