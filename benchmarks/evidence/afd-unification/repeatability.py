# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Repeat full native reports in fresh processes with canonical request identity.

This checks JSON binding repeatability, including per-request and power evidence.
The Rust pressure regression separately uses CanonicalReplayRecord and captures
execution-owned pressure evidence unavailable in the compact JSON binding.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
BASE = Path("/private/tmp/afd-unification-baseline")
GOLDENS = ROOT / "crates/core/parity_tests/perfmodel/goldens"


def worker(raw):
    from aisimulate import _runtime
    from aisimulate.compiler import prediction_to_replay_spec
    from aisimulate.config import CorePredictionConfig
    from aisimulate.runner import EngineReplayRunnerFactory
    from aisimulate.sweeper.replay import ReplayOutputRequirements

    runner = EngineReplayRunnerFactory(determinism="canonical_v1").create(0)
    try:
        result = runner.run(
            prediction_to_replay_spec(CorePredictionConfig.model_validate(raw)),
            output_requirements=ReplayOutputRequirements(include_raw_report=True, capture_per_request=True),
        )
    finally:
        runner.close()
    report = result.metadata["native_report"]
    for key in ["wall_time_ms", "processed_tokens_per_s", "processed_output_tokens_per_s"]:
        report.pop(key, None)
    report["per_request"].sort(key=lambda record: record["uuid"])
    encoded = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {
        "binary_sha256": hashlib.sha256(Path(_runtime.__file__).read_bytes()).hexdigest(),
        "report_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        "counts": {k: result.metrics[k] for k in ["completed_requests", "total_input_tokens", "total_output_tokens"]},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(json.load(sys.stdin))))
        return
    sys.path.insert(0, str(GOLDENS.parent))
    from test_afd_execution_plan_parity import case_config

    plans = json.loads((GOLDENS / "afd_execution_plan.json").read_text())["cases"]
    e2e = json.loads((GOLDENS / "afd_e2e.json").read_text())["cases"]
    hashes = {
        arm: hashlib.sha256((source / "python/aisimulate/src/aisimulate/_runtime.abi3.so").read_bytes()).hexdigest()
        for arm, source in [("baseline", BASE), ("candidate", ROOT)]
    }
    results = {"determinism": "canonical_v1", "binary_sha256": hashes, "cases": {}}
    with tempfile.TemporaryDirectory(prefix="afd-repeatability-") as temp:
        directory = Path(temp)
        configs = {name: case_config(case, directory) for name, case in plans.items()}
        configs.update(
            {
                name: yaml.safe_load((ROOT / case["config"]).read_text())
                for name, case in e2e.items()
                if not case.get("historical")
            }
        )
        for name, config in configs.items():
            results["cases"][name] = {}
            for arm, source in [("baseline", BASE), ("candidate", ROOT)]:
                rows = []
                for _ in range(2):
                    env = {
                        **os.environ,
                        "PYTHONPATH": str(source / "python/aisimulate/src"),
                        "PYTHONHASHSEED": "0",
                        "RAYON_NUM_THREADS": "1",
                        "OMP_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                    }
                    p = subprocess.run(
                        [sys.executable, str(Path(__file__).resolve()), "--worker"],
                        input=json.dumps(config),
                        text=True,
                        capture_output=True,
                        env=env,
                        check=True,
                    )
                    row = json.loads(p.stdout)
                    assert row["binary_sha256"] == hashes[arm]
                    rows.append(row)
                assert rows[0] == rows[1], (name, arm, rows)
                results["cases"][name][arm] = rows
            assert results["cases"][name]["baseline"][0]["counts"] == results["cases"][name]["candidate"][0]["counts"]
            print(name, "stable", flush=True)
    (Path(__file__).parent / "repeatability.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
