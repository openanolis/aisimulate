# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare pinned baseline/current CLI results and refresh explained AFD goldens."""

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
CLI = ROOT / "python/aisimulate/.venv/bin/aisimulate"
GOLDENS = ROOT / "crates/core/parity_tests/perfmodel/goldens"
sys.path.insert(0, str(GOLDENS.parent))
from test_afd_execution_plan_parity import case_config

FIELDS = ("completed_requests", "total_input_tokens", "total_output_tokens")
ENV = {
    **os.environ,
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
}


def run(source, config, output):
    env = {**ENV, "PYTHONPATH": str(source / "python/aisimulate/src")}
    subprocess.run(
        [
            str(CLI),
            "predict",
            "--stack",
            "engine",
            "--config",
            str(config),
            "--output-dir",
            str(output),
            "--format",
            "json",
        ],
        env=env,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads((output / "prediction.json").read_text())
    return report.get("summary", report)


def main():
    execution = json.loads((GOLDENS / "afd_execution_plan.json").read_text())
    e2e = json.loads((GOLDENS / "afd_e2e.json").read_text())
    old_goldens = BASE / "crates/core/parity_tests/perfmodel/goldens"
    old_execution = json.loads((old_goldens / "afd_execution_plan.json").read_text())
    old_e2e = json.loads((old_goldens / "afd_e2e.json").read_text())
    evidence = {
        "baseline": "23d21fa4f62ff47e742c48c26d3ed202f97308d2",
        "binary_sha256": {
            arm: hashlib.sha256((src / "python/aisimulate/src/aisimulate/_runtime.abi3.so").read_bytes()).hexdigest()
            for arm, src in [("baseline", BASE), ("candidate", ROOT)]
        },
        "cases": {},
    }
    with tempfile.TemporaryDirectory(prefix="afd-unification-") as temp:
        directory = Path(temp)
        rows = [("plan", name, row) for name, row in execution["cases"].items()]
        rows += [("e2e", name, row) for name, row in e2e["cases"].items() if not row.get("historical", False)]
        for kind, name, row in rows:
            work = directory / name
            work.mkdir()
            if kind == "plan":
                raw = case_config(row, work)
                path = work / "config.yaml"
                path.write_text(yaml.safe_dump(raw))
            else:
                path = ROOT / row["config"]
            before = run(BASE, path, work / "baseline")
            after = run(ROOT, path, work / "candidate")
            assert all(before[k] == after[k] for k in FIELDS), name
            expected = row["expected"] if kind == "plan" else row["summary"]
            old_expected = (
                old_execution["cases"][name]["expected"] if kind == "plan" else old_e2e["cases"][name]["summary"]
            )
            changed = {
                key: {"before": before.get(key), "after": after.get(key)}
                for key in expected
                if key not in ("wall_time_ms", "processed_tokens_per_s", "processed_output_tokens_per_s")
                and before.get(key) != after.get(key)
            }
            evidence["cases"][name] = {
                "kind": kind,
                "changes": changed,
                "counts": {k: after[k] for k in FIELDS},
                "baseline_matches_old_expected": all(
                    not isinstance(value, (int, float))
                    or abs(before.get(key, 0) - value) <= max(1e-9, abs(value) * 1e-6)
                    for key, value in old_expected.items()
                    if key not in ("wall_time_ms", "processed_tokens_per_s", "processed_output_tokens_per_s")
                ),
                "attribution": (
                    "Actual-batch prefill in all identities; pure TP1 now uses endpoint work, mapped NIC load "
                    "and node-partial combine. Decode keeps its frozen/instantaneous identity; timing shifts "
                    "propagate to queueing and completion metrics."
                ),
            }
            if kind == "plan":
                row["expected"] = {key: after[key] for key in expected}
            else:
                row["summary"] = {
                    key: value
                    for key, value in after.items()
                    if key not in ("wall_time_ms", "processed_tokens_per_s", "processed_output_tokens_per_s")
                }
            row["modeling_change"] = evidence["cases"][name]["attribution"]
            print(name, len(changed), "changed fields", flush=True)
    execution["description"] = "Source-ordered AFD expectations, including pure TP1 and actual-batch prefill."
    (Path(__file__).parent / "semantic-deltas.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    for name, value in [("afd_execution_plan.json", execution), ("afd_e2e.json", e2e)]:
        (GOLDENS / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
