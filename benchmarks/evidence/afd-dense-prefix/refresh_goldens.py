# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Refresh only the five AFD golden cases with full-batch dense prefixes."""

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PARITY = ROOT / "crates/core/parity_tests/perfmodel"
sys.path.insert(0, str(PARITY))
import yaml
from test_afd_execution_plan_parity import case_config
from test_afd_parity import _run_native


def main():
    path = PARITY / "goldens/afd_execution_plan.json"
    baseline = json.loads(subprocess.check_output(["git", "show", "HEAD:" + str(path.relative_to(ROOT))], cwd=ROOT))
    candidate = json.loads(path.read_text())
    changes = {}
    for name in ["deepseek_v32_tp2", "deepseek_v32_tp4_trace", "deepseek_v3_tp1", "mimo_tp2", "mimo_tp4_trace"]:
        case = candidate["cases"][name]
        with tempfile.TemporaryDirectory(prefix="afd-prefix-refresh-") as temp:
            config = Path(temp) / "prediction.yaml"
            config.write_text(yaml.safe_dump(case_config(case, Path(temp))))
            prediction = _run_native(config)
        summary = prediction.get("summary", prediction)
        assert summary["completed_requests"] == summary["num_requests"] == 4
        before = baseline["cases"][name]["expected"]
        after = {key: summary[key] for key in before}
        for key in ["completed_requests", "total_input_tokens", "total_output_tokens"]:
            assert after[key] == before[key], (name, key)
        changes[name] = {"before": before, "after": after, "delta": {k: after[k] - before[k] for k in before}}
        case["expected"] = after
        case["modeling_change"] = (
            "Embedding and the contiguous dense prefix execute once at full replica batch before "
            "the routed microbatch pipeline. Full-batch activations also enter memory/KV preflight. "
            "Request and token totals are unchanged."
        )
    # Collect and validate every result before replacing either artifact.
    runtime = ROOT / "python/aisimulate/src/aisimulate/_runtime.abi3.so"
    evidence = {
        "baseline_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "candidate_binary_sha256": hashlib.sha256(runtime.read_bytes()).hexdigest(),
        "oracle": (
            "Rust synthetic non-linear GEMM fixture: B=8, m=3, prefix=9 ms plus routed pipeline=7 ms, total=16 ms"
        ),
        "cases": changes,
    }
    path.write_text(json.dumps(candidate, sort_keys=True, indent=2) + "\n")
    Path(__file__).with_name("golden-deltas.json").write_text(json.dumps(evidence, indent=2) + "\n")
    for name, row in changes.items():
        print(name, row["delta"], flush=True)


if __name__ == "__main__":
    main()
