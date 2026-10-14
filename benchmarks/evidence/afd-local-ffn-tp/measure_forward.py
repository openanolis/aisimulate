# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""macOS stdio adapter for existing revision-local forward gate workers.

No taskset is available on macOS. Worker protocol, priming, timing and gate
comparison remain the repository's implementations. Run sequentially after
builds/tests; CPU affinity is explicitly unavailable.
"""

import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python/aisimulate"))
from tools.forward_perf_gate import PROTOCOL_VERSION, cases, compare

OUT = pathlib.Path(__file__).parent / "forward-gate"
OUT.mkdir(exist_ok=True)
BASE = pathlib.Path("/tmp/aisimulate-afd-baseline")
sha = json.loads((BASE / "revision.json").read_text())["sha"]
selected = [
    c
    for c in cases.expand_cases()
    if c["database_mode"] == "SILICON"
    and (
        (c["model_id"] == "qwen3-32b" and c["batch_size"] == 1 and c["isl"] == 1024)
        or ("deepseek-v3.2" in c["model_id"] and c["isl"] == 1024 and c["tp_size"] == 8)
    )
]
assert len(selected) == 5
arms = {
    "base": (BASE / "repo/python/aisimulate/tools/forward_perf_gate/worker.py", BASE / "source", sha),
    "head": (
        ROOT / "python/aisimulate/tools/forward_perf_gate/worker.py",
        ROOT / "python/aisimulate/src",
        sha + "+working-tree",
    ),
}
raw = {
    "schema_version": 1,
    "base_revision": sha,
    "head_revision": sha + "+working-tree",
    "configuration": {
        "mode": "local-macos-subset",
        "rounds": 5,
        "warmup": 10,
        "iterations": 100,
        "cpu_affinity": "unavailable on macOS",
        "matrix_case_count": 64,
        "selected_case_count": len(selected),
        "expected_case_ids": [c["case_id"] for c in selected],
    },
    "run_errors": [],
    "cases": [{"case": c, "rounds": []} for c in selected],
}
for round_index in range(5):
    responses = {}
    for arm in ("base", "head") if round_index % 2 == 0 else ("head", "base"):
        worker, source, revision = arms[arm]
        env = {
            **os.environ,
            "PYTHONPATH": str(source),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "RAYON_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
            "AIC_ALLOW_UNLISTED_VERSIONS": "1",
        }
        request = {
            "protocol_version": PROTOCOL_VERSION,
            "revision": revision,
            "cases": selected if round_index % 2 == 0 else selected[::-1],
            "warmup": 10,
            "iterations": 100,
        }
        p = subprocess.run(
            [sys.executable, str(worker)],
            input=json.dumps(request),
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        results = json.loads(p.stdout)["results"]
        responses[arm] = {r["case_id"]: r for r in results}
    for entry in raw["cases"]:
        cid = entry["case"]["case_id"]
        r = {"round": round_index, **{a: responses[a][cid] for a in arms}}
        assert all(r[a].get("status") == "OK" for a in arms), r
        assert abs(r["base"]["predicted_value_ms"] - r["head"]["predicted_value_ms"]) < 1e-9
        entry["rounds"].append(r)
    (OUT / "raw-results.json").write_text(json.dumps(raw, indent=2) + "\n")
    print("forward round", round_index + 1, flush=True)
comparison = compare.compare_raw(raw)
compare.write_outputs(comparison, OUT)
print(compare.render_markdown(comparison))
assert not comparison["blocking"], "forward gate requires investigation"
