# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Freeze source, data, toolchain and binary identities for this local comparison."""

import hashlib
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = Path("/private/tmp/afd-unification-baseline")
OUT = Path(__file__).parent
SHA = "23d21fa4f62ff47e742c48c26d3ed202f97308d2"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest(root):
    files = [
        root / p
        for p in [
            "Cargo.toml",
            "Cargo.lock",
            "crates/core/Cargo.toml",
            "python/aisimulate/pyproject.toml",
            "python/aisimulate/uv.lock",
        ]
    ]
    for directory in ["crates/core/src", "python/aisimulate/src"]:
        files.extend(
            p
            for p in (root / directory).rglob("*")
            if p.is_file()
            and "__pycache__" not in p.parts
            and p.suffix not in {".pyc", ".so", ".dylib"}
            and p.name != ".DS_Store"
        )
    return {str(p.relative_to(root)): digest(p) for p in sorted(files)}


def main():
    records = {}
    for arm, root in [("baseline", BASE), ("candidate", ROOT)]:
        source = manifest(root)
        encoded = json.dumps(source, sort_keys=True, separators=(",", ":")).encode()
        binary = root / "python/aisimulate/src/aisimulate/_runtime.abi3.so"
        records[arm] = {
            "revision": SHA,
            "working_tree": arm == "candidate",
            "source_manifest_sha256": hashlib.sha256(encoded).hexdigest(),
            "source_sha256": source,
            "binary_sha256": digest(binary),
            "binary_bytes": binary.stat().st_size,
            "fixtures_sha256": {
                str(p.relative_to(root)): digest(p)
                for directory in [
                    "crates/core/parity_tests/perfmodel/configs",
                    "crates/core/parity_tests/perfmodel/goldens",
                ]
                for p in sorted((root / directory).rglob("*"))
                if p.is_file()
            },
        }
        if platform.system() == "Darwin":
            records[arm]["macho_size"] = subprocess.check_output(["size", "-m", str(binary)], text=True)
    data = "python/aisimulate/src/aisimulate_core/systems/"
    data_manifests = [
        {k: v for k, v in records[a]["source_sha256"].items() if k.startswith(data)} for a in ["baseline", "candidate"]
    ]
    assert data_manifests[0] == data_manifests[1], "profile/model metadata inputs changed"
    result = {
        "arms": records,
        "source_manifest_scope": [
            "crates/core/src (all files)",
            "python/aisimulate/src (all source, model metadata and profile data)",
            "Cargo.toml, Cargo.lock, crates/core/Cargo.toml, Python pyproject.toml and uv.lock",
            "Excludes __pycache__, native binaries and .DS_Store; binary identity is separate",
        ],
        "profile_data_unchanged": True,
        "harness_sha256": {p.name: digest(p) for p in sorted(OUT.glob("*.py"))},
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version,
            "cpu_affinity": "unavailable on macOS",
        },
        "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
        "cargo": subprocess.check_output(["cargo", "--version"], text=True).strip(),
        "build": {
            "profile": "release",
            "cargo_features": ["extension-module"],
            "expanded_core_features": ["extension-module", "python"],
            "pyo3_features": ["abi3-py311", "extension-module"],
            "rustc_flags": ["-C", "link-arg=-undefined", "-C", "link-arg=dynamic_lookup"],
            "command": (
                "PYO3_PYTHON=$PWD/python/aisimulate/.venv/bin/python cargo rustc --offline --release "
                "-p aisimulate-core --features extension-module --lib -- "
                "-C link-arg=-undefined -C link-arg=dynamic_lookup"
            ),
        },
        "thread_env": dict.fromkeys(
            ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"], "1"
        ),
        "python_hash_seed": "0",
        "canonical_result_exclusions": [
            "/summary/wall_time_ms",
            "/summary/processed_tokens_per_s",
            "/summary/processed_output_tokens_per_s",
            "/planner/html_report_path",
        ],
    }
    sys.path.insert(0, str(ROOT / "crates/core/parity_tests/perfmodel"))
    from test_afd_execution_plan_parity import CASES, case_config

    with tempfile.TemporaryDirectory(prefix="afd-input-identity-") as temp:
        directory = Path(temp)
        case_config(next(c for c in CASES.values() if c["trace"]), directory)
        trace = directory / "requests.jsonl"
        result["generated_trace"] = {
            "source": "case_config in test_afd_execution_plan_parity.py, repository-authored",
            "generator_sha256": digest(ROOT / "crates/core/parity_tests/perfmodel/test_afd_execution_plan_parity.py"),
            "format": "mooncake",
            "rows": 4,
            "sha256": digest(trace),
            "bytes": trace.stat().st_size,
            "rule": "Four ordered requests at i*2 ms, ISL 128+(i%2)*32, OSL 8+i%2, hashes [42,43,1000+i]",
        }
    if platform.system() == "Darwin":
        for key, name in [
            ("cpu_brand", "machdep.cpu.brand_string"),
            ("logical_cpus", "hw.logicalcpu"),
            ("physical_cpus", "hw.physicalcpu"),
        ]:
            value = subprocess.check_output(["sysctl", "-n", name], text=True).strip()
            result["host"][key] = value if key == "cpu_brand" else int(value)
    (OUT / "identity.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({a: {k: v for k, v in r.items() if k != "source_sha256"} for a, r in records.items()}, indent=2))


if __name__ == "__main__":
    main()
