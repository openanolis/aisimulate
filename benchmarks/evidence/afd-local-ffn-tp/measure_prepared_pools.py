# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fresh-process preparation and first/warm query costs of selective A/F graphs.

These are canonical pool-graph queries, excluding AFD communication/pipeline
composition and memory qualification. Full CLI/native replay covers that work
separately. Use the repository forward gate's timing helper; no Python timing
prediction or per-rank reconstruction is used.
"""

import argparse
import hashlib
import json
import os
import pathlib
import platform
import resource
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python/aisimulate"))


def worker(config):
    from tools.forward_perf_gate.measurement import measure_cold_and_warm

    import aisimulate._runtime as native
    from aisimulate_core.sdk.rust_engine_step import RustForwardPassPerfModel

    started = time.perf_counter_ns()
    model = RustForwardPassPerfModel.best_available(config)
    prep_ms = (time.perf_counter_ns() - started) / 1e6
    phases = {}
    for prefill in [True, False]:
        name = "prefill" if prefill else "decode"
        shapes = {}
        for batch, length in [(1, 128), (64, 2048)]:
            call = lambda: model.static_phase_latency(
                batch_size=batch, input_tokens=length, output_tokens=2, prefill=prefill
            )
            value, first, samples, stats = measure_cold_and_warm(call, warmup=10, iterations=100)
            assert value > 0
            shapes[f"batch{batch}_isl{length}"] = {
                "pool_graph_latency_ms": value,
                "first_us": first,
                "warm": stats,
                "warm_samples_us": samples,
            }
        phases[name] = shapes
    return {
        "config": config,
        "preparation_ms": prep_ms,
        "phases": phases,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "runtime_sha256": hashlib.sha256(pathlib.Path(native.__file__).read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(json.load(sys.stdin))))
        return
    result = {"host": platform.platform(), "scope": __doc__, "fresh_processes_per_row": 3, "rows": []}
    for model, tp in [("Qwen/Qwen3-30B-A3B", 2), ("deepseek-ai/DeepSeek-V3.2", 4), ("XiaomiMiMo/MiMo-V2-Flash", 2)]:
        for role in ["a", "f"]:
            config = {
                "model": model,
                "system": "h200_sxm",
                "backend": "sglang",
                "backend_version": "0.5.14",
                "worker_type": "aggregated",
                "afd_pool_role": role,
                "afd_routing_location": "a",
                "tp": tp if role == "a" else 8,
                "moe_tp_size": tp if role == "a" else 1,
                "moe_ep_size": 1 if role == "a" else 8,
                "estimation_mode": "op_level",
            }
            row = {"config": config, "samples": []}
            for _ in range(3):
                env = {
                    **os.environ,
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "RAYON_NUM_THREADS": "1",
                    "PYTHONHASHSEED": "0",
                }
                p = subprocess.run(
                    [sys.executable, __file__, "--worker"],
                    input=json.dumps(config),
                    env=env,
                    text=True,
                    capture_output=True,
                    check=True,
                )
                row["samples"].append(json.loads(p.stdout))
            result["rows"].append(row)
            pathlib.Path(__file__).with_name("prepared-pools.json").write_text(json.dumps(result, indent=2) + "\n")
            print(model, role, "OK", flush=True)


if __name__ == "__main__":
    main()
