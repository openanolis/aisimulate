#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare and refresh native AFD replay goldens after explained modeling changes."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = ROOT / "crates/core/parity_tests/perfmodel/goldens"

E2E_RTOL = 1e-6
# Absolute floor under a relative tolerance: golden fields that are pure
# float noise around zero (e.g. std_* of a constant frozen step, ~1e-13)
# have no meaningful relative error. 1e-9 is orders of magnitude tighter
# than the e2e rtol on any ms / tok-s quantity.
ABS_FLOOR = 1e-9

# summary fields that are counts/identifiers rather than deterministic floats
# of the measurement: compared exactly. Keys absent from the golden summary are
# not listed — a dead key here silently widens nothing but hides drift.
EXACT_KEYS = {
    "completed_requests",
    "num_requests",
    "num_e2e_latency_samples",
    "num_ttft_samples",
    "num_tpot_samples",
    "total_input_tokens",
    "total_output_tokens",
    "decode_gpus_per_worker",
    "prefill_gpus_per_worker",
}

# Wall-clock-derived fields: the replay's real execution speed on the golden
# machine, not simulated time. Excluded from the gate like every other
# wall-clock timestamp in the parity conventions.
WALL_CLOCK_KEYS = {
    "wall_time_ms",
    "processed_tokens_per_s",
    "processed_output_tokens_per_s",
}


class Divergence(Exception):
    pass


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def diff_maps(
    golden,
    actual,
    *,
    rtol: float,
    path: str = "",
    exact_keys: frozenset | set = frozenset(),
    require_keys: bool = False,
) -> list[str]:
    """Field-by-field recursive diff over golden keys; returns failure lines.

    Actual-only keys are allowed. Missing golden keys fail when
    ``require_keys`` is set, as required by the native replay gate.
    """
    failures: list[str] = []
    if isinstance(golden, dict) and isinstance(actual, dict):
        for key in sorted(golden):
            if key in WALL_CLOCK_KEYS:
                continue
            if key not in actual:
                if require_keys:
                    failures.append(f"{path}/{key}: golden key missing from actual")
                continue
            failures += diff_maps(
                golden[key],
                actual[key],
                rtol=rtol,
                path=f"{path}/{key}",
                exact_keys=exact_keys,
                require_keys=require_keys,
            )
        return failures
    if _is_number(golden) and _is_number(actual):
        leaf = path.rsplit("/", 1)[-1]
        if leaf in exact_keys:
            if golden != actual:
                failures.append(f"{path}: exact mismatch golden={golden!r} actual={actual!r}")
            return failures
        g, a = float(golden), float(actual)
        if not (math.isfinite(g) and math.isfinite(a)):
            if g != a:
                failures.append(f"{path}: non-finite mismatch golden={g!r} actual={a!r}")
            return failures
        delta = abs(g - a)
        limit = max(rtol * max(abs(g), 1e-300), ABS_FLOOR)
        if delta > limit:
            rel = delta / max(abs(g), 1e-300)
            failures.append(f"{path}: golden={g!r} actual={a!r} abs_delta={delta:.3e} rel_delta={rel:.3e} rtol={rtol}")
        return failures
    if golden != actual:
        failures.append(f"{path}: golden={golden!r} actual={actual!r}")
    return failures


def _load(path: Path):
    return json.loads(path.read_text())


def _check_e2e(case: str, actual_path: Path) -> list[str]:
    golden_case = _load(GOLDEN_DIR / "afd_e2e.json")["cases"][case]
    prediction = _load(actual_path)
    metrics = prediction.get("summary", prediction)
    return diff_maps(
        {"summary": golden_case["summary"]},
        {"summary": metrics},
        rtol=E2E_RTOL,
        exact_keys=EXACT_KEYS,
        require_keys=True,
    )


def _without_wall_clock(value):
    if isinstance(value, dict):
        return {key: _without_wall_clock(item) for key, item in value.items() if key not in WALL_CLOCK_KEYS}
    if isinstance(value, list):
        return [_without_wall_clock(item) for item in value]
    return value


def _refresh_e2e(cases: list[str], reason: str, attributions: dict[str, str] | None = None) -> None:
    path = GOLDEN_DIR / "afd_e2e.json"
    fixture = _load(path)
    selected = (
        [name for name, record in fixture["cases"].items() if not record.get("historical", False)]
        if cases == ["all"]
        else cases
    )
    unknown = set(selected) - set(fixture["cases"])
    if unknown or len(selected) != len(set(selected)):
        raise ValueError(f"unknown or duplicate cases: {cases}")
    if not reason.strip():
        raise ValueError("a modeling-change reason is required")
    attributions = attributions or {}
    if set(attributions) - set(selected) or any(not value.strip() for value in attributions.values()):
        raise ValueError("attribution overrides require a selected case and a nonempty description")
    env = dict(os.environ)
    cli = ROOT / "python/aisimulate/.venv/bin/aisimulate"
    for case in selected:
        record = fixture["cases"][case]
        with tempfile.TemporaryDirectory(prefix="afd-golden-refresh-") as directory:
            command = [
                str(cli),
                "predict",
                "--config",
                str(ROOT / record["config"]),
                "--output-dir",
                directory,
                "--format",
                "json",
            ]
            result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
            if result.returncode:
                raise ValueError(f"{case}: prediction failed:\n{result.stderr}")
            prediction = _load(Path(directory) / "prediction.json")
            summary = prediction.get("summary", prediction)
        for key in EXACT_KEYS:
            if key in record["summary"] and summary.get(key) != record["summary"][key]:
                raise ValueError(f"{case}: lifecycle/accounting field {key} changed")
        differences = diff_maps(record["summary"], summary, rtol=E2E_RTOL, exact_keys=EXACT_KEYS, require_keys=True)
        print(f"{case}: {len(differences)} changed fields", flush=True)
        record["summary"] = _without_wall_clock(summary)
        record["modeling_change"] = reason
        if case in attributions:
            if record.get("attribution") != attributions[case]:
                record.setdefault("historical_attribution", []).append(record.get("attribution"))
            record["attribution"] = attributions[case]
    # All selected cases must succeed before the fixture is replaced.
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix="afd-golden-", delete=False) as handle:
        json.dump(fixture, handle, indent=2, sort_keys=True)
        handle.write("\n")
        replacement = Path(handle.name)
    replacement.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    e2e = sub.add_parser("e2e", help="diff an actual prediction.json against the e2e golden")
    e2e.add_argument("--case", required=True, help="case id in goldens/afd_e2e.json")
    e2e.add_argument("--actual", required=True, type=Path, help="actual prediction.json")

    refresh = sub.add_parser("refresh-e2e", help="regenerate selected current CLI cases after a modeling change")
    refresh.add_argument("--case", required=True, nargs="+", help="case IDs, or 'all'")
    refresh.add_argument("--reason", required=True, help="explain the intentional modeling change")
    refresh.add_argument(
        "--attribution",
        nargs=2,
        action="append",
        metavar=("CASE", "DESCRIPTION"),
        help="Replace a selected case's description while retaining its historical attribution",
    )

    args = parser.parse_args(argv)
    try:
        if args.command == "refresh-e2e":
            _refresh_e2e(args.case, args.reason, dict(args.attribution or []))
            return 0
        failures = _check_e2e(args.case, args.actual)
    except (KeyError, OSError, ValueError) as exc:
        print(f"parity harness error: {exc}", file=sys.stderr)
        return 1

    if failures:
        print(f"PARITY DIVERGED ({len(failures)} fields):", file=sys.stderr)
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        return 1
    print("parity OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
