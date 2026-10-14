<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AFD DP/EP support restriction validation

## Measurement identity and scope

Baseline: `138338d10035bd59947e5d3845b6947565dd3a4d` in isolated sources;
[validation.json](validation.json) identifies the historical candidate diff,
loaded module hashes, release build and samples. Both arms match Cargo.lock,
Python and unchanged performance data on macOS arm64; baseline data was symlinked.

The recorded `afd-predict-pure.yaml` uses Qwen3-30B-A3B, A TP1/eight replicas,
F TP1/explicit EP8 and 16 GPUs in both arms. Two- and 32-request cases each have
three alternating pairs, after native parity finished. This is the restriction
snapshot's scope, not today's A TP support; see
[current AFD configuration](../../../docs/replay/engine/analytical.md#afd).

## Results

Every non-wall-clock summary field agrees exactly in all six pairs, including
latency, throughput, work/GPU hours and power diagnostics. Host-throughput fields
are excluded. Independent Rust fixtures cover volumes, geometry, capacity and
positive energy; native data has power coverage0 and withholds `power_w`.

| Requests | Full CLI baseline median (ms) | Full CLI candidate median (ms) | Native interval baseline median (ms) | Native interval candidate median (ms) |
|---|---:|---:|---:|---:|
| 2 | 2256.805 | 2152.767 | 186.627 | 182.556 |
| 32 | 2194.691 | 2201.880 | 179.823 | 178.060 |

Full CLI includes startup/preparation/replay/artifact writing. Native replay spans
runtime setup through collector finalization, excluding model preparation and
artifact serialization; it is not loop-only timing. The 32-request CLI ranges
2188.765–2217.848 versus 2195.290–2218.362 ms and native ranges172.092–184.379
versus177.903–184.860 ms overlap. Rust AFD74, serialization21, Python200 and
native parity9 (six supported/three historical cases) passed, as did Ruff/rustfmt/whitespace.

## Limits

Small local samples show no material regression, not a speedup or serving accuracy.
Matching absent power is not positive energy coverage. Preparation checks reuse
metadata; compiled identity adds four u32 (16 bytes) per Engine, with a constant-time
communication guard and no new per-request state or token/layer loops. Historical
pure-EP formulas/evaluation order were unchanged. Digests do not archive candidate sources.

## Reproduction

Recover matching source snapshots/release extensions and use the configuration,
build identity and pair counts recorded in `validation.json` with the predict CLI.
No standalone measurement harness is retained in this directory; the JSON preserves
the original samples rather than a self-contained source/binary archive.
