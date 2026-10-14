<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AFD execution unification

## Measurement identity and scope

Baseline: `23d21fa4f62ff47e742c48c26d3ed202f97308d2`; historical candidate:
the source manifest and loaded binary recorded in [identity.json](identity.json).
Both arms use matching release flags, Cargo dependencies, Python environment,
profiles and model metadata. `/private/tmp/afd-unification-baseline` was the
measurement-time baseline location, not a persistent source archive.

[performance.json](performance.json) records three alternating pairs: fresh A/F
construction, first native call, ten warmups/30 native calls, and full CLI at
four/1,000 requests. Native workers check binary hashes. Apple M2 Pro/macOS15.8,
Python3.13.16/rustc1.99.0; threads fixed to one, no CPU affinity or competing tests.
Full CLI includes supervision, construction, memory/replay/evidence and output;
`wall_time_ms` excludes constructor and final evidence costs.

<a id="semantic-changes"></a>

## Results

[semantic-deltas.json](semantic-deltas.json) retains 14 source-plan and six native
end-to-end comparisons. Baselines match their original goldens; request/token
totals agree. Ordinary engine/compile goldens are unchanged.

| Cases | Explanation |
|---|---|
| Qwen TP2/TP4, DeepSeek V3/V3.2, MiMo fixed synthetic | Actual-batch prefill replaces configured-capacity charging. TTFT and duration decrease; decode pricing is retained. Last-bit TPOT differences follow subtraction at shifted timestamps. |
| Qwen TP1/MTP and pure TP1 end-to-end | Source-plan endpoints run once per microbatch instead of layer amortization. Transfers use mapped per-NIC send/receive load; return combine uses F-node partials. These changes also affect decode cadence and power evidence. |
| Prefix/chunked prefill | Each chunk uses its real uncached length, prefix and batch. Queueing metrics follow the revised pass duration. |
| Heterogeneous A/F pools | Actual-batch prefill applies independently of hardware. Each pool retains its selected data root. |
| AFD+PD | Prefill-side AFD changes first-token/handoff timing. Pure TP1 decode-side AFD also changes destination cadence. TP2 decode-only AFD has no numerical change. |
| Trace-only DeepSeek/MiMo and Qwen TP8 full-batch control | No numerical changes: these already used source plans and actual/full batch geometry. |

Independent tests check rank slices, timelines, residency and energy. MTP depth3
keeps prefill activation unchanged, quadruples decode activation and reduces KV
budget; plan buffers are added once. Explicit source policies survive all caller
boundaries. The F EP64 constructor emits one routed template for 48 layers;
source model identity must agree even when dimensions match. Single-request
prefill is invariant under configured batch capacity for synthetic and trace inputs.

[repeatability.json](repeatability.json) covers two fresh processes per revision
for all 20 cases: full canonical reports, including request/power data, agree
after excluding the three recorded host-time fields. The independent pressure
test captures one to three retractions, readmits/completes three requests with
72 output tokens and compares both decode identities.

<a id="host-cost"></a>

| Median across rounds | Qwen TP1 baseline → candidate | DeepSeek V3.2 TP2 baseline → candidate |
|---|---:|---:|
| Cold A constructor | 1,133 → 694 ms | 237 → 234 ms |
| Cold F constructor | 1,116 → 639 ms | 635 → 646 ms |
| First complete native call | 1,548 → 1,508 ms | 2,183 → 2,154 ms |
| Warm complete native call | 23.61 → 19.34 ms | 23.51 → 20.63 ms |
| Full CLI, four requests | 2,236 → 2,208 ms | 2,904 → 2,849 ms |
| Full CLI, 1,000 requests | 2,234 → 2,204 ms | 2,896 → 2,904 ms |

Warm medians range 23.50–23.98 → 19.27–19.47 ms for Qwen and
23.42–23.75 → 20.61–20.96 ms for DeepSeek. Process peak RSS ranges are
214.0–217.7 → 212.8–216.2 MiB and 276.0–277.4 → 263.1–273.3 MiB.
Preparation/per-pass caches are bounded by compiled work, reuse shapes and
KV-independent communication, retain one frozen timing/evidence pair and prepare
F memory metadata once for both phases, without request history.

The [ordinary forward gate](forward-gate/summary.md) covers five cases, five
alternating rounds, ten warmups and 100 calls. Predictions match; eight cold/warm
comparisons are stable, two DeepSeek BS128 cold comparisons noisy (one round
each), without consistent regression or blocking result.

| Check | Result |
|---|---:|
| `cargo test --offline -p aisimulate-core --lib` | 2,001 passed, one ignored |
| `cargo test --offline -p aisimulate-core --features embed-python afd --lib` | 76 passed |
| SDK AFD memory/KV/composition/materialization | 90 passed |
| Sweeper/config/artifact tests | 279 passed in the final subset; wider 352-test subset also passed |
| Ordinary engine/compile parity | 385 passed |
| Native AFD parity | 34 passed |
| Single-oracle and operation-spec contract | 11 passed |
| Pool fixture regeneration | Three byte-identical files |
| Ruff, Rust formatting, diff whitespace, documentation links and CODEOWNERS coverage | Passed |

## Limits

This is local AFD qualification and five of 64 forward cases, not the long-corpus
campaign or physical serving-accuracy/deployment validation. First-call/CLI
differences are small relative to startup variation. RSS shows no consistent
regression but establishes no memory reduction. Identity manifests identify
source/binary content without archiving it; the original temporary baseline
location and candidate artifacts must be recovered for historical reproduction.

<a id="verification-and-reproduction"></a>

## Reproduction

Use matching source snapshots/release extensions and the project interpreter.
The scripts hard-code `/private/tmp/afd-unification-baseline`; restore the
recorded baseline there. Set absolute `PYO3_PYTHON` and
`PYTHONPATH=$PWD/python/aisimulate/src`; embedded tests use `-p no:timeout`.
Normal macOS process/sysctl access is needed for CLI and `record_identity.py`.
Run [repeatability.py](repeatability.py) for repeated semantics, then
[measure.py](measure.py) and [measure_forward.py](measure_forward.py) sequentially.
[qualify.py](qualify.py) compares paired cases and **refreshes AFD goldens** after
matching totals; it is not read-only. [record_identity.py](record_identity.py)
writes source/data/binary identity. None of these historical campaigns was rerun
for documentation cleanup.
