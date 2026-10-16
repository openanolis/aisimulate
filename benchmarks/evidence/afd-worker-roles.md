<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# AFD worker roles: validation and performance evidence

<a id="scope-and-revisions"></a>

## Measurement identity and scope

Measured 2026-10-14. [afd-worker-roles.json](afd-worker-roles.json) retains timing
samples, configurations/results, source/binary identities and metric hashes.
Baseline: `0f344f77abd1094b03a264313b26f7238a8d57d0`; measured snapshot:
`6cbba3f950201bcb9fb63bfd79748df00f8cdf00`, also identified by the runtime patch
checksum in JSON. This identifies the historical campaign, not the current branch.
Final compiler/Sweeper KV helpers additionally used A-replica EP1 geometry;
matched benchmarks omit KV transfer, so the final role matrix checks it separately.

Matching release builds, dependencies, interpreter and macOS linker flags run
five alternating fresh-process pairs with numerical threads fixed to one.
Qwen3-30B-A3B/H200/SGLang0.5.14 covers the three original AFD combinations,
ordinary fixed/op-level controls and AFD decode search with pinned ordinary prefill.

<a id="cost-and-numerical-behavior"></a>

## Results

Medians are milliseconds:

| Case, 1,000 requests | Compile, base → candidate | Complete Runner, base → candidate | Native replay, base → candidate | Fresh process, base → candidate |
|---|---:|---:|---:|---:|
| AFD aggregated | 1.435 → 1.568 | 1641.919 → 1627.805 | 233.436 → 230.268 | 2072.519 → 2056.493 |
| AFD prefill | 1.457 → 1.726 | 1746.866 → 1746.330 | 320.101 → 320.235 | 2181.409 → 2178.178 |
| AFD decode | 1.462 → 1.763 | 1465.490 → 1463.600 | 242.293 → 241.941 | 1891.872 → 1903.952 |
| Ordinary fixed | 0.047 → 0.049 | 73.783 → 73.759 | 41.536 → 41.472 | 490.700 → 486.339 |
| Ordinary op-level | 0.179 → 0.180 | 1415.846 → 1423.861 | 224.543 → 224.647 | 1844.385 → 1853.724 |

All simulated keys/work counts agree, including the recorded 32-request cases;
maximum rounding difference is `1.64e-11`, within `1e-9` relative/absolute tolerance.
No numeric golden changed. Recommendation lowering/enumeration changes
131.077 → 90.614 ms cold and 43.586 → 3.150 ms warm with identical candidate
counts; GPU-bucket composition avoids the full Cartesian product and preserves
role order. No scheduler/event-loop or token scan is added.

Zero-request preparation includes materialization, model/memory construction,
empty replay, report serialization and teardown, excluding compilation/startup:

| Case | Baseline ms | Candidate ms |
|---|---:|---:|
| AFD aggregated | 1397.617 | 1386.208 |
| AFD prefill + ordinary decode | 1420.474 | 1419.723 |
| Ordinary prefill + AFD decode | 1225.286 | 1234.800 |
| Ordinary fixed | 0.355 | 0.319 |
| Ordinary op-level | 1201.062 | 1211.064 |

Candidate-only double-AFD (five runs) medians: preparation1440.230 ms,
compile1.737 ms, Runner1747.885 ms, native269.369 ms, fresh process2198.510 ms.
Each role uses 1A+1F; 1,000 requests complete ISL1024/OSL128 at concurrency32.
Runner includes model construction/memory/report normalization; native excludes
model construction/final JSON; fresh process excludes supervision/output files.

<a id="semantic-qualification"></a>

Four-combination tests include parsing/search/reload, native completion, both
handoff timing modes and independent memory failures. Asymmetric P:1A+1F/four-stage,
D:2A+2F/two-stage verifies 48 GPUs and separate capacities. Automatic KV bytes use
single-request A geometry, independent of F EP or whole-pool batch.

Eight ordinary SGLang/TRT-LLM/vLLM agg/disagg/ADP scenarios replay 5,000 Mooncake
rows twice per arm. Canonical records match byte-for-byte; every output completes,
disagg has 5,000 handoffs, and pressure records are readmitted. Baseline-tuned
capacities were frozen: SGLang agg2048, disagg8500/role, ADP decode1000 blocks;
vLLM disagg8800; other seeds unchanged. Earlier seed digests are historical.
Duration aggregation is checked within `1e-6` ms. Both builds use the same
regenerated offline runner lock (supplied core0.12 lock versus checkout0.13).
Trace revision/path/full/slice hashes and runner hashes are recorded in JSON.

| Validation | Recorded outcome |
| --- | --- |
| Python regressions | 1,621 passed; five existing integration-only skips. |
| Native role / Rust AFD / existing CLI parity | 42 / 70 / 19 passed. |
| Collector follow-up | 23 passed from package root on both arms; three subprocess checks passed with absolute `PYTHONPATH`. |
| Quality checks | Ruff, rustfmt, documentation links and CODEOWNERS coverage passed. |

## Limits

These local unpinned host timings are not portable guarantees; replay parity is
not serving accuracy. Upstream trace content is not distributed. Earlier Collector
failures involved cwd-relative imports, module caching and subprocess paths; no
clean full-repository unit run is claimed. Source/binary/patch digests alone do
not reconstruct unavailable historical artifacts or an absent Git snapshot.

<a id="checks-and-reproduction"></a>

## Reproduction

Build matching historical source trees/extensions, then:

```sh
python/aisimulate/.venv/bin/python scripts/benchmark_afd_roles.py \
  --baseline-source /path/to/baseline/python/aisimulate/src \
  --candidate-source /path/to/candidate/python/aisimulate/src \
  --pairs 5 --output /tmp/afd-worker-role-performance.json
```

Use `--preparation-only` for preparation; candidate-only runs use
`--worker afd:dual:large` and candidate `PYTHONPATH`. Run SDK tests from
`python/aisimulate` with absolute source/package paths; on macOS retain the timeout
plugin with `--timeout-method=thread` to avoid SIGALRM/strict-marker conflicts.
