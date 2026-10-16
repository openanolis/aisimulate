# AFD local FFNs and attention TP evidence

<a id="revisions-and-measurement-boundaries"></a>

## Measurement identity and scope

[identity.json](identity.json) records baseline
`4a934b0756c28b53b42027b955c23148ee1f738c`, the historical candidate source
manifest, loaded binaries, dependencies and toolchain. Both arms use matching
release `extension-module` builds and macOS dynamic-lookup flags, with unchanged
system data. Campaigns ran sequentially after builds/tests, without CPU affinity.

After the final routed precision/distribution composition fix, new golden,
prepared-pool and scaling campaigns were rerun. Legacy benchmark/forward-gate
paths did not use selective composition; their binary and path were unchanged.
Additional `cfg(test)` assertions followed the release build.

<a id="functional-evidence"></a>

## Results

| Validation | Recorded outcome |
| --- | --- |
| Rust library | 2,010 passed, one existing ignored test; independent shape, residency, energy, padding, pipeline and wire oracles. |
| Native parity | 417 passed; all 404 existing cases passed. Legacy fixtures changed only schema 28 → 29, without timing refresh. |
| SDK/search/artifacts/model | 269 passed; two unavailable HF metadata tests fail on both arms. Final composition suite: 18 passed, including GLM-5.2's 21 full-index/57 reuse positions and routed precision overrides. |
| Quality checks | Ruff, rustfmt, whitespace, packaged legal files and changed-path CODEOWNERS coverage passed. |

[feature-results.json](feature-results.json) covers 12 native cases: Qwen TP2/4/8,
DeepSeek V3 TP1/V3.2 TP2/4, MiMo TP2/4, both layouts, variable traces,
heterogeneous H200/B200, cached/chunked prefill and either AFD+PD role. Token
counts, binary identity and source plans are checked.
[recommendations/](recommendations/) retains a real three-trial DeepSeek search
and TP1/2/4 saved configurations.

[padding-oracle.json](padding-oracle.json) and
[padding-expectation-deltas.json](padding-expectation-deltas.json) retain the
independent busiest-rank message sizes and explained changes to four new points.
Nonmonotonic small-message curves can lower latency with larger padding; no
unexplained legacy golden change was accepted.

<a id="host-execution-cost"></a>

The [forward gate](forward-gate/summary.md) runs five ordinary SILICON cases,
five paired rounds, ten warmups and 100 calls through [the stdio adapter](measure_forward.py).
Nine cold/warm comparisons are stable and one noisy; none regress or are invalid.
Predicted values match. [prepared-pools.json](prepared-pools.json) uses three
fresh processes per A/F graph for Qwen TP2, DeepSeek V3.2 TP4 and MiMo TP2:
construction 275–766 ms, lazy first A-prefill 189–421 ms, cached warm queries
0.2–4.8 µs, process peak RSS 139–170 MiB.

[host-performance.json](host-performance.json) uses four fresh CLI pairs per row
after one warmup per arm. Full CLI includes startup, construction, replay and
output; native `Replayer::run_inner` includes collector finalization but excludes
model construction and final evidence/output processing.

| Legacy workload | CLI paired median change | Native paired median change |
| --- | ---: | ---: |
| 16-request fixed control, four stages | −0.5% | −3.9% |
| 1,024-request variable trace, four stages | +1.5% | +0.4% |
| 1,024-request variable trace, two stages | −0.4% | +6.9% |

The two-stage increase prompted two warmups and eight further pairs in
[host-performance-two-stage-repeat.json](host-performance-two-stage-repeat.json):
+0.2% CLI, +1.5% native median, native paired ratios 0.943–1.074. Simulated
durations and all request/token totals match exactly between arms.

[new-replay-scaling.json](new-replay-scaling.json) has three CLI runs per new
layout at 16/1,024 variable requests, concurrency32, ISL512/640/768 and
OSL32/40/48, with saturated admission and prefix reuse. All token totals match.

| New layout | Native median, 16 requests | Native median, 1,024 requests |
| --- | ---: | ---: |
| Qwen TP2, four stages | 186 ms | 231 ms |
| DeepSeek V3.2 TP4, four stages | 437 ms | 512 ms |
| MiMo TP4, two stages | 199 ms | 297 ms |

<a id="accuracy-and-remaining-limits"></a>

## Limits

These samples establish local behavior/cost, not a reproducible speedup,
material regression or physical serving/overlap/power accuracy. The forward gate
is five of 64 CI cases. Pool queries exclude AFD communication, pipeline and
memory qualification; warm queries can hit cache. Process RSS is not retained
cache size. Declared data reuse and inferred BF16 FMHA remain visible; these
checks do not supply measured FP8 DSA coverage for DeepSeek V3.2. New scaling
rows have zero measured power coverage; energy is checked arithmetically.
Unavailable HF metadata is not passed coverage. The campaign did not qualify
pure dense/state families, F TP or F-side routing. No launch adapter is claimed.

## Reproduction

Use matching source snapshots/extensions from the recorded identity; a source
manifest alone does not reconstruct the candidate. Run [measure_forward.py](measure_forward.py),
[measure_prepared_pools.py](measure_prepared_pools.py) and
[measure_new_replay.py](measure_new_replay.py) sequentially with the recorded
build settings. Workloads use repository metadata and deterministic traces;
no external workload was copied.
