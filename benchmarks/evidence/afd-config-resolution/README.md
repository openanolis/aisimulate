# AFD configuration resolution

## Measurement identity and scope

[results.json](results.json) retains source/binary hashes, toolchain, release
features/flags, dependency lock and raw samples. The clean source revision was
`4ed29151f6dc28219f0c9b223b13c33e203918d6`; the baseline release artifact is also
identified in [the preceding campaign](../afd-unification/identity.json).
The candidate is the historical source manifest and freshly built extension
recorded in `results.json`, not the reader's checkout.

Three alternating process pairs use Qwen3-30B-A3B/H200/SGLang, ISL512, OSL32,
concurrency16, with four or 1,000 synthetic requests. Each process runs three
warmups and 15 native calls; no competing builds/tests ran. The baseline-only
wire adapter reconstructs redundant inputs from the authoritative rank.

<a id="behavior-checks"></a>
<a id="context-window-resolution-follow-up"></a>

## Results

| Validation stage | Recorded outcome |
| --- | --- |
| Initial ownership/precision change | Rust AFD: 71; embedded Python AFD: 77; config: 40 passed. Python SDK/search/artifacts: 150 passed, plus two later native boundary regressions; final search file: 45 passed. |
| Initial numerical parity | 391 passed with 28 CLI cases blocked by macOS process access; rerunning both AFD files passed 34 cases, covering the complete 419-case set (385 ordinary). No golden changed. |
| Context-window follow-up | Rust AFD: 80; config: 40; ordinary-runtime default test passed. Python: 591 distinct cases (154 SDK/search, 34 AFD CLI, 403 runtime/artifact/ordinary parity). Goldens unchanged. |

Regressions check shared topology/timing identity, early backend/precision
rejection, FP8 serial SOL latency, idempotent legacy migration/conflicts, and
explicit versus absent context declarations across reload and runtime admission.
The follow-up preserves optional `rank.max_model_len`; unrepresentable legacy
scheduler caps fail. Ruff, Rust formatting and whitespace checks passed.

<a id="local-host-cost"></a>

| Workload | Baseline warm medians | Candidate warm medians | Median paired change |
| --- | --- | --- | --- |
| Four requests | 19.47–19.61 ms | 19.49–19.56 ms | −0.2% |
| 1,000 requests | 56.22–58.64 ms | 56.30–57.29 ms | +1.0% |

The native-call timer includes pool construction, memory preflight, replay,
evidence and JSON serialization; it excludes Python lowering/imports/startup.
The first four-request call includes cold construction: 1,501–1,554 ms baseline,
1,481–1,544 ms candidate. The first 1,000-request call has warm metadata.
Peak process RSS is 310–324 versus 307–317 MiB, including transient reports/tables.
Canonical reports match exactly after excluding only `wall_time_ms`,
`processed_tokens_per_s` and `processed_output_tokens_per_s`; completion and
token counts agree. An exploratory sample taken during tests was discarded.

## Limits

Host-cost samples **predate the context-window follow-up**. That follow-up adds
one borrowed workload decode per role factory, outside prediction/request loops;
its host timings were not repeated. Shared topology ownership adds one allocation
per parsed rank and avoids deep clones; no request-history work was added.
The samples show variation, not a reproducible material slowdown or a retained
memory reduction. They do not measure serving accuracy or the full forward matrix.
Source hashes identify snapshots but do not archive their source or binaries.

## Reproduction

Use the recorded source snapshots and matching release extensions; build flags
are in `results.json`. Run [measure.py](measure.py) with `BINARY --legacy-input`
for baseline and `BINARY` for candidate, alternating three pairs. Set
`PYTHONPATH=python/aisimulate/src`, `PYTHONHASHSEED=0`, and `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `RAYON_NUM_THREADS` to 1.
The follow-up's 403-case command is:

```sh
python/aisimulate/.venv/bin/python -m pytest -p no:timeout -c python/aisimulate/pytest.ini -o addopts='' tests/sweeper/test_afd_search_integration.py::test_native_afd_undeclared_window_still_limits_runtime_prompt_length tests/test_afd_artifacts.py crates/core/parity_tests/perfmodel/test_engine_step_parity.py crates/core/parity_tests/perfmodel/test_compile_engine_parity.py -q --tb=short -x
```
