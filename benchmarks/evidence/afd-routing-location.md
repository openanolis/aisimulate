# AFD routing location: implementation evidence

<a id="revisions-and-measurement-scope"></a>

## Measurement identity and scope

Baseline: `af61a7e9f2080a4914f3be12dbdde7381e0b6e5c`; historical candidate:
that revision plus production diff SHA-256
`3703ee5140cc320fe79b808b71acd88f1be920dd3c730e4ea0eaa0bbaf705d2a`.
[afd-routing-location.json](afd-routing-location.json) records binaries, paths,
ranges and raw samples. Both release extensions share dependencies, interpreter,
macOS link flags and H200/SGLang 0.5.14/SILICON data.

Five alternating pairs use batch4, ISL128/OSL16, ten warmups and 100 forward-worker
samples. Construction and first use are separate; cached canonical queries include
the Python wrapper. Native CLI uses two A workers (TP4), one F node, A batch4,
two microbatches, four stages, concurrency2 and eight requests. Full CLI includes
startup/construction/output; `prediction.wall_time_ms` excludes model construction.

<a id="host-cost"></a>

## Results

Medians are milliseconds except the forward-worker table (µs); parentheses
show median paired changes.

| Model | Canonical build, base → candidate | First query, base → candidate | Cached query, base → candidate | Full native CLI, base → candidate |
|---|---|---|---|---|
| Qwen/Qwen3-32B | 783.514708 → 752.206792 (-4.05%) | 175.388625 → 176.693375 (+0.91%) | 0.000875 → 0.000875 (+0.00%) | 1802.200750 → 1744.538500 (-3.13%) |
| Qwen/Qwen3-30B-A3B | 1199.046875 → 1175.019625 (-2.07%) | 171.135417 → 170.865334 (-0.38%) | 0.000958 → 0.000959 (+0.10%) | 2211.471458 → 2206.506416 (-1.00%) |

| Forward worker case | Base warm median (µs) | Candidate warm median (µs) | Paired change | Votes exceeding both +10% and +2 µs |
|---|---:|---:|---:|---:|
| Qwen3-32B-context | 9.584 | 9.333 | -3.45% | 0/5 |
| Qwen3-32B-generation | 10.208 | 10.000 | -2.04% | 0/5 |
| Qwen3-30B-A3B-context | 8.791 | 8.791 | +0.00% | 0/5 |
| Qwen3-30B-A3B-generation | 9.750 | 9.875 | +1.28% | 0/5 |

<a id="behavior-and-validation"></a>

Forward predictions and dense simulated metrics match, excluding host-time rates;
MoE AFD changes intentionally as router compute/weights move to A. Eight requests
finish in every CLI run. Only qualification adds routing identity/ReplaySpec SHA;
no numeric golden changed. Historical checks passed: Rust AFD 32, identity 1,
embedded error propagation 3; Python SDK 37, topology/search/artifacts 96,
memory/single-oracle 30, parity 394; Ruff/rustfmt/whitespace checks passed.
Explicit F routing fails predict/recommend with exit1 before data loading or
successful artifacts; native Replay preserves `NotImplementedError`.

## Limits

This campaign predates later AFD changes; its dense/boundary cases describe that
snapshot's scope. Samples show no material host-cost increase, not serving
accuracy or a guarantee for all deployment sizes. Preparation adds a bounded mode
check/identity field; classification stays O(ops), without per-request allocation
or new query cache. Recorded digests alone do not archive the candidate source.

## Reproduction

Stage matching baseline/candidate sources and release extensions, sharing unchanged
data, then run:

```sh
python/aisimulate/.venv/bin/python benchmarks/evidence/afd-routing-location-benchmark.py \
  --baseline-source /path/to/baseline/python-source \
  --output /tmp/afd-routing-location.json --rounds 5
```

Normal macOS process-monitoring/priority access is needed; no GPU execution is required.
