# Full-batch dense-prefix AFD validation

## Measurement identity and scope

Baseline source/extension were staged at `5b668b9873664db042e20711ea31a348d607b625`.
The historical candidate is identified by changed-source and loaded release
extension hashes in [performance.json](performance.json), with build flags,
host, workloads and every pair. Performance tables were unchanged; there was
no new configuration field or wire-layout change.

## Results

The independent B=8/m=3 oracle uses nonlinear GEMM rows (m=3: 2 ms, m=8: 3 ms,
padded m=9: 9 ms). Three full-batch prefix ops cost 9 ms once, followed by 7 ms
of routed work, giving 16 ms. It also checks 100 W attribution, verification
width, both layouts, activation bytes and saved-plan rejection.
[validation.json](validation.json) records 567 passing tests;
[golden-deltas.json](golden-deltas.json) explains five timing changes. Request/token
totals and nine no-dense goldens are unchanged. Maverick rejection identifies
dense layer 2 after routed layer 1. These checks qualify the `D* R+` restriction;
shared experts remain part of routed layers.

<a id="host-performance"></a>

Six alternating pairs per row follow one warmup per arm. All requests/tokens
match; the pure-routed control also preserves simulated duration exactly.
Median paired changes are candidate/baseline − 1:

| Workload | Full CLI time | Native replay time |
|---|---:|---:|
| Pure-routed control, 16 requests | -0.63% | -0.73% |
| Dense prefix, four-stage, 128 requests | -1.84% | -0.16% |
| Dense prefix, two-stage, 128 requests | -1.40% | +1.33% |

Full CLI includes supervision, preparation, simulation and output. Native replay
covers `Replayer::run_inner` through collector finalization, excluding provider
construction/output. Preparation adds an O(layers) source check and prefix scalar;
the existing op cache is cleared between prefix/routed shapes without new caches.

## Limits

This unpinned six-pair check shows no material regression in its samples; it is
not a formal statistical gate or serving-accuracy measurement. Increased modeled
activation residency is serving memory, not simulator retained-memory growth.
Hashes identify the historical candidate but do not archive its source/binary.

## Reproduction

With matching staged source snapshots and extensions:

```sh
python/aisimulate/.venv/bin/python benchmarks/evidence/afd-dense-prefix/measure.py \
  --baseline-source /path/to/baseline-python-source --pairs 6
```

[refresh_goldens.py](refresh_goldens.py) reproduces the five selected refreshes
and **writes goldens**; it is not a read-only validation command. Supervised CLI
needs normal macOS process-monitoring permissions.
