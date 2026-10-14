<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AFD MTP communication work scaling

## Measurement identity and scope

Baseline: `811e04f64881b9b019fae997972efbc55ab63785`; candidate: the historical
AFD communication patch on that revision. Both native builds use matched offline
release settings and macOS linker flags. These unique build-time digests are
retained here; source hashes predate subsequent comment-only cleanup:

| Library | SHA-256 |
|---|---|
| Baseline | `caa452d04905db13141d04e2f309fba0af4b48ea215ab55ae1a933c25cb000f0` |
| Candidate | `a64c7ce1b110f233be2678129cb4412d661cfde94319f97e330ab66338bbfa10` |

| File | SHA-256 |
|---|---|
| `crates/core/src/perfmodel/afd/model.rs` | `73a5994633faa691d1abec9fbd2ef85d6f9ddd91c9738b7ef1978ab63f608023` |
| `crates/core/src/perfmodel/afd/execution.rs` | `824816c7939bca4aba935a5df75148a4a2bfbbf33ffdb16a86ad12e28fcb91d5` |

<a id="native-numerical-comparison"></a>
<a id="behavioral-anchors"></a>

## Results

Native predict cases `qwen_tp1_mtp`/`qwen_tp2_mtp` in
[`afd_execution_plan.json`](../../../crates/core/parity_tests/perfmodel/goldens/afd_execution_plan.json)
record Qwen3-30B-A3B/H200/SGLang0.5.14, 48 layers, `nextn=3`,
`nextn_accepted=1.5`, two microbatches, four stages, concurrency4, ISL128/OSL8.

| A TP | Baseline TPOT (ms) | Candidate TPOT (ms) | Baseline duration (ms) | Candidate duration (ms) | TTFT (ms), both |
|---|---:|---:|---:|---:|---:|
| 1, uniform path | 4.682654571428571 | 4.757493428571428 | 59.545449000000005 | 60.069321 | 26.766867 |
| 2, selective plan | 4.311746142857143 | 4.375325571428573 | 47.874949 | 48.32000500000001 | 17.692726 |

All four requests complete 512 input/32 output tokens on both arms; prefill
TTFT is unchanged. Independent oracles isolate the `17/16` communication work
factor from four-token verification, covering both layouts/paths, dense/routed
layers, shared work, multi-node F, TP2, transfer correction and positive energy.
A-local TP and one-time work remain singly scaled. Rust AFD49, Python AFD65 and
parity419 passed (parity284.89 s). All 12 prior execution-plan records are
unchanged; only two MTP records were appended. Ruff/rustfmt/link/whitespace checks passed.

<a id="host-cost-assessment"></a>

## Limits

These are simulated numerical effects, not GPU serving-accuracy or measured
simulator host-cost results. Static overhead analysis bounds added work to
28 multiplications per communication bundle plus scalar/branch work, independent
of layer count, batch or resident state; no retained allocation or query/cache
shape change is added. Uniform queries stay outside stride integration; selective
queries are reused across templates per sample. Prefill and nextn0 skip scaling.
No host speedup or measured regression threshold is claimed. Hashes identify but
do not archive the historical candidate source/binary.

## Reproduction

Build matching historical snapshots with the project interpreter:

```sh
PYO3_PYTHON=python/aisimulate/.venv/bin/python cargo rustc --offline --release \
  -p aisimulate-core --features extension-module --lib -- \
  -C link-arg=-undefined -C link-arg=dynamic_lookup
```

Place each built library at `python/aisimulate/src/aisimulate/_runtime.abi3.so`
in its isolated source tree. Run `aisimulate predict --stack engine --config
CASE.yaml --output-dir OUTPUT --format json` with the recorded cases.
<a id="validation"></a>

Historical validation commands were:

```sh
cargo test --offline -p aisimulate-core perfmodel::afd --lib

python/aisimulate/.venv/bin/pytest -p no:timeout \
  -c python/aisimulate/pytest.ini \
  python/aisimulate/tests/unit/sdk/test_afd_engine_materialize.py \
  python/aisimulate/tests/unit/sdk/test_afd_composition.py \
  python/aisimulate/tests/unit/sdk/test_afd_routing.py -q

python/aisimulate/.venv/bin/pytest -p no:timeout \
  -c python/aisimulate/pytest.ini \
  crates/core/parity_tests/perfmodel/test_engine_step_parity.py \
  crates/core/parity_tests/perfmodel/test_compile_engine_parity.py \
  crates/core/parity_tests/perfmodel/test_afd_parity.py \
  crates/core/parity_tests/perfmodel/test_afd_execution_plan_parity.py -q
```

CLI subprocess tests needed macOS `sysctl` process access outside the filesystem
sandbox; no numerical/source change addressed that environment limitation.
