## Forward Prediction Performance (advisory)

**PASS** — 8 of 10 comparisons stable: 6 faster, 2 unchanged, 0 regressions; 2 noisy, 0 invalid.

Base `23d21fa4f62ff47e742c48c26d3ed202f97308d2` vs head `23d21fa4f62ff47e742c48c26d3ed202f97308d2+working-tree`.

<details>
<summary>⚠️ Noisy comparisons (2)</summary>

| case | cache | base | head | change | rounds |
|---|---|---:|---:|---:|---:|
| `deepseek-v3.2-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8/silicon/context/bs1-isl1024-prefix0` | cold | 44.71 µs | 40.25 µs | -10.0% (-4.46 µs) | 1/5 |
| `deepseek-v3.2-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8/silicon/generation/bs128-isl1024-prefix0` | cold | 30.54 µs | 26.25 µs | -14.1% (-4.29 µs) | 1/5 |

</details>

<details>
<summary>Full matrix (8 cells)</summary>

| status | model | database | phase | cache | geometric mean | worst point | regressions | noisy | invalid |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| ⚠️ noisy | deepseek-v3.2-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | context | cold | -10.0% | -10.0% | 0 | 1 | 0 |
| ⚠️ noisy | deepseek-v3.2-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | generation | cold | -7.4% | -0.2% | 0 | 1 | 0 |
| ✅ stable | deepseek-v3.2-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | context | warm | +0.5% | +0.5% | 0 | 0 | 0 |
| ✅ stable | deepseek-v3.2-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | generation | warm | -0.0% | +0.5% | 0 | 0 | 0 |
| ✅ stable | qwen3-32b | SILICON | context | cold | -1.1% | -1.1% | 0 | 0 | 0 |
| ✅ stable | qwen3-32b | SILICON | context | warm | -0.5% | -0.5% | 0 | 0 | 0 |
| ✅ stable | qwen3-32b | SILICON | generation | cold | -10.3% | -10.3% | 0 | 0 | 0 |
| ✅ stable | qwen3-32b | SILICON | generation | warm | -0.2% | -0.2% | 0 | 0 | 0 |

</details>

This check is advisory and is not required by branch protection.
