## Forward Prediction Performance (advisory)

**PASS** — 9 of 10 comparisons stable: 6 faster, 3 unchanged, 0 regressions; 1 noisy, 0 invalid.

Base `4a934b0756c28b53b42027b955c23148ee1f738c` vs head `4a934b0756c28b53b42027b955c23148ee1f738c+working-tree`.

<details>
<summary>⚠️ Noisy comparisons (1)</summary>

| case | cache | base | head | change | rounds |
|---|---|---:|---:|---:|---:|
| `deepseek-v3.2-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8/silicon/generation/bs1-isl1024-prefix0` | cold | 25.42 µs | 24.54 µs | -3.4% (-0.87 µs) | 1/5 |

</details>

<details>
<summary>Full matrix (8 cells)</summary>

| status | model | database | phase | cache | geometric mean | worst point | regressions | noisy | invalid |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| ⚠️ noisy | deepseek-v3.2-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | generation | cold | -9.3% | -3.4% | 0 | 1 | 0 |
| ✅ stable | deepseek-v3.2-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | context | cold | -4.4% | -4.4% | 0 | 0 | 0 |
| ✅ stable | deepseek-v3.2-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | context | warm | -0.9% | -0.9% | 0 | 0 | 0 |
| ✅ stable | deepseek-v3.2-b200_sxm-vllm-0.24.0-tp8-pp1-adp1-mtp1-ep8 | SILICON | generation | warm | +0.4% | +1.2% | 0 | 0 | 0 |
| ✅ stable | qwen3-32b | SILICON | context | cold | +3.7% | +3.7% | 0 | 0 | 0 |
| ✅ stable | qwen3-32b | SILICON | context | warm | +0.4% | +0.4% | 0 | 0 | 0 |
| ✅ stable | qwen3-32b | SILICON | generation | cold | -0.9% | -0.9% | 0 | 0 | 0 |
| ✅ stable | qwen3-32b | SILICON | generation | warm | -0.1% | -0.1% | 0 | 0 | 0 |

</details>

This check is advisory and is not required by branch protection.
