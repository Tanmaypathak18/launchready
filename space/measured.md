## Measured results so far

Three models, measured with vLLM on 2× Tesla T4 at 4,096 tokens, against what LaunchReady projected.

| Model | Rev. | Type | Weights GiB, projected / measured | Max users, projected / measured |
|---|---|---|--:|--:|
| Qwen/Qwen3-4B | `1cfa9a7` | Dense | 7.49 / 7.56 | 7.1 / 7.4 |
| allenai/OLMoE-1B-7B-0924-Instruct | `7f1c97f` | MoE | 12.89 / 12.90 | 20.2 / 21.5 |
| ibm-granite/granite-4.0-h-tiny | `791e0d3` | Hybrid Mamba | 12.92 / 12.96 | 118.1 / 50.6 |

Measured on 2026-11-03: Tesla T4 (2 in the machine), driver 580.178.04, CUDA 13.0, vLLM 0.32.0. Measured values come from vLLM's startup log.

![Projected against measured weights, KV-cache tokens and max concurrency for the three models]({RAW}/{REF}/results/memory_validation.png)

[Validation data (CSV)]({GITHUB}/blob/{REF}/results/memory_validation.csv) · [Environment]({GITHUB}/blob/{REF}/results/environment.json) · [All results]({GITHUB}/tree/{REF}/results)
