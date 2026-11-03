# Test fixtures

Saved from the Hugging Face Hub on 2026-11-02 so tests run with no network. No weights were
downloaded: only `config.json`, model info and safetensors headers.

## `configs/`: raw `config.json`

| File | Source repo | Revision | Why |
|---|---|---|---|
| `unsloth__Meta-Llama-3.1-8B.json` | unsloth/Meta-Llama-3.1-8B | e9a141a2091ea561b96483212645a2a05e6f99fc | 6.5 reference (mirror of gated meta-llama/Llama-3.1-8B) |
| `unsloth__Meta-Llama-3.1-70B.json` | unsloth/Meta-Llama-3.1-70B | 1b7306651142d0cc65d993076a250a6a82cf046c | 6.5 reference (mirror of gated meta-llama/Llama-3.1-70B) |
| `NousResearch__Llama-2-7b-hf.json` | NousResearch/Llama-2-7b-hf | 8efe6c9b93655b934e27bd9981e3ec13e55aee9d | 6.5 reference, no GQA (mirror of gated meta-llama/Llama-2-7b-hf) |
| `deepseek-ai__DeepSeek-V3.json` | deepseek-ai/DeepSeek-V3 | e815299b0bcbac849fa540c768ef21845365c9eb | 6.5 reference, MLA |
| `Qwen__Qwen3-4B.json` | Qwen/Qwen3-4B | 1cfa9a7208912126459214e8b04321603b3df60c | Dense GQA |
| `Qwen__Qwen3-30B-A3B.json` | Qwen/Qwen3-30B-A3B | ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 | MoE |
| `openai__gpt-oss-20b.json` | openai/gpt-oss-20b | 6cee5e81ee83917806bbde320786a8fb61efebee | Sliding window via `layer_types` |
| `ibm-granite__granite-4.0-h-tiny.json` | ibm-granite/granite-4.0-h-tiny | 791e0d3d28c86e106c9b6e0b4cecdee0375b6124 | Hybrid Mamba MoE (`layer_types`) |
| `nvidia__NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16.json` | nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 | a9904d24bcc1d289a1950fa9d2b978c47cf903b9 | Hybrid Mamba MoE (`layers_block_type`) |

## `hub/`: launchready hub cache entries

Model info, safetensors parameter counts by dtype (from headers) and `config.json`, in the
format `launchready.hub` caches under `~/.cache/launchready/hub/`. Used to test `launchready fit`
end to end offline. Same revisions as above for Qwen3-4B, granite-4.0-h-tiny, Nemotron 3.5
Lightning and DeepSeek-V3.
