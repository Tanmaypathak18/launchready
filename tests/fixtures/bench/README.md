# Bench fixtures

- `aiperf_profile_export.json`: keys used by `bench/run_matrix.py`, trimmed from a real AIPerf 0.13.0
  `profile_export_aiperf.json` produced against `bench/mock_server.py` (no GPU, no model) on 2026-11-03.
- `nvidia_smi_samples.csv`: hand-written rows in the format of
  `nvidia-smi --query-gpu=timestamp,index,memory.used,power.draw,utilization.gpu --format=csv,noheader,nounits`,
  including an `[N/A]` row.
- `lm_eval_results.json`: hand-written in the shape of lm-evaluation-harness 0.4.x `results_*.json`.
