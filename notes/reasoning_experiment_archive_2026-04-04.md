# Reasoning Experiment Archive (2026-04-04)

This document records the archived reasoning benchmark runs for AIME24.

## Scope

Archived run groups:
- `outputs_compare_30` (AIME24 full 30 samples, `n_sampling=1`)
- `outputs_paper_like_compare30_daemon` (AIME24 full 30 samples, `n_sampling=8`)
- `outputs_quick_kv_bench` (AIME24 quick KV throughput stress runs, `n_sampling=1`)

Archived logs:
- `benchmark/reasoning/logs/full_bs1_aime24_30.log`
- `benchmark/reasoning/logs/retro_bs1_aime24_30.log`
- `benchmark/reasoning/logs/full_paperlike_daemon.log`
- `benchmark/reasoning/logs/retro_paperlike_daemon.log`
- `benchmark/reasoning/logs/quick_kv_bench_r1.log`
- `benchmark/reasoning/logs/quick_kv_bench_r2.log`
- `benchmark/reasoning/logs/quick_kv_bench_r3.log`
- `benchmark/reasoning/logs/quick_kv_bench_r4.log`

## Environment

- GPU: `NVIDIA L20 (46068 MiB)`
- Driver: `570.195.03`
- Model: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`
- Attention types: `Full_Flash_Attn`, `RetroInfer`

## Key Parameters

Common:
- `temperature=0.6`
- `top_p=0.95`
- `top_k=20`
- `do_sample=True`
- `dtype=fp16`
- `batch_size=1`
- `retrieval_budget=0.018`
- `estimation_budget=0.232`

Group-specific:
- `outputs_compare_30`: `n_sampling=1`, `max_tokens_per_call=32768`
- `outputs_paper_like_compare30_daemon`: `n_sampling=8`, `max_tokens_per_call=16384`
- `outputs_quick_kv_bench`: `n_sampling=1` for quick sanity accuracy; fixed-length throughput stress with `ignore_eos=True`

## Results Summary

### 1) outputs_compare_30 (AIME24, n_sampling=1)
- Full_Flash_Attn: `pass@1=6.7` (num_scores=30)
- RetroInfer: `pass@1=6.7` (num_scores=30)
- Runtime from logs:
  - Full: `22:17` (`44.58 s/it`)
  - RetroInfer: `25:21` (`50.71 s/it`)

### 2) outputs_paper_like_compare30_daemon (AIME24, n_sampling=8)
- Full_Flash_Attn: `pass@8=76.7` (metrics key in file is `pass@1`, num_scores=240)
- RetroInfer: `pass@8=73.3` (metrics key in file is `pass@1`, num_scores=240)
- Runtime from logs:
  - Full: `17:19:50` (`259.96 s/it`)
  - RetroInfer: `18:35:19` (`278.83 s/it`)

### 3) outputs_quick_kv_bench (AIME24 quick KV stress, n_sampling=1)
This group is used to quickly probe throughput under fixed decode lengths and increasingly long effective context.

`quick_kv_bench_r1`
- Parameters: `accuracy_samples=4`, `accuracy_max_tokens=2048`, `throughput_samples=2`, `decode_tokens=3072`, `gpu_only=True`, `cache_ratio=0.0`, `throughput_prompt_repeat=1`
- Accuracy sanity: Full `pass@1=0.0`, RetroInfer `pass@1=0.0`
- Throughput: Full `45.19 tok/s`, RetroInfer `45.88 tok/s`, speedup `1.02x`
- Wall time: `15:33.33`

`quick_kv_bench_r2`
- Parameters: `accuracy_samples=4`, `accuracy_max_tokens=2048`, `throughput_samples=2`, `decode_tokens=6144`, `gpu_only=True`, `cache_ratio=0.05`, `throughput_prompt_repeat=1`
- Accuracy sanity: Full `pass@1=0.0`, RetroInfer `pass@1=0.0`
- Throughput: Full `44.58 tok/s`, RetroInfer `45.27 tok/s`, speedup `1.02x`
- Wall time: `20:07.98`

`quick_kv_bench_r3`
- Parameters: `accuracy_samples=4`, `accuracy_max_tokens=2048`, `throughput_samples=2`, `decode_tokens=4096`, `gpu_only=True`, `cache_ratio=0.05`, `throughput_prompt_repeat=12`
- Accuracy sanity: Full `pass@1=0.0`, RetroInfer `pass@1=0.0`
- Avg input tokens in throughput phase: `3036`
- Throughput: Full `43.71 tok/s`, RetroInfer `44.40 tok/s`, speedup `1.02x`
- Wall time: `17:13.61`

`quick_kv_bench_r4`
- Parameters: `accuracy_samples=4`, `accuracy_max_tokens=2048`, `throughput_samples=2`, `decode_tokens=4096`, `gpu_only=True`, `cache_ratio=0.05`, `throughput_prompt_repeat=80`
- Accuracy sanity: Full `pass@1=0.0`, RetroInfer `pass@1=0.0`
- Avg input tokens in throughput phase: `20240`
- Throughput: Full `37.41 tok/s`, RetroInfer `39.64 tok/s`, speedup `1.06x`
- Wall time: `18:08.04`
- Notes: run crosses long context regime and no longer stays at `n_centroids=0`; tokenizer warning appears for input lengths exceeding model nominal 16k (`21280 > 16384`).

## Archived Artifacts

`outputs_compare_30`
- `benchmark/reasoning/outputs/outputs_compare_30/aime24/test_orz_-1_seed2025_Full_Flash_Attn_budget0.018_es0.232_04-02_05-29.jsonl`
- `benchmark/reasoning/outputs/outputs_compare_30/aime24/test_orz_-1_seed2025_Full_Flash_Attn_budget0.018_es0.232_04-02_05-29_orz_metrics.json`
- `benchmark/reasoning/outputs/outputs_compare_30/aime24/test_orz_-1_seed2025_RetroInfer_budget0.018_es0.232_04-02_05-54.jsonl`
- `benchmark/reasoning/outputs/outputs_compare_30/aime24/test_orz_-1_seed2025_RetroInfer_budget0.018_es0.232_04-02_05-54_orz_metrics.json`

`outputs_paper_like_compare30_daemon`
- `benchmark/reasoning/outputs/outputs_paper_like_compare30_daemon/aime24/test_orz_-1_seed2025_Full_Flash_Attn_budget0.018_es0.232_04-02_17-33.jsonl`
- `benchmark/reasoning/outputs/outputs_paper_like_compare30_daemon/aime24/test_orz_-1_seed2025_Full_Flash_Attn_budget0.018_es0.232_04-02_17-33_orz_metrics.json`
- `benchmark/reasoning/outputs/outputs_paper_like_compare30_daemon/aime24/test_orz_-1_seed2025_RetroInfer_budget0.018_es0.232_04-03_10-53.jsonl`
- `benchmark/reasoning/outputs/outputs_paper_like_compare30_daemon/aime24/test_orz_-1_seed2025_RetroInfer_budget0.018_es0.232_04-03_10-53_orz_metrics.json`

`outputs_quick_kv_bench`
- `benchmark/reasoning/outputs/outputs_quick_kv_bench/aime24/test_orz_4_seed2025_Full_Flash_Attn_budget0.018_es0.232_04-04_18-54.jsonl`
- `benchmark/reasoning/outputs/outputs_quick_kv_bench/aime24/test_orz_4_seed2025_Full_Flash_Attn_budget0.018_es0.232_04-04_18-54_orz_metrics.json`
- `benchmark/reasoning/outputs/outputs_quick_kv_bench/aime24/test_orz_4_seed2025_RetroInfer_budget0.018_es0.232_04-04_18-58.jsonl`
- `benchmark/reasoning/outputs/outputs_quick_kv_bench/aime24/test_orz_4_seed2025_RetroInfer_budget0.018_es0.232_04-04_18-58_orz_metrics.json`
- `benchmark/reasoning/outputs/outputs_quick_kv_bench/aime24/test_orz_4_seed2025_Full_Flash_Attn_budget0.018_es0.232_04-04_19-10.jsonl`
- `benchmark/reasoning/outputs/outputs_quick_kv_bench/aime24/test_orz_4_seed2025_Full_Flash_Attn_budget0.018_es0.232_04-04_19-10_orz_metrics.json`
- `benchmark/reasoning/outputs/outputs_quick_kv_bench/aime24/test_orz_4_seed2025_RetroInfer_budget0.018_es0.232_04-04_19-15.jsonl`
- `benchmark/reasoning/outputs/outputs_quick_kv_bench/aime24/test_orz_4_seed2025_RetroInfer_budget0.018_es0.232_04-04_19-15_orz_metrics.json`
- `benchmark/reasoning/outputs/outputs_quick_kv_bench/aime24/test_orz_4_seed2025_Full_Flash_Attn_budget0.018_es0.232_04-04_19-32.jsonl`
- `benchmark/reasoning/outputs/outputs_quick_kv_bench/aime24/test_orz_4_seed2025_Full_Flash_Attn_budget0.018_es0.232_04-04_19-32_orz_metrics.json`
- `benchmark/reasoning/outputs/outputs_quick_kv_bench/aime24/test_orz_4_seed2025_RetroInfer_budget0.018_es0.232_04-04_19-36.jsonl`
- `benchmark/reasoning/outputs/outputs_quick_kv_bench/aime24/test_orz_4_seed2025_RetroInfer_budget0.018_es0.232_04-04_19-36_orz_metrics.json`
- `benchmark/reasoning/outputs/outputs_quick_kv_bench/aime24/test_orz_4_seed2025_Full_Flash_Attn_budget0.018_es0.232_04-04_23-14.jsonl`
- `benchmark/reasoning/outputs/outputs_quick_kv_bench/aime24/test_orz_4_seed2025_Full_Flash_Attn_budget0.018_es0.232_04-04_23-14_orz_metrics.json`
- `benchmark/reasoning/outputs/outputs_quick_kv_bench/aime24/test_orz_4_seed2025_RetroInfer_budget0.018_es0.232_04-04_23-18.jsonl`
- `benchmark/reasoning/outputs/outputs_quick_kv_bench/aime24/test_orz_4_seed2025_RetroInfer_budget0.018_es0.232_04-04_23-18_orz_metrics.json`
