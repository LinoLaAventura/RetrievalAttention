# Reasoning Experiment Archive (2026-04-04)

This document records the archived reasoning benchmark runs for AIME24.

## Scope

Archived run groups:
- `outputs_compare_30` (AIME24 full 30 samples, `n_sampling=1`)
- `outputs_paper_like_compare30_daemon` (AIME24 full 30 samples, `n_sampling=8`)

Archived logs:
- `benchmark/reasoning/logs/full_bs1_aime24_30.log`
- `benchmark/reasoning/logs/retro_bs1_aime24_30.log`
- `benchmark/reasoning/logs/full_paperlike_daemon.log`
- `benchmark/reasoning/logs/retro_paperlike_daemon.log`

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
