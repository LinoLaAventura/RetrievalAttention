# RetroInfer Internal Digest (for Copilot)

## Scope
- Purpose: internal working notes for future ideation and code edits.
- Built from:
  - repository code scan (core modules + benchmark pipelines)
  - local PDF extraction using pypdf (first 10-12 pages per paper; DiskANN19 full 11 pages)
- Date: 2026-03-31

## 1) Project Structure and Responsibilities

### 1.1 Top-level architecture
- `model_hub/`: model loading and forward path orchestration.
- `attn_hub/`: attention kernels and attention variants (`full`, `RetroInfer`, optional `xattn` and `minfer` prefill).
- `cache_hub/`: KV cache backends (`flash_attn_cache`, `retroinfer_cache` offload, `retroinfer_cache_gpu`).
- `config/`: runtime config generation (`n_centroids`, budgets, NUMA core count, mode flags).
- `library/retroinfer/`: C++/CUDA extension build (`WaveBuffer`, gather/copy kernels, batch gemm+softmax kernel).
- `benchmark/`: accuracy benchmarks (RULER, LongBench, reasoning).
- `throughput_eval/`: throughput and e2e scripts against baselines (full attention, vLLM, RetroInfer, RetroInfer-GPU).
- `simple_test.py`: quick inference smoke test path.

### 1.2 Main execution path (runtime)
1. CLI parse and config:
- `simple_test.py` or benchmark script parses model + attention + budgets.
- `config/generate_config` computes cluster counts and fills method-specific runtime knobs.

2. Model load:
- `model_hub.load_model` dispatches to `LlamaModel` or `QwenModel`.
- Model class inherits `LLM` and materializes weights/layers (single GPU or auto multi-GPU sharding).

3. KV cache init:
- `LLM.generate` -> `init_kv_cache(valid_start, attn_config)`.
- Chooses backend by `attn_type` and `gpu_only`:
  - full baseline: `flash_attn_cache`
  - RetroInfer offload: `retroinfer_cache`
  - RetroInfer GPU-only: `retroinfer_cache_gpu`

4. Prefill + decode:
- Prefill attention can be `full`, `xattn`, `minfer` (optional external deps).
- Decode attention routes by attention type:
  - full: flash-attn over full cache
  - RetroInfer: cache backend `attn_func` for retrieval + estimation + static pattern composition

5. Post-prefill transition:
- full baseline calls `move_gpu` when needed.
- RetroInfer calls `prepare_cache` to set execution buffers/index state.

### 1.3 Core implementation details to remember
- `LLM.layer_prefill` and `LLM.layer_decode` implement per-layer path; `parameter_move` handles multi-GPU tensor migration and backend state rebinding.
- RetroInfer cache logic includes:
  - static pattern zone + dynamic zones
  - segmented k-means index construction/update
  - retrieval budget and estimation budget split
  - host pinned memory + async copy stream + wave buffer metadata management
- `cache_hub/kmeans.py` uses Triton kernels for assignment/update/reverse indexing/index-add.
- `library/retroinfer/setup.py` builds required extensions and expects CUDA/Cutlass integration.

### 1.4 Script-level entrypoints
- Simple smoke run: `simple_test.py`
- Throughput experiments: `throughput_eval/run.sh`
- RULER benchmark: `benchmark/ruler/ruler_run.sh`
- LongBench benchmark: `benchmark/longbench/longbench_run.sh`
- Reasoning benchmark: `benchmark/reasoning/eval.sh`

## 2) PDF Notes (4 papers)

## 2.1 DiskANN19.pdf
- Extracted title: DiskANN: Fast Accurate Billion-point Nearest ...
- Pages: 11
- Detected major sections:
  - 1 Introduction
  - 1.1 Our technical contribution
  - 2 The Vamana Graph Construction Algorithm
  - 3 DiskANN index design/layout/beam search/caching
  - 4 Evaluation
- Abstract-level core points:
  - proposes SSD-oriented ANN system with graph index (`DiskANN`)
  - introduces `Vamana` graph index
  - targets high recall + low latency + high density at billion scale
  - claims strong recall/QPS/latency under limited DRAM + SSD setup
- Auto keywords:
  - search, vamana, index, graph, algorithm, points, recall, hnsw, diskann
- Relevance to RetroInfer:
  - graph-based candidate routing and beam-style search ideas map to token retrieval over KV
  - memory hierarchy tradeoff perspective (RAM/SSD) is analogous to GPU/CPU KV placement

## 2.2 DiskANN21.pdf (FreshDiskANN)
- Extracted title: FreshDiskANN: A Fast and Accurate Graph-Based ...
- Pages: 19
- Detected major sections:
  - 1 Introduction (+ shortcomings + contributions)
  - 3 Graph-based ANNS indices
  - 4 FreshVamana (insertion/deletion/recall stability)
  - 5 FreshDiskANN system and API
- Abstract-level core points:
  - addresses dynamic ANN index updates in graph-based setup
  - supports real-time insert/delete/search while retaining high recall
  - aims to avoid expensive periodic full rebuilds
- Auto keywords:
  - index, search, points, recall, graph, streamingmerge
- Relevance to RetroInfer:
  - online index update policy parallels decode-time KV growth and periodic re-index
  - useful for designing stable retrieval quality under continuously appended context

## 2.3 DiskANN23.pdf (Filtered DiskANN)
- Extracted title: Filtered DiskANN: Graph Algorithms for Approximate Nearest ...
- Pages: 11
- Detected major sections:
  - 1 Introduction (filtered ANNS framing)
  - 3 FilteredVamana
  - 4 StitchedVamana
  - 5 Evaluation
- Abstract-level core points:
  - focuses on filtered ANN (label-constrained nearest neighbors)
  - introduces graph methods that account for vector geometry + metadata labels
  - reports latency/recall improvements on filtered queries
- Auto keywords:
  - algorithms, filters, recall, graph, query, filteredvamana
- Relevance to RetroInfer:
  - potential analogy: add token/head/layer constraints as filter conditions before retrieval
  - can inspire "metadata-aware" token retrieval policies in long-context decoding

## 2.4 RetroInfer.pdf
- Extracted title: RetroInfer: A Vector-Storage Approach for ...
- Pages: 17
- Detected major sections:
  - 1 Introduction
  - 2 Background and Motivation
  - 3 When ANNS Meets Sparse Attention
  - 4 RetroInfer (overview, wave index, wave buffer, prefill latency)
  - 5 Evaluation
- Abstract-level core points:
  - reframes KV cache as vector storage and uses sparse retrieval for long-context inference
  - introduces wave index (attention-aware vector index)
  - introduces wave buffer (GPU/CPU placement + overlap of transfer/compute)
  - targets speedup with full-attention-level accuracy retention
- Auto keywords:
  - attention, cache, index, retroinfer, memory, vectors, tokens, wave, buffer
- Code mapping in this repo:
  - wave index/update logic: `cache_hub/retroinfer_cache.py`, `cache_hub/retroinfer_cache_gpu.py`, `cache_hub/kmeans.py`
  - attention entry: `attn_hub/retroinfer_attn.py`
  - execution and orchestration: `model_hub/LLM.py`, `model_hub/llama.py`, `model_hub/qwen.py`
  - kernel support: `library/retroinfer/retroinfer_kernels/*`

## 3) Cross-paper Synthesis for Engineering

### 3.1 Shared technical axis
- Graph-based retrieval quality vs compute/memory budget.
- Dynamic update stability under streaming additions.
- Tiered memory hierarchy optimization.
- Candidate pruning and re-ranking tradeoff.

### 3.2 Directly actionable design directions in this codebase
- Dynamic decode-time index refresh schedule:
  - adapt update cadence to generation uncertainty or entropy.
- Filtered retrieval over KV candidates:
  - use token metadata (position bucket, segment, head affinity) to constrain candidate set.
- Hybrid scoring:
  - fuse ANN distance with lightweight attention prior before final gather.
- Auto budget controller:
  - per-step adaptive `retrieval_budget` and `estimation_budget` under latency SLA.

## 4) Reliability Notes
- PDF summaries are based on local text extraction and first 10-12 pages sampling for long papers (except DiskANN19 fully covered due 11 pages).
- For formula-level exactness, re-run extraction over all pages and/or manually verify quoted equations/claims before publication-grade writing.

## 5) Quick Reference: Key Files
- Runtime API and inference loop:
  - `simple_test.py`
  - `model_hub/LLM.py`
- Model adapters:
  - `model_hub/llama.py`
  - `model_hub/qwen.py`
- Attention methods:
  - `attn_hub/full_attn.py`
  - `attn_hub/retroinfer_attn.py`
  - `attn_hub/xattn.py`
  - `attn_hub/minfer.py`
- Cache backends and indexing:
  - `cache_hub/flash_attn_cache.py`
  - `cache_hub/retroinfer_cache.py`
  - `cache_hub/retroinfer_cache_gpu.py`
  - `cache_hub/kmeans.py`
- Config and knobs:
  - `config/config.py`
- Native kernels:
  - `library/retroinfer/setup.py`
  - `library/retroinfer/retroinfer_kernels/__init__.py`
- Benchmarks:
  - `benchmark/ruler/ruler_run.sh`
  - `benchmark/longbench/longbench_run.sh`
  - `benchmark/reasoning/eval.sh`
  - `throughput_eval/run.sh`

## 6) Internal TODO Hooks
- Add per-function call graph for `retroinfer_cache.attn_func` path.
- Build parameter sensitivity table from scripts (retrieval/estimation/cache ratio).
- Create experiment matrix template for idea validation.
