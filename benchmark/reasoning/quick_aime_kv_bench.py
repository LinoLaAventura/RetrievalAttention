import argparse
import glob
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch

# Keep imports local to reasoning benchmark package
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
sys.path.append(str(CURRENT_DIR))
sys.path.append(str(PROJECT_ROOT))

from config import generate_config  # noqa: E402
from data_loader import load_data  # noqa: E402
from model_utils import load_lm_and_tokenizer  # noqa: E402
from parser import parse_question  # noqa: E402
from utils import construct_prompt, set_seed  # noqa: E402


@dataclass
class RunResult:
    attention: str
    pass_at_1: float
    num_scores: int
    metrics_path: str


@dataclass
class ThroughputResult:
    attention: str
    samples: int
    decode_tokens: int
    avg_input_tokens: float
    avg_elapsed_s: float
    avg_tokens_per_s: float


def _run_accuracy_eval(args, attention: str) -> RunResult:
    output_dir = args.output_dir
    cmd = [
        sys.executable,
        "-u",
        "math_eval.py",
        "--data_names",
        "aime24",
        "--model_name_or_path",
        args.model_name_or_path,
        "--output_dir",
        output_dir,
        "--split",
        "test",
        "--prompt_type",
        "orz",
        "--num_test_sample",
        str(args.accuracy_samples),
        "--max_tokens_per_call",
        str(args.accuracy_max_tokens),
        "--seed",
        str(args.seed),
        "--n_sampling",
        "1",
        "--temperature",
        "0.6",
        "--top_p",
        "0.95",
        "--top_k",
        "20",
        "--start",
        "0",
        "--end",
        "-1",
        "--save_outputs",
        "--overwrite",
        "--attn_type",
        attention,
        "--do_sample",
        "--dtype",
        args.dtype,
        "--batch_size",
        "1",
        "--retrieval_budget",
        str(args.retrieval_budget),
        "--estimation_budget",
        str(args.estimation_budget),
    ]
    if args.gpu_only:
        cmd.append("--gpu_only")

    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    print(f"\n[Accuracy] Running {attention} ...")
    subprocess.run(cmd, cwd=str(CURRENT_DIR), env=env, check=True)

    pattern = (
        f"aime24/test_orz_{args.accuracy_samples}_seed{args.seed}_{attention}"
        f"_budget{args.retrieval_budget}_es{args.estimation_budget}_*_orz_metrics.json"
    )
    candidate_roots = [
        CURRENT_DIR / output_dir,
        CURRENT_DIR / "outputs" / output_dir,
    ]
    metrics_files: List[str] = []
    for root in candidate_roots:
        metrics_files.extend(glob.glob(str(root / pattern)))
    metrics_files = sorted(set(metrics_files))
    if not metrics_files:
        raise FileNotFoundError(
            f"No metrics file found for {attention}. Searched: "
            f"{[str(r / pattern) for r in candidate_roots]}"
        )

    metrics_path = metrics_files[-1]
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    return RunResult(
        attention=attention,
        pass_at_1=float(metrics.get("pass@1", 0.0)),
        num_scores=int(metrics.get("num_scores", 0)),
        metrics_path=metrics_path,
    )


def _build_prompts(args, sample_count: int) -> List[str]:
    data_args = argparse.Namespace(num_test_sample=-1)
    examples = load_data("aime24", "test", "./data", data_args)
    examples = examples[:sample_count]

    prompts: List[str] = []
    for example in examples:
        ex = dict(example)
        ex["question"] = parse_question(ex, "aime24")
        prompt = construct_prompt(ex, "aime24", argparse.Namespace(prompt_type="orz", num_shots=0, adapt_few_shot=False))
        if args.throughput_prompt_repeat > 1:
            prompt = "\n\n".join([prompt] * args.throughput_prompt_repeat)
        prompts.append(prompt)
    return prompts


def _run_throughput_eval(args, prompts: List[str], attention: str) -> ThroughputResult:
    print(f"\n[Throughput] Running {attention} ...")

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    llm, tokenizer = load_lm_and_tokenizer(
        model_path=args.model_name_or_path,
        max_len=max(args.max_length, args.decode_tokens + 2048),
        dtype=dtype,
        device="auto",
    )

    elapsed_list: List[float] = []
    tps_list: List[float] = []
    in_len_list: List[int] = []

    for prompt in prompts:
        inputs = tokenizer([prompt], return_tensors="pt", padding=True, add_special_tokens=True)
        input_ids = inputs.input_ids.to(llm.layers[0].device)
        attention_masks = inputs.attention_mask.to(llm.layers[0].device)

        in_len = int(input_ids.shape[1])
        in_len_list.append(in_len)

        attn_config = generate_config(
            args.model_name_or_path,
            in_len,
            attention,
            retrieval_budget=args.retrieval_budget,
            estimation_budget=args.estimation_budget,
            cache_ratio=args.cache_ratio,
            use_cuda_graph=args.use_cuda_graph,
            gpu_only=args.gpu_only,
        )
        # Match existing reasoning path behavior
        attn_config["RetroInfer"]["buffer_cluster_num"] = args.buffer_cluster_num

        torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = llm.generate(
            attention_type=attention,
            inputs_ids=input_ids,
            attention_masks=attention_masks,
            max_new_length=args.decode_tokens,
            attn_config=attn_config,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            ignore_eos=True,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        generated_tokens = len(outputs[0]) if outputs else 0
        tps = generated_tokens / max(elapsed, 1e-8)

        elapsed_list.append(elapsed)
        tps_list.append(tps)

    # release model before next attention run
    del llm
    torch.cuda.empty_cache()

    return ThroughputResult(
        attention=attention,
        samples=len(prompts),
        decode_tokens=args.decode_tokens,
        avg_input_tokens=sum(in_len_list) / max(len(in_len_list), 1),
        avg_elapsed_s=sum(elapsed_list) / max(len(elapsed_list), 1),
        avg_tokens_per_s=sum(tps_list) / max(len(tps_list), 1),
    )


def main():
    parser = argparse.ArgumentParser(description="Quick AIME KV/throughput benchmark for Full vs RetroInfer")
    parser.add_argument("--model_name_or_path", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--max_length", type=int, default=65536)

    # Accuracy phase (short run)
    parser.add_argument("--accuracy_samples", type=int, default=4)
    parser.add_argument("--accuracy_max_tokens", type=int, default=768)
    parser.add_argument("--output_dir", type=str, default="./outputs_quick_kv_bench")

    # Throughput phase (KV stress)
    parser.add_argument("--throughput_samples", type=int, default=2)
    parser.add_argument("--decode_tokens", type=int, default=1024)
    parser.add_argument("--throughput_prompt_repeat", type=int, default=1)

    # RetroInfer knobs
    parser.add_argument("--retrieval_budget", type=float, default=0.018)
    parser.add_argument("--estimation_budget", type=float, default=0.232)
    parser.add_argument("--cache_ratio", type=float, default=0.0)
    parser.add_argument("--buffer_cluster_num", type=int, default=200)
    parser.add_argument("--gpu_only", action="store_true")
    parser.add_argument("--use_cuda_graph", action="store_true")

    args = parser.parse_args()
    set_seed(args.seed)

    print("=== Quick AIME KV Bench ===")
    print(json.dumps(vars(args), indent=2))

    # 1) Short accuracy sanity check
    acc_full = _run_accuracy_eval(args, "Full_Flash_Attn")
    acc_retro = _run_accuracy_eval(args, "RetroInfer")

    # 2) Fixed-length throughput stress (ignore EOS)
    prompts = _build_prompts(args, args.throughput_samples)
    t_full = _run_throughput_eval(args, prompts, "Full_Flash_Attn")
    t_retro = _run_throughput_eval(args, prompts, "RetroInfer")

    speedup = t_retro.avg_tokens_per_s / max(t_full.avg_tokens_per_s, 1e-8)

    print("\n=== Summary ===")
    print(
        f"Accuracy pass@1 (n={args.accuracy_samples}): "
        f"Full={acc_full.pass_at_1:.1f}, RetroInfer={acc_retro.pass_at_1:.1f}"
    )
    print(
        f"Avg input tokens in throughput phase: "
        f"Full={t_full.avg_input_tokens:.0f}, RetroInfer={t_retro.avg_input_tokens:.0f}"
    )
    print(
        f"Throughput tokens/s (fixed decode {args.decode_tokens}, n={args.throughput_samples}): "
        f"Full={t_full.avg_tokens_per_s:.2f}, RetroInfer={t_retro.avg_tokens_per_s:.2f}, speedup={speedup:.2f}x"
    )
    print("\nArtifacts:")
    print(f"- Full metrics: {acc_full.metrics_path}")
    print(f"- Retro metrics: {acc_retro.metrics_path}")


if __name__ == "__main__":
    main()
