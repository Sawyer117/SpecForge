#!/usr/bin/env python3
"""
prepare_paper_benchmarks.py — Convert the benchmark datasets used in
EAGLE-3 / DFlash papers into the unified JSONL format consumed by
eval_spec_bench.py.

Output schema (one record per line, MT-Bench / Spec-Bench compatible):
    {"question_id": int, "category": str, "turns": [str]}

`category` is the dataset name (gsm8k / humaneval / math-500 / mbpp /
aime25 / livecodebench). `turns[0]` is the prompt to feed to the target.

Usage:
    # one dataset → one file
    python docs/ascend_npu/prepare_paper_benchmarks.py \\
        --dataset gsm8k --out ./eval_data/gsm8k.jsonl

    python docs/ascend_npu/prepare_paper_benchmarks.py \\
        --dataset humaneval --out ./eval_data/humaneval.jsonl

    # multiple datasets → one combined file (for one-shot eval)
    python docs/ascend_npu/prepare_paper_benchmarks.py \\
        --dataset gsm8k,humaneval --out ./eval_data/combined.jsonl

    # smoke test: small subset per dataset
    python docs/ascend_npu/prepare_paper_benchmarks.py \\
        --dataset gsm8k --limit 20 --out ./eval_data/gsm8k_smoke.jsonl

Then feed into eval_spec_bench.py via --questions <out>.jsonl.

Currently supported (Batch 1): gsm8k, humaneval. Add more datasets by
extending DATASET_LOADERS.
"""

import argparse
import json
import sys
from collections import Counter

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: `pip install datasets` first.", file=sys.stderr)
    sys.exit(1)


# ----------------------------------------------------------------------------
# Per-dataset loaders. Each returns an iterable of (category, prompt) tuples.
#
# For instruct-tuned targets (Qwen3-8B etc.), the prompt may be passed through
# the tokenizer's chat template by eval_spec_bench.py via --chat-template.
# Dataset-specific prompt wrapping happens here when raw text alone is awkward
# (e.g. HumanEval needs an explicit "complete this function" instruction).
# ----------------------------------------------------------------------------

def _load_gsm8k():
    ds = load_dataset("openai/gsm8k", "main", split="test")
    for item in ds:
        # question is a math word problem; works fine as a user message
        yield ("gsm8k", item["question"])


def _load_humaneval():
    ds = load_dataset("openai/openai_humaneval", split="test")
    for item in ds:
        # prompt = function signature + docstring. For instruct models, wrap
        # with an explicit instruction so the chat template makes sense.
        wrapped = (
            "Complete the following Python function. Output only the function "
            "body (no extra commentary).\n\n"
            f"{item['prompt']}"
        )
        yield ("humaneval", wrapped)


DATASET_LOADERS = {
    "gsm8k": _load_gsm8k,
    "humaneval": _load_humaneval,
    # Batch 2 (todo):
    # "math-500":     _load_math500,
    # "mbpp":         _load_mbpp,
    # "aime25":       _load_aime25,
    # "livecodebench":_load_livecodebench,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        required=True,
        help=(
            "Comma-separated dataset names. Supported: "
            + ", ".join(sorted(DATASET_LOADERS))
        ),
    )
    p.add_argument("--out", required=True, help="output JSONL path")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="optional cap PER dataset (smoke test)",
    )
    args = p.parse_args()

    names = [n.strip() for n in args.dataset.split(",") if n.strip()]
    unknown = [n for n in names if n not in DATASET_LOADERS]
    if unknown:
        raise ValueError(
            f"Unknown dataset(s): {unknown}. "
            f"Supported: {sorted(DATASET_LOADERS)}"
        )

    records = []
    qid = 0
    for name in names:
        print(f"[load] {name} ...", flush=True)
        loader = DATASET_LOADERS[name]
        count_this = 0
        for cat, prompt in loader():
            records.append(
                {"question_id": qid, "category": cat, "turns": [prompt]}
            )
            qid += 1
            count_this += 1
            if args.limit is not None and count_this >= args.limit:
                break
        print(f"       {count_this} records", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    cnt = Counter(r["category"] for r in records)
    print(f"\nWrote {len(records)} records to {args.out}")
    for cat, n in cnt.most_common():
        print(f"  {cat}: {n}")


if __name__ == "__main__":
    main()
