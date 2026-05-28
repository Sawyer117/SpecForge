#!/usr/bin/env python3
"""
eval_spec_bench.py — Acceptance length (τ) evaluation for a trained DFlash draft.

Loads one draft checkpoint + the target, runs spec_generate on a JSONL of
prompts (Spec-Bench compatible), and reports mean τ per category + overall.

Spec-Bench data:
    git clone https://github.com/hemingkx/Spec-Bench
    # → data/spec_bench/question.jsonl  (480 prompts × 6 categories × 80 each)

Each record:
    {"question_id": int, "category": str, "turns": [str, ...]}

We use turns[0] as the prompt (single-turn). Multi-turn handling can be added
later if needed.

Example:
    python docs/ascend_npu/eval_spec_bench.py \\
        --draft-ckpt ./outputs/clip-B-fixed/epoch_0_step_500 \\
        --target-model /share/canada_group_folder/ckpt/Qwen3-8B \\
        --questions /path/to/Spec-Bench/data/spec_bench/question.jsonl \\
        --chat-template \\
        --out-jsonl ./outputs/eval-clip-B-fixed.jsonl

For a quick smoke test:
    ... --limit-per-category 5

Acceptance length τ is hardware-independent; the per-category mean you get
here is directly comparable to numbers reported in EAGLE-3 / DFlash papers
(modulo dataset choice). Use 3+ seeds + bootstrap CI for paper-grade results.
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict

import torch
import torch_npu  # noqa: F401  (triggers transfer_to_npu)
from transformers import AutoModelForCausalLM, AutoTokenizer

from specforge.modeling.draft.dflash import DFlashDraftModel


# ----------------------------------------------------------------------------
# NPU patch — same as in compare_decoding.py.
# DFlashDraftModel.spec_generate's bool.cumprod() crashes on NPU because
# aclnnCumprod doesn't support DT_BOOL. Cast bool -> int first.
# Patched method also stashes per-block acceptance lengths on the instance
# (self._last_acceptance_lengths) so the caller can read τ.
# ----------------------------------------------------------------------------
@torch.inference_mode()
def _spec_generate_npu_safe(self, target, input_ids, max_new_tokens,
                            stop_token_ids, temperature):
    from transformers import DynamicCache
    from specforge.modeling.draft.dflash import sample, extract_context_feature

    self.eval()
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens

    block_size = self.block_size
    output_ids = torch.full(
        (1, max_length + block_size),
        self.mask_token_id,
        dtype=torch.long,
        device=target.device,
    )
    position_ids = torch.arange(
        output_ids.shape[1], device=target.device,
    ).unsqueeze(0)

    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()

    # Prefill
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True,
    )
    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens:num_input_tokens + 1] = sample(
        output.logits, temperature,
    )
    target_hidden = extract_context_feature(
        output.hidden_states, self.target_layer_ids,
    )

    acceptance_lengths = []
    start = input_ids.shape[1]
    while start < max_length:
        block_output_ids = output_ids[:, start:start + block_size].clone()
        block_position_ids = position_ids[:, start:start + block_size]
        noise_embedding = target.model.embed_tokens(block_output_ids)
        draft_logits = target.lm_head(
            self(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=position_ids[
                    :, past_key_values_draft.get_seq_length():start + block_size
                ],
                past_key_values=past_key_values_draft,
                use_cache=True,
                is_causal=False,
            )[:, -block_size + 1:, :]
        )
        past_key_values_draft.crop(start)
        block_output_ids[:, 1:] = sample(draft_logits)

        output = target(
            block_output_ids,
            position_ids=block_position_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True,
        )

        posterior = sample(output.logits, temperature)
        # NPU fix: cast bool -> int before cumprod
        acceptance_length = (
            (block_output_ids[:, 1:] == posterior[:, :-1])
            .int()
            .cumprod(dim=1)
            .sum(dim=1)[0]
            .item()
        )
        output_ids[:, start:start + acceptance_length + 1] = block_output_ids[
            :, :acceptance_length + 1
        ]
        output_ids[:, start + acceptance_length + 1] = posterior[
            :, acceptance_length
        ]
        start += acceptance_length + 1
        past_key_values_target.crop(start)
        target_hidden = extract_context_feature(
            output.hidden_states, self.target_layer_ids,
        )[:, :acceptance_length + 1, :]
        acceptance_lengths.append(acceptance_length + 1)
        if stop_token_ids is not None and any(
            stop_token_id in output_ids[:, num_input_tokens:]
            for stop_token_id in stop_token_ids
        ):
            break

    output_ids = output_ids[:, :max_length]
    output_ids = output_ids[:, output_ids[0] != self.mask_token_id]
    if stop_token_ids is not None and len(stop_token_ids) > 0:
        stop_ids_t = torch.tensor(stop_token_ids, device=output_ids.device)
        stop_token_indices = torch.isin(
            output_ids[0][num_input_tokens:], stop_ids_t,
        ).nonzero(as_tuple=True)[0]
        if stop_token_indices.numel() > 0:
            output_ids = output_ids[
                :, :num_input_tokens + stop_token_indices[0] + 1
            ]
    self._last_acceptance_lengths = acceptance_lengths
    return output_ids


DFlashDraftModel.spec_generate = _spec_generate_npu_safe


# ----------------------------------------------------------------------------
# Spec-Bench loader
# ----------------------------------------------------------------------------
def load_questions(path):
    """Load Spec-Bench (or MT-Bench-style) jsonl.
    Returns list of (category, prompt_str). Uses turns[0]; ignores subsequent turns.
    """
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cat = obj.get("category", "unknown")
            turns = obj.get("turns", [])
            if not turns:
                continue
            out.append((cat, turns[0]))
    return out


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--draft-ckpt", required=True,
                   help="trained DFlash draft checkpoint directory")
    p.add_argument("--target-model", required=True,
                   help="HF target model path (e.g. Qwen3-8B)")
    p.add_argument("--questions", required=True,
                   help="Spec-Bench question.jsonl (or compatible)")
    p.add_argument("--device", default="npu:0")
    p.add_argument("--max-new-tokens", type=int, default=128,
                   help="max tokens to generate per prompt")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0.0 = greedy (recommended for τ comparison)")
    p.add_argument("--chat-template", action="store_true",
                   help="apply tokenizer's chat template (use for instruct models)")
    p.add_argument("--limit", type=int, default=None,
                   help="optional: total prompt limit (smoke test)")
    p.add_argument("--limit-per-category", type=int, default=None,
                   help="optional: limit prompts per category (smoke test)")
    p.add_argument("--out-jsonl", default=None,
                   help="optional: write per-prompt results to jsonl")
    p.add_argument("--shard-rank", type=int, default=0,
                   help="data-parallel shard: this process's rank (0-indexed)")
    p.add_argument("--num-shards", type=int, default=1,
                   help="data-parallel shard: total number of shards. "
                        "Each shard processes prompts where idx %% num_shards == shard_rank. "
                        "Launch num_shards instances in parallel, one per NPU, "
                        "for near-linear speedup. Aggregate via cat *.jsonl.")
    args = p.parse_args()
    if not (0 <= args.shard_rank < args.num_shards):
        raise SystemExit(
            f"--shard-rank ({args.shard_rank}) must be in [0, {args.num_shards})"
        )

    print(f"[load] target: {args.target_model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model, trust_remote_code=True,
    )
    target = AutoModelForCausalLM.from_pretrained(
        args.target_model, dtype=torch.bfloat16, trust_remote_code=True,
    ).to(args.device).eval()

    print(f"[load] draft : {args.draft_ckpt}", flush=True)
    draft = DFlashDraftModel.from_pretrained(
        args.draft_ckpt, dtype=torch.bfloat16,
    ).to(args.device).eval()

    print(f"[load] questions: {args.questions}", flush=True)
    questions = load_questions(args.questions)
    print(f"       {len(questions)} questions loaded", flush=True)

    # Per-category subsample for smoke tests
    if args.limit_per_category:
        cat_count = defaultdict(int)
        filtered = []
        for cat, prompt in questions:
            if cat_count[cat] < args.limit_per_category:
                filtered.append((cat, prompt))
                cat_count[cat] += 1
        questions = filtered
    if args.limit:
        questions = questions[:args.limit]

    # Data-parallel sharding: keep only prompts whose global index hits this rank.
    # Same filtering AFTER limit/limit-per-category so the smoke-test subsets
    # behave consistently.
    if args.num_shards > 1:
        all_count = len(questions)
        questions = [
            q for i, q in enumerate(questions)
            if i % args.num_shards == args.shard_rank
        ]
        print(
            f"       shard {args.shard_rank}/{args.num_shards}: "
            f"{len(questions)}/{all_count} prompts",
            flush=True,
        )
    else:
        print(f"       using {len(questions)} after filters", flush=True)

    eos_ids = (
        [tokenizer.eos_token_id]
        if tokenizer.eos_token_id is not None
        else []
    )

    per_category_taus = defaultdict(list)  # cat -> list of per-prompt mean τ
    per_category_blocks = defaultdict(int)
    per_prompt_records = []

    for i, (cat, prompt) in enumerate(questions):
        if args.chat_template and tokenizer.chat_template is not None:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )
        else:
            text = prompt
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(args.device)

        out = draft.spec_generate(
            target=target,
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            stop_token_ids=eos_ids,
            temperature=args.temperature,
        )
        acc = getattr(draft, "_last_acceptance_lengths", [])
        tau = (sum(acc) / len(acc)) if acc else 0.0

        if acc:
            per_category_taus[cat].append(tau)
            per_category_blocks[cat] += len(acc)

        if i % 10 == 0 or i == len(questions) - 1:
            print(
                f"  [{i+1}/{len(questions)}] cat={cat:<15} "
                f"tau={tau:.3f} blocks={len(acc)} "
                f"tokens_gen={sum(acc)}",
                flush=True,
            )

        if args.out_jsonl is not None:
            per_prompt_records.append({
                "idx": i,
                "category": cat,
                "prompt_preview": prompt[:200],
                "tau": tau,
                "n_blocks": len(acc),
                "n_tokens_generated": int(sum(acc)),
            })

    # ------------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(
        f"{'category':<20} {'n_prompts':>10} {'n_blocks':>10} "
        f"{'mean_tau':>10} {'std_tau':>10}"
    )
    print("-" * 70)
    overall_taus = []
    for cat in sorted(per_category_taus.keys()):
        taus = per_category_taus[cat]
        mean = statistics.mean(taus) if taus else 0.0
        std = statistics.stdev(taus) if len(taus) > 1 else 0.0
        print(
            f"{cat:<20} {len(taus):>10} {per_category_blocks[cat]:>10} "
            f"{mean:>10.3f} {std:>10.3f}"
        )
        overall_taus.extend(taus)
    print("-" * 70)
    overall_mean = statistics.mean(overall_taus) if overall_taus else 0.0
    overall_std = statistics.stdev(overall_taus) if len(overall_taus) > 1 else 0.0
    overall_blocks = sum(per_category_blocks.values())
    print(
        f"{'OVERALL':<20} {len(overall_taus):>10} {overall_blocks:>10} "
        f"{overall_mean:>10.3f} {overall_std:>10.3f}"
    )
    print("=" * 70)

    if args.out_jsonl is not None:
        with open(args.out_jsonl, "w") as f:
            for rec in per_prompt_records:
                f.write(json.dumps(rec) + "\n")
        print(f"\nWrote per-prompt results to {args.out_jsonl}")


if __name__ == "__main__":
    main()
