#!/usr/bin/env python3
"""
aggregate_eval_shards.py — combine per-shard per-prompt JSONLs from
eval_spec_bench.py (run with --shard-rank/--num-shards) and print the
unified per-category τ table.

Usage:
    python docs/ascend_npu/aggregate_eval_shards.py \\
        ./outputs/eval-gsm8k-shard0.jsonl \\
        ./outputs/eval-gsm8k-shard1.jsonl \\
        ... \\
        ./outputs/eval-gsm8k-shard7.jsonl

Or with a glob:
    python docs/ascend_npu/aggregate_eval_shards.py ./outputs/eval-gsm8k-shard*.jsonl
"""

import argparse
import glob
import json
import statistics
import sys
from collections import defaultdict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+",
                   help="per-shard JSONL files from eval_spec_bench.py "
                        "(may include globs)")
    args = p.parse_args()

    # Expand globs (shell may already do this, but be defensive)
    paths = []
    for f in args.files:
        expanded = glob.glob(f)
        paths.extend(expanded if expanded else [f])

    if not paths:
        print("ERROR: no input files matched.", file=sys.stderr)
        sys.exit(1)

    per_cat_taus = defaultdict(list)
    per_cat_blocks = defaultdict(int)
    total_records = 0
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                cat = rec.get("category", "unknown")
                tau = rec.get("tau", 0.0)
                nb = rec.get("n_blocks", 0)
                if nb > 0:  # skip prompts that produced 0 blocks
                    per_cat_taus[cat].append(tau)
                    per_cat_blocks[cat] += nb
                total_records += 1

    print(f"Loaded {total_records} records from {len(paths)} file(s)")
    print()
    print("=" * 70)
    print(
        f"{'category':<20} {'n_prompts':>10} {'n_blocks':>10} "
        f"{'mean_tau':>10} {'std_tau':>10}"
    )
    print("-" * 70)
    overall = []
    for cat in sorted(per_cat_taus):
        taus = per_cat_taus[cat]
        mean = statistics.mean(taus) if taus else 0.0
        std = statistics.stdev(taus) if len(taus) > 1 else 0.0
        print(
            f"{cat:<20} {len(taus):>10} {per_cat_blocks[cat]:>10} "
            f"{mean:>10.3f} {std:>10.3f}"
        )
        overall.extend(taus)
    print("-" * 70)
    mean = statistics.mean(overall) if overall else 0.0
    std = statistics.stdev(overall) if len(overall) > 1 else 0.0
    total_blocks = sum(per_cat_blocks.values())
    print(
        f"{'OVERALL':<20} {len(overall):>10} {total_blocks:>10} "
        f"{mean:>10.3f} {std:>10.3f}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
