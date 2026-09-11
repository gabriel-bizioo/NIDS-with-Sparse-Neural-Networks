#!/usr/bin/env python3
"""
Multi-Day Composite Dataset Assembler for SLIDE NIDS

Assembles pre-sparsified SVMLight datasets across multiple capture days (Mon, Tue, Wed, Thu, Fri)
into a compact, perfectly balanced training and testing dataset.

Features:
- Fast streaming over multi-gigabyte SVMLight files (< 30 seconds).
- Quota controls per attack class (--max-per-attack).
- Configurable Benign representation (--benign-ratio or --benign-multiplier) to prevent
  distorting real-world priors and exploding False Positive Rates.
- Stratified sampling with deterministic shuffling.
- Direct output of SLIDE Config.csv parameters (totRecords, totRecordsTest).

Usage Example:
  python3 scripts/build_composite_dataset.py \
    --train-inputs datasets/CICIDS2017/friday/v4DefinitiveTrain.txt \
    --test-inputs datasets/CICIDS2017/friday/v4DefinitiveTest.txt \
    --output-train datasets/CICIDS2017/composite_week_train.txt \
    --output-test datasets/CICIDS2017/composite_week_test.txt \
    --max-per-attack 15000 \
    --benign-multiplier 2.0
"""

import argparse
import random
import sys
from collections import defaultdict

LABEL_NAMES = {
    0: "BENIGN",
    1: "Portscan",
    2: "DDoS",
    3: "Botnet",
    4: "Infiltration",
    5: "Web Attack - Brute Force",
    6: "Web Attack - XSS",
    7: "Web Attack - Sql Injection",
    8: "FTP-Patator",
    9: "SSH-Patator",
    10: "DoS slowloris",
    11: "DoS Slowhttptest",
    12: "DoS Hulk",
    13: "DoS GoldenEye",
    14: "Heartbleed"
}


def stream_collect_samples(file_paths, max_per_class_map, seed=42):
    """
    Pass 1: Counts lines per class across all inputs.
    Pass 2: Uses reservoir or deterministic stride sampling to select target quotas.
    """
    random.seed(seed)
    # Count totals per class
    print("Scanning available class counts across input files...")
    counts_per_class = defaultdict(int)
    for p in file_paths:
        print(f"  Inspecting: {p}")
        with open(p, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if not line.strip():
                    continue
                lbl = int(line.split()[0])
                counts_per_class[lbl] += 1

    print("\nAvailable Samples per Class:")
    for lbl in sorted(counts_per_class.keys()):
        name = LABEL_NAMES.get(lbl, f"Class {lbl}")
        print(f"  Class {lbl:2d} ({name:<26}): {counts_per_class[lbl]:>10,}")

    # Determine selection rates
    sample_targets = {}
    for lbl, total in counts_per_class.items():
        quota = max_per_class_map.get(lbl, max_per_class_map.get("default", 10000))
        sample_targets[lbl] = min(total, quota)

    print("\nTarget Quotas to Extract:")
    for lbl in sorted(sample_targets.keys()):
        name = LABEL_NAMES.get(lbl, f"Class {lbl}")
        print(f"  Class {lbl:2d} ({name:<26}): {sample_targets[lbl]:>10,}")

    # Pass 2: Single-pass Reservoir Sampling (guarantees exactly target quota with uniform probability K/N)
    reservoirs = defaultdict(list)
    seen_per_class = defaultdict(int)

    print("\nExtracting balanced sample subsets via Reservoir Sampling...")
    for p in file_paths:
        print(f"  Streaming: {p}")
        with open(p, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if not line.strip():
                    continue
                lbl = int(line.split()[0])
                K = sample_targets[lbl]
                if K <= 0:
                    continue
                
                i = seen_per_class[lbl]
                seen_per_class[lbl] += 1

                if len(reservoirs[lbl]) < K:
                    reservoirs[lbl].append(line)
                else:
                    j = random.randint(0, i)
                    if j < K:
                        reservoirs[lbl][j] = line

    # Consolidate and shuffle
    all_lines = []
    actual_targets = {}
    for lbl, lines in reservoirs.items():
        all_lines.extend(lines)
        actual_targets[lbl] = len(lines)
    random.shuffle(all_lines)
    return all_lines, actual_targets


def main():
    parser = argparse.ArgumentParser(description="Multi-Day Composite Dataset Assembler for SLIDE NIDS")
    parser.add_argument("--train-inputs", nargs="+", required=True, help="List of pre-sparsified train SVMLight files")
    parser.add_argument("--test-inputs", nargs="+", required=True, help="List of pre-sparsified test SVMLight files")
    parser.add_argument("--output-train", required=True, type=str, help="Path to output composite train SVMLight file")
    parser.add_argument("--output-test", required=True, type=str, help="Path to output composite test SVMLight file")
    parser.add_argument("--max-per-attack", default=15000, type=int, help="Maximum training samples per attack class (default: 15,000)")
    parser.add_argument("--benign-multiplier", default=2.0, type=float, help="Ratio of Benign samples relative to max-per-attack (e.g. 2.0 = 30k Benign if attack quota is 15k)")
    parser.add_argument("--test-max-per-class", default=5000, type=int, help="Maximum samples per class in test set (default: 5,000)")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for deterministic sampling")
    args = parser.parse_args()

    benign_train_target = int(args.max_per_attack * args.benign_multiplier)
    train_quota_map = {"default": args.max_per_attack, 0: benign_train_target}
    test_quota_map = {"default": args.test_max_per_class, 0: int(args.test_max_per_class * args.benign_multiplier)}

    print("=" * 75)
    print("               BUILDING COMPOSITE TRAINING SET")
    print("=" * 75)
    train_lines, train_targets = stream_collect_samples(args.train_inputs, train_quota_map, seed=args.seed)

    print(f"\nWriting {len(train_lines):,} training samples to: {args.output_train} ...")
    with open(args.output_train, "w", encoding="utf-8") as f:
        f.writelines(train_lines)

    print("\n" + "=" * 75)
    print("               BUILDING COMPOSITE TESTING SET")
    print("=" * 75)
    test_lines, test_targets = stream_collect_samples(args.test_inputs, test_quota_map, seed=args.seed + 1)

    print(f"\nWriting {len(test_lines):,} testing samples to: {args.output_test} ...")
    with open(args.output_test, "w", encoding="utf-8") as f:
        f.writelines(test_lines)

    priors_path = f"{args.output_train}.priors.json"
    priors_data = {int(k): int(v) for k, v in train_targets.items()}
    import json
    with open(priors_path, "w") as f:
        json.dump(priors_data, f, indent=2)
    print(f"Saved Training Class Counts to: {priors_path}")

    print("\n" + "=" * 75)
    print("                    ASSEMBLY COMPLETE! SUMMARY")
    print("=" * 75)
    print(f"Total Composite Train Samples: {len(train_lines):,}")
    print(f"Total Composite Test Samples:  {len(test_lines):,}")
    print("\nSLIDE Configuration Parameters (Paste into Config.csv):")
    print("-" * 55)
    print(f"totRecords = {len(train_lines)}")
    print(f"totRecordsTest = {len(test_lines)}")
    print(f"trainData = {args.output_train}")
    print(f"testData = {args.output_test}")
    print("-" * 55 + "\n")


if __name__ == "__main__":
    main()
