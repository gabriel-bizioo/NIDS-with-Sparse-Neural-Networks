#!/usr/bin/env python3
"""
Trivial Static Baseline Classifier for SLIDE-NIDS.

Implements a deterministic, zero-intelligence lookup table (majority-vote rule)
over the discrete port-agnostic feature representations.

Purpose:
Establishes the scientific ground-floor performance (baseline ruler).
The neural network (Stage 3) must significantly outperform this trivial baseline
to justify deep learning over simple hash-table memorization.

Metrics Evaluated:
1. Overall Accuracy and Macro/Weighted F1-score on the Flow-Disjoint Test Set.
2. Per-class Precision, Recall, F1-score, and Support.
3. Pattern Coverage: Accuracy on seen behavioral patterns vs unseen fallbacks.
4. Representation Ambiguity: Pure vs conflicting patterns in training.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict, Counter
from datetime import datetime


def train_lookup_table(train_path, max_samples=None):
    print(f"\n[1/2] Building Static Lookup Table from: {train_path} ...")
    t0 = time.time()
    
    pattern_label_counts = defaultdict(Counter)
    global_class_counts = Counter()
    total_train = 0

    with open(train_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            try:
                label = int(parts[0])
            except ValueError:
                continue
            
            feat_str = " ".join(parts[1:])
            # Fast 16-byte MD5 digest for pattern key
            pattern_hash = hashlib.md5(feat_str.encode('utf-8')).digest()
            
            pattern_label_counts[pattern_hash][label] += 1
            global_class_counts[label] += 1
            total_train += 1

            if line_idx % 1000000 == 0:
                print(f"  Ingested {line_idx:,} training samples ({len(pattern_label_counts):,} unique patterns)...")

            if max_samples and total_train >= max_samples:
                break

    elapsed = time.time() - t0
    global_majority_class = global_class_counts.most_common(1)[0][0]

    # Compile decision table by majority vote
    decision_table = {}
    pure_patterns = 0
    conflicting_patterns = 0

    for pat, counts in pattern_label_counts.items():
        majority_label, count = counts.most_common(1)[0]
        decision_table[pat] = majority_label
        if len(counts) == 1:
            pure_patterns += 1
        else:
            conflicting_patterns += 1

    print(f"  Training Complete in {elapsed:.2f}s ({total_train / elapsed:,.0f} samples/s).")
    print(f"  Total Training Samples:      {total_train:,}")
    print(f"  Unique Behavioral Patterns:  {len(decision_table):,}")
    print(f"  Pure Patterns (100% 1 class):{pure_patterns:,} ({pure_patterns / len(decision_table) * 100:.2f}%)")
    print(f"  Conflicting Patterns:        {conflicting_patterns:,} ({conflicting_patterns / len(decision_table) * 100:.2f}%)")
    print(f"  Global Majority Class:       {global_majority_class} ({global_class_counts[global_majority_class]:,} samples)")

    return {
        "decision_table": decision_table,
        "global_majority_class": global_majority_class,
        "total_train_samples": total_train,
        "unique_patterns": len(decision_table),
        "pure_patterns": pure_patterns,
        "conflicting_patterns": conflicting_patterns,
        "global_class_counts": global_class_counts,
        "training_time_seconds": elapsed
    }


def evaluate_lookup_table(test_path, model_dict, label_names, max_samples=None):
    decision_table = model_dict["decision_table"]
    global_majority_class = model_dict["global_majority_class"]

    print(f"\n[2/2] Evaluating Static Lookup Table on: {test_path} ...")
    t0 = time.time()

    total_test = 0
    correct = 0
    seen_in_table = 0
    correct_seen = 0
    unseen_samples = 0
    correct_unseen = 0

    # Class metrics counters
    true_counts = Counter()
    pred_counts = Counter()
    tp_counts = Counter()
    fp_counts = Counter()
    fn_counts = Counter()

    with open(test_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            try:
                true_label = int(parts[0])
            except ValueError:
                continue

            feat_str = " ".join(parts[1:])
            pattern_hash = hashlib.md5(feat_str.encode('utf-8')).digest()

            total_test += 1
            true_counts[true_label] += 1

            if pattern_hash in decision_table:
                pred = decision_table[pattern_hash]
                seen_in_table += 1
                if pred == true_label:
                    correct += 1
                    correct_seen += 1
            else:
                pred = global_majority_class
                unseen_samples += 1
                if pred == true_label:
                    correct += 1
                    correct_unseen += 1

            pred_counts[pred] += 1

            if pred == true_label:
                tp_counts[true_label] += 1
            else:
                fp_counts[pred] += 1
                fn_counts[true_label] += 1

            if line_idx % 200000 == 0:
                print(f"  Evaluated {line_idx:,} test samples (current accuracy: {correct / total_test * 100:.2f}%)...")

            if max_samples and total_test >= max_samples:
                break

    elapsed = time.time() - t0
    accuracy = correct / total_test if total_test > 0 else 0.0
    acc_seen = correct_seen / seen_in_table if seen_in_table > 0 else 0.0
    acc_unseen = correct_unseen / unseen_samples if unseen_samples > 0 else 0.0
    coverage = seen_in_table / total_test if total_test > 0 else 0.0

    # Calculate per-class Precision, Recall, F1
    all_classes = sorted(list(set(true_counts.keys()) | set(pred_counts.keys())))
    class_metrics = {}
    macro_p, macro_r, macro_f1 = 0.0, 0.0, 0.0
    weighted_p, weighted_r, weighted_f1 = 0.0, 0.0, 0.0

    for c in all_classes:
        tp = tp_counts[c]
        fp = fp_counts[c]
        fn = fn_counts[c]
        support = true_counts[c]

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        c_name = label_names.get(c, f"Class {c}")
        class_metrics[c] = {
            "name": c_name,
            "support": support,
            "predictions": pred_counts[c],
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        }

        macro_p += prec
        macro_r += rec
        macro_f1 += f1

        weighted_p += prec * support
        weighted_r += rec * support
        weighted_f1 += f1 * support

    num_classes = len(all_classes) if all_classes else 1
    macro_p /= num_classes
    macro_r /= num_classes
    macro_f1 /= num_classes

    weighted_p /= total_test if total_test > 0 else 1
    weighted_r /= total_test if total_test > 0 else 1
    weighted_f1 /= total_test if total_test > 0 else 1

    print(f"\nEvaluation Complete in {elapsed:.2f}s ({total_test / elapsed:,.0f} samples/s).")
    print(f"=" * 75)
    print(f"                TRIVIAL BASELINE (STATIC LOOKUP TABLE) RESULTS")
    print(f"=" * 75)
    print(f"Overall Test Accuracy:        {accuracy * 100:.4f}% ({correct:,} / {total_test:,})")
    print(f"Macro F1-Score:               {macro_f1 * 100:.4f}%")
    print(f"Weighted F1-Score:            {weighted_f1 * 100:.4f}%")
    print(f"Pattern Coverage (Seen):      {coverage * 100:.2f}% ({seen_in_table:,} / {total_test:,})")
    print(f"  - Accuracy on Seen Patterns:{acc_seen * 100:.2f}%")
    print(f"  - Accuracy on Unseen Fallback:{acc_unseen * 100:.2f}% ({unseen_samples:,} samples defaulted to Class {global_majority_class})")
    print("-" * 75)
    print(f"{'Class ID':<8} | {'Class Name':<20} | {'Support':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 75)
    for c in all_classes:
        m = class_metrics[c]
        print(f"{c:<8} | {m['name']:<20} | {m['support']:<10,} | {m['precision'] * 100:>8.2f}% | {m['recall'] * 100:>8.2f}% | {m['f1_score'] * 100:>8.2f}%")
    print("-" * 75)
    print(f"{'Macro':<8} | {'Average':<20} | {total_test:<10,} | {macro_p * 100:>8.2f}% | {macro_r * 100:>8.2f}% | {macro_f1 * 100:>8.2f}%")
    print(f"{'Weighted':<8} | {'Average':<20} | {total_test:<10,} | {weighted_p * 100:>8.2f}% | {weighted_r * 100:>8.2f}% | {weighted_f1 * 100:>8.2f}%")
    print(f"=" * 75)

    return {
        "overall_accuracy": accuracy,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_p,
        "weighted_recall": weighted_r,
        "weighted_f1": weighted_f1,
        "pattern_coverage": coverage,
        "accuracy_on_seen": acc_seen,
        "accuracy_on_unseen": acc_unseen,
        "total_test_samples": total_test,
        "correct_samples": correct,
        "seen_samples": seen_in_table,
        "unseen_samples": unseen_samples,
        "evaluation_time_seconds": elapsed,
        "class_metrics": class_metrics
    }


def main():
    parser = argparse.ArgumentParser(description="Trivial Static Baseline Classifier for SLIDE-NIDS")
    parser.add_argument("--train", required=True, type=str, help="Path to Train SVMLight file")
    parser.add_argument("--test", required=True, type=str, help="Path to Test SVMLight file")
    parser.add_argument("--label-map", default="configs/label_maps/label_map_cicids2017_multiclass.json", type=str, help="Path to label map JSON")
    parser.add_argument("--output-json", default=None, type=str, help="Path to save JSON results")
    parser.add_argument("--output-md", default=None, type=str, help="Path to save Markdown report")
    parser.add_argument("--max-train-samples", default=None, type=int, help="Optional max train samples to ingest")
    parser.add_argument("--max-test-samples", default=None, type=int, help="Optional max test samples to evaluate")
    args = parser.parse_args()

    # Load label names
    label_map_path = args.label_map
    if not os.path.exists(label_map_path):
        fallback = "scripts/label_map_cicids2017_multiclass.json"
        if os.path.exists(fallback):
            label_map_path = fallback

    label_names = {}
    if os.path.exists(label_map_path):
        with open(label_map_path, "r") as f:
            raw = json.load(f)
        for k, v in raw.items():
            if v not in label_names:
                label_names[v] = k

    # Train Table
    model_stats = train_lookup_table(args.train, args.max_train_samples)

    # Evaluate Table
    eval_stats = evaluate_lookup_table(args.test, model_stats, label_names, args.max_test_samples)

    results = {
        "timestamp": datetime.now().isoformat(),
        "baseline_name": "Trivial Static Lookup Table (Port-Agnostic Behavioral Memory)",
        "train_file": args.train,
        "test_file": args.test,
        "model_training": {
            "train_samples": model_stats["total_train_samples"],
            "unique_patterns_in_table": model_stats["unique_patterns"],
            "pure_patterns": model_stats["pure_patterns"],
            "conflicting_patterns": model_stats["conflicting_patterns"],
            "global_majority_class": model_stats["global_majority_class"],
            "training_time_seconds": model_stats["training_time_seconds"]
        },
        "evaluation_summary": {
            "test_samples": eval_stats["total_test_samples"],
            "overall_accuracy": eval_stats["overall_accuracy"],
            "macro_f1": eval_stats["macro_f1"],
            "weighted_f1": eval_stats["weighted_f1"],
            "pattern_coverage": eval_stats["pattern_coverage"],
            "accuracy_on_seen_patterns": eval_stats["accuracy_on_seen"],
            "accuracy_on_unseen_patterns": eval_stats["accuracy_on_unseen"],
            "evaluation_time_seconds": eval_stats["evaluation_time_seconds"]
        },
        "per_class_results": eval_stats["class_metrics"]
    }

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"JSON Baseline Results saved to: {args.output_json}")

    if args.output_md:
        os.makedirs(os.path.dirname(args.output_md) or ".", exist_ok=True)
        with open(args.output_md, "w") as f:
            f.write(f"# Trivial Baseline Benchmark Certificate: SLIDE-NIDS\n\n")
            f.write(f"**Execution Timestamp**: `{results['timestamp']}`  \n")
            f.write(f"**Baseline Architecture**: Zero-Intelligence Deterministic Majority Table  \n")
            f.write(f"**Dataset**: V5 Port-Agnostic (2.910 Dimensions, Flow-Disjoint Split)  \n\n")
            f.write(f"## Ground Floor Performance Summary\n\n")
            f.write(f"- **Overall Test Accuracy**: **`{eval_stats['overall_accuracy'] * 100:.4f}%`**\n")
            f.write(f"- **Macro F1-Score**: **`{eval_stats['macro_f1'] * 100:.4f}%`**\n")
            f.write(f"- **Weighted F1-Score**: **`{eval_stats['weighted_f1'] * 100:.4f}%`**\n")
            f.write(f"- **Pattern Coverage (Seen Patterns)**: `{eval_stats['pattern_coverage'] * 100:.2f}%` ({eval_stats['seen_samples']:,} / {eval_stats['total_test_samples']:,})\n")
            f.write(f"  - Accuracy on Seen Patterns: `{eval_stats['accuracy_on_seen'] * 100:.2f}%`\n")
            f.write(f"  - Accuracy on Unseen Patterns (Default Fallback): `{eval_stats['accuracy_on_unseen'] * 100:.2f}%`\n\n")
            f.write(f"> [!IMPORTANT]\n")
            f.write(f"> This trivial lookup table memorizes identical feature sequences without neural learning. ")
            f.write(f"To justify deep learning, the neural network (Stage 3) must outperform this accuracy and F1 score, ")
            f.write(f"especially on unseen flows and minority attack classes.\n\n")
            f.write(f"## Per-Class Breakdown\n\n")
            f.write(f"| Class ID | Class Name | Support | Precision | Recall | F1-Score |\n")
            f.write(f"| :---: | :--- | :---: | :---: | :---: | :---: |\n")
            for c, m in eval_stats["class_metrics"].items():
                f.write(f"| {c} | {m['name']} | {m['support']:,} | {m['precision'] * 100:.2f}% | {m['recall'] * 100:.2f}% | {m['f1_score'] * 100:.2f}% |\n")
            f.write(f"| **Macro** | **Macro Average** | `{eval_stats['total_test_samples']:,}` | `{eval_stats['macro_precision'] * 100:.2f}%` | `{eval_stats['macro_recall'] * 100:.2f}%` | `{eval_stats['macro_f1'] * 100:.2f}%` |\n")
            f.write(f"| **Weighted** | **Weighted Average** | `{eval_stats['total_test_samples']:,}` | `{eval_stats['weighted_precision'] * 100:.2f}%` | `{eval_stats['weighted_recall'] * 100:.2f}%` | `{eval_stats['weighted_f1'] * 100:.2f}%` |\n")
        print(f"Markdown Baseline Report saved to: {args.output_md}")


if __name__ == "__main__":
    main()
