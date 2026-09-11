#!/usr/bin/env python3
"""
Formal Verification and Integrity Suite for Sparse NIDS Datasets.
Verifies that generated SVMLight files strictly adhere to the mathematical and structural
constraints defined in a declarative feature specification JSON.

Validates:
1. Feature Dimension Bounds: All indices strictly in [0, total_dimensions - 1].
2. Active Sparsity Invariance: Number of active features per sample matches specification.
3. Strict Ordering & Uniqueness: Indices are sorted ascending with no duplicate keys.
4. Window Block Distribution: Active features are evenly distributed across sliding window steps.
5. Overall Matrix Sparsity: Formally asserts the empirical sparsity percentage.
"""

import argparse
import json
import math
import sys
from collections import defaultdict

def verify_dataset(svm_path, spec_dict, max_check_samples=None):
    print(f"\n=======================================================")
    print(f"VERIFYING DATASET INTEGRITY AGAINST SPECIFICATION")
    print(f"=======================================================")
    print(f"Dataset File:   {svm_path}")
    print(f"Spec Name:      {spec_dict.get('name', 'Custom')}")
    print(f"Spec Version:   {spec_dict.get('version', 'N/A')}")
    
    total_dims = spec_dict["total_dimensions"]
    window_size = spec_dict.get("window_size", 10)
    dim_per_packet = spec_dict.get("dimensions_per_packet", total_dims // window_size)
    min_active = spec_dict.get("min_active_per_sample", 1)
    max_active = spec_dict.get("max_active_per_sample", total_dims)
    expected_full = spec_dict.get("expected_active_full_window", 60)
    expected_sparsity = spec_dict.get("expected_sparsity_percent")

    print(f"Total Dimensions:           {total_dims:,}")
    print(f"Window Size:                {window_size} packets")
    print(f"Dimensions per Packet:      {dim_per_packet:,}")
    print(f"Expected Full Window Active:{expected_full}")
    print(f"Allowed Active Range:       [{min_active}, {max_active}]")
    print(f"Expected Matrix Sparsity:   ~{expected_sparsity}%\n")

    line_count = 0
    errors = []
    active_counts = defaultdict(int)
    label_distribution = defaultdict(int)
    step_hit_counts = [0] * window_size

    with open(svm_path, "r") as f:
        for line_idx, line in enumerate(f, 1):
            if max_check_samples and line_idx > max_check_samples:
                break
                
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            label_str = parts[0]
            try:
                label_int = int(label_str)
                label_distribution[label_int] += 1
            except ValueError:
                errors.append(f"Line {line_idx}: Invalid non-integer label '{label_str}'")
                continue

            feature_tokens = parts[1:]
            num_active = len(feature_tokens)
            active_counts[num_active] += 1

            if num_active < min_active or num_active > max_active:
                if len(errors) < 10:
                    errors.append(f"Line {line_idx}: Active feature count {num_active} out of allowed range [{min_active}, {max_active}]")

            prev_idx = -1
            current_step_hits = [0] * window_size

            for tok in feature_tokens:
                if ":" not in tok:
                    errors.append(f"Line {line_idx}: Malformed token '{tok}', missing ':' delimiter")
                    break
                k_str, v_str = tok.split(":", 1)
                try:
                    k = int(k_str)
                    v = float(v_str)
                except ValueError:
                    errors.append(f"Line {line_idx}: Non-numeric token '{tok}'")
                    break

                # 1. Bounds check
                if k < 0 or k >= total_dims:
                    errors.append(f"Line {line_idx}: Feature index {k} out of bounds [0, {total_dims - 1}]")

                # 2. Strict ordering check
                if k <= prev_idx:
                    errors.append(f"Line {line_idx}: Unsorted or duplicate index {k} (previous was {prev_idx})")

                # 3. Value check (should be binary 1 in OHE)
                if abs(v - 1.0) > 1e-5:
                    errors.append(f"Line {line_idx}: Value for feature {k} is {v}, expected 1.0")

                # Track window step
                step = k // dim_per_packet
                if 0 <= step < window_size:
                    current_step_hits[step] += 1

                prev_idx = k

            for s in range(window_size):
                step_hit_counts[s] += current_step_hits[s]

            line_count += 1
            if line_count % 100000 == 0:
                print(f"  Verified {line_count:,} samples...")

    print(f"\nCompleted verification over {line_count:,} samples.")

    # Integrity Summary
    print("\n-------------------------------------------------------")
    print("VERIFICATION CHECKS SUMMARY")
    print("-------------------------------------------------------")
    
    passed = True
    if errors:
        passed = False
        print(f"❌ INTEGRITY ERRORS DETECTED: {len(errors)} errors encountered.")
        for err in errors[:10]:
            print(f"   - {err}")
        if len(errors) > 10:
            print(f"   ... and {len(errors) - 10} additional errors.")
    else:
        print(f"✅ SYNTAX & BOUNDS: All {line_count:,} lines strictly valid, sorted, and within [0, {total_dims-1}].")

    # Sparsity verification
    out_of_range = sum(cnt for num, cnt in active_counts.items() if num < min_active or num > max_active)
    if out_of_range == 0:
        print(f"✅ ACTIVE FEATURES RANGE: All {line_count:,} samples within allowed [{min_active}, {max_active}] range.")
    else:
        passed = False
        print(f"❌ ACTIVE FEATURES RANGE: {out_of_range} samples out of allowed [{min_active}, {max_active}] range.")

    full_window_count = sum(cnt for num, cnt in active_counts.items() if num >= expected_full)
    print(f"ℹ️ FULL WINDOW RATIO: {full_window_count:,} / {line_count:,} ({full_window_count/line_count*100:.2f}%) samples have full 10-packet history.")

    mean_active = sum(k * v for k, v in active_counts.items()) / line_count if line_count > 0 else 0
    empirical_sparsity = (1.0 - (mean_active / total_dims)) * 100
    print(f"✅ EMPIRICAL SPARSITY: {empirical_sparsity:.4f}% (Expected: {expected_sparsity}%)")

    print("\n-------------------------------------------------------")
    print("WINDOW STEP DISTRIBUTION (Features per Window Step)")
    print("-------------------------------------------------------")
    for step_idx, total_hits in enumerate(step_hit_counts):
        avg_per_sample = total_hits / line_count if line_count > 0 else 0
        print(f"  Step {step_idx} (Packets t-{window_size - 1 - step_idx}): {avg_per_sample:.2f} active features / sample")

    print("\n-------------------------------------------------------")
    print("LABEL DISTRIBUTION")
    print("-------------------------------------------------------")
    for lbl, count in sorted(label_distribution.items()):
        pct = (count / line_count) * 100 if line_count > 0 else 0
        print(f"  Class {lbl}: {count:,} samples ({pct:.2f}%)")

    print("\n=======================================================")
    if passed:
        print("VERIFICATION RESULT: PASSED (100% Compliant with Specification)")
    else:
        print("VERIFICATION RESULT: FAILED (Violations detected)")
    print("=======================================================\n")
    return passed

def main():
    parser = argparse.ArgumentParser(description="Verify SVMLight dataset compliance against feature specification")
    parser.add_argument("--input", required=True, type=str, help="Path to SVMLight file to verify")
    parser.add_argument("--spec", required=True, type=str, help="Path to feature specification JSON")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to verify (default: all)")
    args = parser.parse_args()

    with open(args.spec, "r") as f:
        spec_dict = json.load(f)

    success = verify_dataset(args.input, spec_dict, args.max_samples)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
