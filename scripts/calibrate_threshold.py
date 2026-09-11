#!/usr/bin/env python3
"""
Zero-Attack Baseline Calibrator & Operational Threshold Quantile Tool.

Ingests a guaranteed pure-benign network trace (e.g., Monday-WorkingHours or
pre-attack morning traces), runs inference with optional Bayesian Logit Adjustment,
computes empirical noise percentiles on benign traffic, and calculates the exact
decision threshold tau needed to guarantee carrier-grade False Positive Rates
(e.g., FPR <= 1%, 0.1%, 0.01%).

Outputs:
  configs/operational_threshold.json

Usage:
  python3 scripts/calibrate_threshold.py \
    --weights ../HashingDeepLearning/dataset/CICIDS2017_improved/savedWeights_v4.npz \
    --benign-test datasets/CICIDS2017/friday/v4Test_50k_test.txt \
    --output configs/operational_threshold.json \
    --adjust-logits \
    --real-benign-prior 0.999 \
    --plot configs/threshold_calibration.png
"""

import argparse
import json
import os
import sys
import numpy as np


def compute_logit_adjustment(num_classes, train_priors_spec, real_benign_prior, tau_adjust):
    if train_priors_spec == "balanced":
        pi_train = np.ones(num_classes, dtype=np.float64) / num_classes
    elif os.path.exists(train_priors_spec):
        with open(train_priors_spec, "r") as f:
            counts = json.load(f)
        pi_train = np.zeros(num_classes, dtype=np.float64)
        for k, v in counts.items():
            idx = int(k)
            if idx < num_classes:
                pi_train[idx] = float(v)
        pi_train = pi_train / np.sum(pi_train)
    else:
        vals = [float(x.strip()) for x in train_priors_spec.split(",")]
        pi_train = np.array(vals, dtype=np.float64)
        pi_train = pi_train / np.sum(pi_train)

    pi_real = np.zeros(num_classes, dtype=np.float64)
    pi_real[0] = real_benign_prior
    attack_train_sum = np.sum(pi_train[1:])
    remaining_prob = 1.0 - real_benign_prior
    if attack_train_sum > 0:
        pi_real[1:] = remaining_prob * (pi_train[1:] / attack_train_sum)
    else:
        pi_real[1:] = remaining_prob / (num_classes - 1)

    eps = 1e-12
    delta = tau_adjust * np.log((pi_train + eps) / (pi_real + eps))
    return delta.astype(np.float32), pi_train, pi_real


def main():
    parser = argparse.ArgumentParser(description="Zero-Attack Baseline Calibrator & Threshold Quantile Tool")
    parser.add_argument("--weights", required=True, type=str, help="Path to savedWeights.npz file")
    parser.add_argument("--benign-test", required=True, type=str, help="Path to SVMLight file containing pure BENIGN traffic")
    parser.add_argument("--output", default="configs/operational_threshold.json", type=str, help="Path to output threshold config JSON")
    parser.add_argument("--max-samples", default=100000, type=int, help="Max benign samples to evaluate (0 for all)")

    # Option 2: Bayesian Logit Adjustment parameters
    parser.add_argument("--adjust-logits", action="store_true", help="Enable Bayesian Logit Adjustment during calibration")
    parser.add_argument("--train-priors", default="balanced", type=str, help="'balanced', comma-separated floats, or JSON file")
    parser.add_argument("--real-benign-prior", default=0.999, type=float, help="Assumed real-world benign prior probability (default: 0.999)")
    parser.add_argument("--filter-benign", action="store_true", default=True, help="Automatically filter for label 0 (benign) samples only")
    parser.add_argument("--tau-adjust", default=1.0, type=float, help="Temperature/scale for logit adjustment (default: 1.0)")

    # Optional visualization
    parser.add_argument("--plot", default=None, type=str, help="Optional output image path to save calibration histogram")

    args = parser.parse_args()

    print("=" * 70)
    print("      ZERO-ATTACK BASELINE CALIBRATION & THRESHOLD ESTIMATION        ")
    print("=" * 70)
    print(f"Loading SLIDE weights from: {args.weights}")
    data = np.load(args.weights)
    layers = []
    layer_idx = 0
    while f"w_layer_{layer_idx}" in data:
        W = data[f"w_layer_{layer_idx}"]
        b = data[f"b_layer_{layer_idx}"]
        layers.append((W, b))
        layer_idx += 1

    if len(layers) == 0:
        raise ValueError(f"No weight layers found in {args.weights}")

    input_dim = layers[0][0].shape[1]
    num_classes = layers[-1][0].shape[0]
    print(f"Model Architecture: Input {input_dim} -> Hidden {[W.shape[0] for W, b in layers[:-1]]} -> Output {num_classes}")

    delta_adjustment = None
    if args.adjust_logits:
        delta_adjustment, pi_train, pi_real = compute_logit_adjustment(
            num_classes, args.train_priors, args.real_benign_prior, args.tau_adjust
        )
        print(f"Bayesian Logit Adjustment: ENABLED (tau={args.tau_adjust}, real benign prior={args.real_benign_prior:.4f})")
    else:
        print("Bayesian Logit Adjustment: DISABLED (Raw training logits)")

    print(f"Ingesting benign trace from: {args.benign_test} ...")

    batch_size = 5000
    batch_indices = []
    batch_values = []
    attack_scores = []
    total_samples = 0
    non_zero_labels = 0

    def evaluate_batch(indices_list, values_list):
        N = len(indices_list)
        W0, b0 = layers[0]
        H = np.zeros((N, W0.shape[0]), dtype=np.float32)
        for i in range(N):
            idx = indices_list[i]
            val = values_list[i]
            if len(idx) > 0:
                H[i] = np.dot(W0[:, idx], val) + b0
            else:
                H[i] = b0
        H = np.maximum(0, H)

        for l_idx in range(1, len(layers) - 1):
            W, b = layers[l_idx]
            H = np.maximum(0, np.dot(H, W.T) + b)

        W_last, b_last = layers[-1]
        Z = np.dot(H, W_last.T) + b_last

        if delta_adjustment is not None:
            Z = Z - delta_adjustment

        Z_max = np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z - Z_max)
        probs = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

        # Total attack probability across all non-benign classes: P(Attack | x) = 1 - P(Benign | x)
        total_attack_prob = 1.0 - probs[:, 0]
        return total_attack_prob

    with open(args.benign_test, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            lbl = int(parts[0])
            if lbl != 0:
                non_zero_labels += 1
                if args.filter_benign:
                    continue
            idxs = []
            vals = []
            for item in parts[1:]:
                if ":" in item:
                    k, v = item.split(":", 1)
                    k_int = int(k)
                    if k_int < input_dim:
                        idxs.append(k_int)
                        vals.append(float(v))

            batch_indices.append(np.array(idxs, dtype=np.int32))
            batch_values.append(np.array(vals, dtype=np.float32))
            total_samples += 1

            if len(batch_indices) >= batch_size:
                scores = evaluate_batch(batch_indices, batch_values)
                attack_scores.extend(scores)
                batch_indices = []
                batch_values = []
                if total_samples % 25000 == 0:
                    print(f"  Processed {total_samples:,} benign samples...")

            if args.max_samples and total_samples >= args.max_samples:
                break

    if len(batch_indices) > 0:
        scores = evaluate_batch(batch_indices, batch_values)
        attack_scores.extend(scores)

    if non_zero_labels > 0:
        print(f"WARNING: Found {non_zero_labels:,} non-benign labels in calibration dataset! "
              "Baseline calibration requires purely BENIGN traffic.")

    attack_scores = np.array(attack_scores, dtype=np.float64)
    N = len(attack_scores)
    print(f"\nSuccessfully evaluated {N:,} benign samples.")
    print("-" * 70)

    # Compute empirical percentiles for target FPRs
    fpr_targets = [
        ("fpr_0.10", 0.10),
        ("fpr_0.05", 0.05),
        ("fpr_0.02", 0.02),
        ("fpr_0.01", 0.01),
        ("fpr_0.005", 0.005),
        ("fpr_0.001", 0.001),
        ("fpr_0.0001", 0.0001),
        ("fpr_0.00001", 0.00001),
    ]

    thresholds = {}
    print(f"{'Target FPR':<15} | {'Quantile':<12} | {'Required Threshold (tau)':<25} | {'Empirical FP Count'}")
    print("-" * 70)

    for key, fpr in fpr_targets:
        quantile = (1.0 - fpr) * 100.0
        # Calculate threshold
        tau_val = float(np.percentile(attack_scores, quantile))
        fp_count = int(np.sum(attack_scores >= tau_val))
        thresholds[key] = round(tau_val, 6)
        print(f"{fpr * 100:>6.3f}% ({key:<10}) | {quantile:>7.4f}%   | tau = {tau_val:>10.6f}           | {fp_count:,} / {N:,}")

    tau_max = float(np.max(attack_scores))
    thresholds["fpr_zero"] = round(tau_max + 1e-6, 6)
    print(f"{'0.000% (fpr_zero)':<15} | {'100.000%':<12} | tau = {tau_max:>10.6f}           | 0 / {N:,}")
    print("-" * 70)

    # Select recommended threshold: fpr_0.001 if samples >= 10k, else fpr_0.01
    rec_key = "fpr_0.001" if N >= 10000 else "fpr_0.01"
    rec_tau = thresholds.get(rec_key, thresholds["fpr_0.01"])

    print(f"\nRecommended Operational Threshold for Carrier Deployment: tau = {rec_tau:.6f} ({rec_key})")

    # Output JSON configuration
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    config_payload = {
        "calibration_dataset": args.benign_test,
        "sample_count": N,
        "model_weights": args.weights,
        "num_classes": num_classes,
        "logit_adjustment_enabled": args.adjust_logits,
        "real_benign_prior": args.real_benign_prior if args.adjust_logits else None,
        "tau_adjust": args.tau_adjust if args.adjust_logits else None,
        "noise_statistics": {
            "min": float(np.min(attack_scores)),
            "mean": float(np.mean(attack_scores)),
            "median": float(np.median(attack_scores)),
            "p95": float(np.percentile(attack_scores, 95.0)),
            "p99": float(np.percentile(attack_scores, 99.0)),
            "p99_9": float(np.percentile(attack_scores, 99.9)),
            "max": float(np.max(attack_scores)),
        },
        "thresholds": thresholds,
        "recommended_threshold": rec_tau,
        "recommended_target": rec_key,
    }

    with open(args.output, "w") as f:
        json.dump(config_payload, f, indent=2)
    print(f"Calibration configuration successfully written to: {args.output}")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plt.figure(figsize=(9, 5))
            plt.hist(attack_scores, bins=100, color="royalblue", alpha=0.75, edgecolor="black", linewidth=0.5)
            plt.axvline(thresholds["fpr_0.01"], color="orange", linestyle="--", linewidth=2, label=f"FPR 1% (tau={thresholds['fpr_0.01']:.4f})")
            plt.axvline(thresholds["fpr_0.001"], color="red", linestyle="-.", linewidth=2, label=f"FPR 0.1% (tau={thresholds['fpr_0.001']:.4f})")
            plt.axvline(rec_tau, color="darkgreen", linestyle=":", linewidth=2.5, label=f"Recommended (tau={rec_tau:.4f})")
            plt.title("Empirical Distribution of Attack Scores on Pure Benign Traffic", fontsize=12, fontweight="bold")
            plt.xlabel("Max Non-Benign Probability $\\max_{c \\geq 1} P(y=c \\mid x)$", fontsize=11)
            plt.ylabel("Sample Count", fontsize=11)
            plt.yscale("log")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plot_dir = os.path.dirname(args.plot)
            if plot_dir:
                os.makedirs(plot_dir, exist_ok=True)
            plt.savefig(args.plot, dpi=150)
            plt.close()
            print(f"Calibration plot saved to: {args.plot}")
        except Exception as e:
            print(f"Notice: Could not generate plot: {e}")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
