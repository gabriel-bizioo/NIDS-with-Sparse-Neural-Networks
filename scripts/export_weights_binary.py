#!/usr/bin/env python3
"""
Exports weights from numpy (.npz) format to flat binary (.bin) format
for high-speed, dependency-free loading in Eigen 3 C++ engine.
"""

import argparse
import json
import os
import struct
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
    return delta.astype(np.float32)

def export_weights(npz_path: str, bin_path: str, threshold_config: str = None, adjust_logits: bool = False, real_benign_prior: float = 0.9999):
    data = np.load(npz_path)
    w0 = data["w_layer_0"].astype(np.float32)  # (2048, 2910)
    b0 = data["b_layer_0"].astype(np.float32)  # (2048,)
    w1 = data["w_layer_1"].astype(np.float32)  # (256, 2048)
    b1 = data["b_layer_1"].astype(np.float32)  # (256,)
    w2 = data["w_layer_2"].astype(np.float32)  # (15, 256)
    b2 = data["b_layer_2"].astype(np.float32)  # (15,)

    h0_dim, input_dim = w0.shape
    h1_dim, _ = w1.shape
    num_classes, _ = w2.shape

    if threshold_config and os.path.exists(threshold_config):
        with open(threshold_config, "r") as f:
            cfg = json.load(f)
        if cfg.get("logit_adjustment_enabled", False):
            prior = cfg.get("real_benign_prior", 0.9999)
            tau = cfg.get("tau_adjust", 1.0)
            delta = compute_logit_adjustment(num_classes, "balanced", prior, tau)
            b2 = b2 - delta
            print(f"Absorbed Bayesian Logit Adjustment Delta into b2 bias (prior={prior}, tau={tau})")
    elif adjust_logits:
        delta = compute_logit_adjustment(num_classes, "balanced", real_benign_prior, 1.0)
        b2 = b2 - delta
        print(f"Absorbed Bayesian Logit Adjustment Delta into b2 bias (prior={real_benign_prior}, tau=1.0)")

    with open(bin_path, "wb") as f:
        # Magic (4B) + Version (1) + Dimensions (4 uint32)
        f.write(b"NIDS")
        header = struct.pack("<5I", 1, input_dim, h0_dim, h1_dim, num_classes)
        f.write(header)
        # W0 written in Fortran (Column-Major) order for contiguous column gather
        f.write(w0.tobytes(order='F'))
        f.write(b0.tobytes(order='C'))
        # W1 and W2 written in C (Row-Major) order for cache-friendly GEMV
        f.write(w1.tobytes(order='C'))
        f.write(b1.tobytes(order='C'))
        f.write(w2.tobytes(order='C'))
        f.write(b2.tobytes(order='C'))

    print(f"Successfully exported: {npz_path} -> {bin_path}")
    print(f"Topology: {input_dim} -> {h0_dim} -> {h1_dim} -> {num_classes}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export NPZ weights to Eigen binary format")
    parser.add_argument("--npz", default="configs/weights_v5_port_agnostic.npz", help="Input NPZ path")
    parser.add_argument("--bin", default="configs/weights_v5.bin", help="Output BIN path")
    parser.add_argument("--threshold-config", default=None, help="Optional path to operational_threshold.json to absorb delta")
    parser.add_argument("--adjust-logits", action="store_true", help="Absorb Bayesian logit adjustment directly")
    parser.add_argument("--real-benign-prior", default=0.9999, type=float, help="Real-world benign prior")
    args = parser.parse_args()
    export_weights(args.npz, args.bin, args.threshold_config, args.adjust_logits, args.real_benign_prior)
