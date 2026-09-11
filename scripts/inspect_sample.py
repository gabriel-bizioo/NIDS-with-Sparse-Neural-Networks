#!/usr/bin/env python3
"""
Human-Readable SVMLight Dataset & Model Prediction Inspector.

Decodes raw high-dimensional SVMLight feature vectors into human-readable network packet attributes:
- Translates sparse indices into protocol names, destination ports, TCP flags, packet size brackets, and subnet directions.
- Shows packet-by-packet sliding window progression (Packets 1 to 10).
- Runs model inference on that exact sample and compares Ground Truth vs Model Prediction.

Usage:
  # Inspect sample by line number:
  python3 scripts/inspect_sample.py \
    --file datasets/CICIDS2017/friday/friday_composite_test.txt \
    --line 100 \
    --weights ../HashingDeepLearning/dataset/CICIDS2017_improved/savedWeights_v4_bottleneck_15class.npz

  # Inspect first occurrence of a specific class (e.g. Portscan = 1, Botnet = 3, DDoS = 2):
  python3 scripts/inspect_sample.py \
    --file datasets/CICIDS2017/friday/friday_composite_test.txt \
    --class-id 1 \
    --weights ../HashingDeepLearning/dataset/CICIDS2017_improved/savedWeights_v4_bottleneck_15class.npz
"""

import argparse
import json
import os
import sys
import numpy as np

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

TCP_FLAG_NAMES = {
    0: "FIN",
    1: "SYN",
    2: "RST",
    3: "PSH",
    4: "ACK",
    5: "URG",
    6: "ECE",
    7: "CWR"
}

SPORT_TYPES = {
    0: "System / Well-Known (0-1023)",
    1: "Registered (1024-49151)",
    2: "Dynamic / Ephemeral (49152-65535)"
}

SUBNET_ROLES = {
    0: "Internal -> Internal (Local-to-Local)",
    1: "Internal -> External (Outbound Internet)",
    2: "External -> Internal (Inbound from Internet)",
    3: "External -> External"
}

PKT_LENS = {
    0: "Tiny (<= 64 bytes, e.g. empty TCP ACK/SYN probe)",
    1: "Small (65-128 bytes, e.g. DNS / Handshake)",
    2: "Medium (129-512 bytes, e.g. HTTP request / NTP)",
    3: "Large (513-1024 bytes, e.g. TLS Hello / Data chunk)",
    4: "Jumbo (> 1024 bytes, MTU full payload segment)"
}


def decode_packet_features(active_indices, dim_per_packet=1301):
    """Decodes active indices within one packet block [0, 1301) into English descriptions."""
    info = {
        "proto": "Unknown",
        "flags": [],
        "sport_type": "Unknown",
        "dport": "Unknown",
        "subnet": "Unknown",
        "pkt_len": "Unknown"
    }

    for idx in sorted(active_indices):
        if 0 <= idx < 256:
            if idx == 6:
                info["proto"] = "TCP (6)"
            elif idx == 17:
                info["proto"] = "UDP (17)"
            elif idx == 1:
                info["proto"] = "ICMP (1)"
            else:
                info["proto"] = f"Proto {idx}"
        elif 256 <= idx < 264:
            flag_bit = idx - 256
            info["flags"].append(TCP_FLAG_NAMES.get(flag_bit, f"Flag_{flag_bit}"))
        elif 264 <= idx < 267:
            info["sport_type"] = SPORT_TYPES.get(idx - 264, "Unknown")
        elif 267 <= idx < 1292:
            port_val = idx - 267
            if port_val < 1024:
                # Common service names
                service_names = {80: "HTTP", 443: "HTTPS", 21: "FTP", 22: "SSH", 53: "DNS", 25: "SMTP", 445: "SMB"}
                name = service_names.get(port_val, "")
                info["dport"] = f"Port {port_val} ({name})" if name else f"Port {port_val}"
            else:
                info["dport"] = "Dynamic / Ephemeral (> 1023)"
        elif 1292 <= idx < 1296:
            info["subnet"] = SUBNET_ROLES.get(idx - 1292, "Unknown")
        elif 1296 <= idx < 1301:
            info["pkt_len"] = PKT_LENS.get(idx - 1296, "Unknown")

    return info


def inspect_sample_line(line_str, weights_path=None, sample_idx=None):
    parts = line_str.strip().split()
    if not parts:
        return

    label_id = int(parts[0])
    label_name = LABEL_NAMES.get(label_id, f"Class {label_id}")

    feature_pairs = []
    for item in parts[1:]:
        if ":" in item:
            k, v = item.split(":", 1)
            feature_pairs.append((int(k), float(v)))

    # Group by packet (window size = 10, 1301 dims per packet)
    dim_per_packet = 1301
    packets = [[] for _ in range(10)]
    for k, v in feature_pairs:
        pkt_idx = k // dim_per_packet
        rel_idx = k % dim_per_packet
        if 0 <= pkt_idx < 10:
            packets[pkt_idx].append(rel_idx)

    print("=" * 80)
    idx_str = f"Sample #{sample_idx} | " if sample_idx is not None else ""
    print(f"      {idx_str}GROUND TRUTH LABEL: {label_id} ({label_name.upper()})")
    print("=" * 80)
    print(f"Total Non-Zero Features: {len(feature_pairs)} / 13,010  (Sparsity: {100.0 - (len(feature_pairs)/13010.0*100):.2f}% zeros)")
    print("-" * 80)
    print(f"{'Packet':<8} | {'Protocol':<10} | {'TCP Flags':<18} | {'Dest Port':<20} | {'Packet Size Bracket'}")
    print("-" * 80)

    for i in range(10):
        if not packets[i]:
            print(f"Pkt {i+1:<4} | [No packet - connection has < 10 packets]")
            continue
        dec = decode_packet_features(packets[i], dim_per_packet)
        flags_str = "[" + ", ".join(dec["flags"]) + "]" if dec["flags"] else "[None]"
        print(f"Pkt {i+1:<4} | {dec['proto']:<10} | {flags_str:<18} | {dec['dport']:<20} | {dec['pkt_len']}")

    print("-" * 80)

    # If weights are provided, compute model forward pass and prediction
    if weights_path and os.path.exists(weights_path):
        data = np.load(weights_path)
        layers = []
        l_idx = 0
        while f"w_layer_{l_idx}" in data:
            layers.append((data[f"w_layer_{l_idx}"], data[f"b_layer_{l_idx}"]))
            l_idx += 1

        # Layer 0
        W0, b0 = layers[0]
        h0 = np.zeros(W0.shape[0], dtype=np.float32)
        for k, v in feature_pairs:
            if k < W0.shape[1]:
                h0 += W0[:, k] * v
        h0 = np.maximum(0, h0 + b0)

        # Intermediate layers
        h = h0
        for l in range(1, len(layers) - 1):
            W, b = layers[l]
            h = np.maximum(0, np.dot(h, W.T) + b)

        # Output logits
        W_last, b_last = layers[-1]
        z = np.dot(h, W_last.T) + b_last
        probs = np.exp(z - np.max(z))
        probs = probs / np.sum(probs)

        pred_id = int(np.argmax(z))
        pred_name = LABEL_NAMES.get(pred_id, f"Class {pred_id}")
        confidence = probs[pred_id] * 100.0

        match_str = "[CORRECT PREDICTION]" if pred_id == label_id else "[MISCLASSIFICATION]"
        print(f"MODEL INFERENCE VERIFICATION:")
        print(f"  Model Output Prediction:  Class {pred_id} ({pred_name}) with {confidence:.2f}% confidence  --> {match_str}")
        print(f"  Top Class Probabilities:")
        top_indices = np.argsort(probs)[::-1][:4]
        for c in top_indices:
            print(f"    - {LABEL_NAMES.get(c, f'Class {c}'):<26}: {probs[c]*100:>6.2f}% (Logit: {z[c]:+.2f})")

    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Human-Readable Dataset & Model Inspector")
    parser.add_argument("--file", required=True, type=str, help="Path to SVMLight file")
    parser.add_argument("--line", default=None, type=int, help="1-indexed line number to inspect")
    parser.add_argument("--class-id", default=None, type=int, help="Inspect first sample matching this class ID")
    parser.add_argument("--weights", default=None, type=str, help="Path to saved weights (.npz) to run inference on this sample")
    args = parser.parse_args()

    target_line = args.line
    found = False

    with open(args.file, "r") as f:
        for idx, line in enumerate(f, start=1):
            if not line.strip():
                continue
            lbl = int(line.split()[0])
            if args.class_id is not None and lbl == args.class_id:
                inspect_sample_line(line, args.weights, sample_idx=idx)
                found = True
                break
            elif target_line is not None and idx == target_line:
                inspect_sample_line(line, args.weights, sample_idx=idx)
                found = True
                break

    if not found:
        print(f"Could not find sample matching criteria in {args.file}")


if __name__ == "__main__":
    main()
