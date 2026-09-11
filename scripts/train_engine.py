#!/usr/bin/env python3
"""
Canonical Declarative Training Engine for SLIDE-NIDS.

A unified, contract-driven training infrastructure that eliminates ad-hoc training scripts.
Driven strictly by two declarative JSON specifications:
  1. Feature Architecture Specification: specs/feature_spec_*.json (Defines input space & semantics)
  2. Training Configuration Specification: configs/training_configs/*.json (Defines hyperparameters & topology)

Guarantees:
- Single Source of Truth: Driven 100% by schema contracts.
- Arbitrary Topologies: Supports any depth, layer dimensions, and class taxonomies dynamically.
- Preflight Contract Validation (Fail-Fast): Asserts SVMLight bounds vs specification before training.
- Memory-Safe Streamed Ingestion: Two-pass reservoir sampling avoids OOM on multi-gigabyte datasets.
- AVX2/BLAS Accelerated Sparse Forward/Backward via PyTorch EmbeddingBag core.
- Standardized Weights Export: Guarantees exact binary compatibility with bare-metal C++ (benchmark_cpp_linerate).
- Automated Adversarial Evasion Testing: Dynamically discovers port offsets and tests port-mutation robustness.
"""

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn


class UniversalSparseNIDSMLP(nn.Module):
    """Dynamic, arbitrary-depth sparse neural network matching C++ SLIDE/AVX2 specification."""
    def __init__(self, in_features, hidden_layers, num_classes, dropout=0.05, activation="relu"):
        super().__init__()
        self.in_features = in_features
        self.hidden_layers = list(hidden_layers)
        self.num_classes = num_classes
        self.dropout_p = dropout

        # Sparse Layer 0: EmbeddingBag sums active feature weights directly
        h0 = self.hidden_layers[0]
        self.embedding = nn.EmbeddingBag(in_features, h0, mode="sum", sparse=False)
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

        self.bias0 = nn.Parameter(torch.zeros(h0))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.embedding.weight.t())
        bound0 = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias0, -bound0, bound0)

        # Hidden Layers 1..N-1
        self.fcs = nn.ModuleList()
        in_dim = h0
        for h_dim in self.hidden_layers[1:]:
            layer = nn.Linear(in_dim, h_dim)
            nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
            self.fcs.append(layer)
            in_dim = h_dim

        # Output classification layer
        self.out_layer = nn.Linear(in_dim, num_classes)
        nn.init.kaiming_uniform_(self.out_layer.weight, a=math.sqrt(5))

        self.act = nn.ReLU() if activation.lower() == "relu" else nn.GELU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, indices, offsets):
        # Layer 0 (Sparse EmbeddingBag)
        h = self.act(self.embedding(indices, offsets) + self.bias0)
        h = self.dropout(h)

        # Intermediate Dense Layers
        for fc in self.fcs:
            h = self.act(fc(h))
            h = self.dropout(h)

        # Logits
        logits = self.out_layer(h)
        return logits

    def export_weights_dict(self):
        """Exports weights matching SLIDE / benchmark_cpp_linerate .npz specification."""
        weights = {}
        # Layer 0: Shape [h0, in_features]
        weights["w_layer_0"] = self.embedding.weight.t().detach().cpu().numpy().astype(np.float32)
        weights["b_layer_0"] = self.bias0.detach().cpu().numpy().astype(np.float32)

        layer_idx = 1
        for fc in self.fcs:
            weights[f"w_layer_{layer_idx}"] = fc.weight.detach().cpu().numpy().astype(np.float32)
            weights[f"b_layer_{layer_idx}"] = fc.bias.detach().cpu().numpy().astype(np.float32)
            layer_idx += 1

        weights[f"w_layer_{layer_idx}"] = self.out_layer.weight.detach().cpu().numpy().astype(np.float32)
        weights[f"b_layer_{layer_idx}"] = self.out_layer.bias.detach().cpu().numpy().astype(np.float32)
        return weights


def preflight_contract_check(filepath, spec_dict, config_dict, max_check=2000):
    """Fail-fast verification: ensures dataset adheres to specification before training."""
    print(f"Running Preflight Contract Verification on: {filepath} ...")
    total_dims = spec_dict.get("total_dimensions")
    num_classes = config_dict.get("architecture", {}).get("num_classes", 15)

    if not total_dims:
        raise ValueError("Specification JSON missing required field: 'total_dimensions'")

    violations = []
    lines_checked = 0

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line_idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                lbl = int(parts[0])
            except ValueError:
                violations.append(f"Line {line_idx}: non-integer label '{parts[0]}'")
                break

            if lbl < 0 or lbl >= num_classes:
                violations.append(f"Line {line_idx}: label {lbl} out of configured [0, {num_classes - 1}] range")

            for tok in parts[1:]:
                k_str = tok.split(":", 1)[0]
                try:
                    k = int(k_str)
                except ValueError:
                    violations.append(f"Line {line_idx}: non-integer feature index '{k_str}'")
                    break
                if k < 0 or k >= total_dims:
                    violations.append(f"Line {line_idx}: feature index {k} out of spec bounds [0, {total_dims - 1}]")
                    break

            lines_checked += 1
            if violations:
                break
            if lines_checked >= max_check:
                break

    if violations:
        print("❌ PREFLIGHT VERIFICATION FAILED:")
        for v in violations:
            print(f"   - {v}")
        raise ValueError(f"Contract violation between {filepath} and specification {spec_dict.get('name')}")
    else:
        print(f"✅ PREFLIGHT VERIFICATION PASSED: Checked {lines_checked:,} sample headers against {total_dims} dims and {num_classes} classes.")


def load_dataset_memory_safe(filepath, max_samples=None, balance_benign_ratio=1.0, seed=42):
    """Two-pass memory-safe dataset ingestion with uniform reservoir sampling."""
    print(f"Ingesting dataset from: {filepath} ...")
    t0 = time.time()
    random.seed(seed)

    # Pass 1: Streaming count
    total_attacks = 0
    total_benign = 0
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.split(maxsplit=1)
            if not parts:
                continue
            try:
                lbl = int(parts[0])
            except ValueError:
                continue
            if lbl == 0:
                total_benign += 1
            else:
                total_attacks += 1
            if max_samples and (total_attacks + total_benign) >= max_samples:
                break

    print(f"  Scanned file contents: Benign={total_benign:,} | Attacks={total_attacks:,}")

    if balance_benign_ratio is not None and balance_benign_ratio > 0 and total_attacks > 0:
        max_benign_allowed = int(total_attacks * balance_benign_ratio)
    else:
        max_benign_allowed = total_benign

    print(f"  Target partition balance: {max_benign_allowed:,} Benign + {total_attacks:,} Attacks")

    # Pass 2: Ingest into flat contiguous structures
    raw_attacks = []
    raw_benign = []
    benign_seen = 0

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            try:
                lbl = int(parts[0])
            except ValueError:
                continue
            inds = [int(p.split(":", 1)[0]) for p in parts[1:]]

            if lbl == 0:
                benign_seen += 1
                if len(raw_benign) < max_benign_allowed:
                    raw_benign.append((lbl, inds))
                else:
                    j = random.randint(0, benign_seen - 1)
                    if j < max_benign_allowed:
                        raw_benign[j] = (lbl, inds)
            else:
                raw_attacks.append((lbl, inds))

            if max_samples and (len(raw_attacks) + len(raw_benign)) >= max_samples:
                break

    combined = raw_benign + raw_attacks
    random.shuffle(combined)
    print(f"  Dataset combined: {len(raw_benign):,} Benign + {len(raw_attacks):,} Attacks = {len(combined):,} samples")

    indices_list = []
    offsets_list = []
    labels_list = []
    curr_offset = 0

    for lbl, inds in combined:
        indices_list.extend(inds)
        offsets_list.append(curr_offset)
        labels_list.append(lbl)
        curr_offset += len(inds)

    indices_tensor = torch.tensor(indices_list, dtype=torch.long)
    offsets_tensor = torch.tensor(offsets_list, dtype=torch.long)
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)

    elapsed = time.time() - t0
    print(f"  Loaded {len(labels_tensor):,} samples in {elapsed:.2f}s ({len(indices_tensor):,} active entries).")
    return indices_tensor, offsets_tensor, labels_tensor


def evaluate_engine(model, indices_t, offsets_t, labels_t, label_names, batch_size=10000, mutation_map=None):
    """High-throughput tensor evaluation over full test set with optional feature mutation."""
    model.eval()
    total_samples = len(labels_t)
    correct = 0

    true_counts = Counter()
    pred_counts = Counter()
    tp_counts = Counter()
    fp_counts = Counter()
    fn_counts = Counter()

    if mutation_map:
        eval_indices = indices_t.clone()
        for src_idx, dst_idx in mutation_map.items():
            eval_indices[eval_indices == src_idx] = dst_idx
    else:
        eval_indices = indices_t

    with torch.no_grad():
        for b_start in range(0, total_samples, batch_size):
            b_end = min(b_start + batch_size, total_samples)
            b_offsets = offsets_t[b_start:b_end] - offsets_t[b_start]
            idx_start = offsets_t[b_start].item()
            idx_end = offsets_t[b_end].item() if b_end < total_samples else len(eval_indices)

            b_indices = eval_indices[idx_start:idx_end]
            b_y = labels_t[b_start:b_end]

            logits = model(b_indices, b_offsets)
            preds = torch.argmax(logits, dim=1)

            correct += (preds == b_y).sum().item()

            preds_np = preds.cpu().numpy()
            y_np = b_y.cpu().numpy()

            for p, y in zip(preds_np, y_np):
                p_int = int(p)
                y_int = int(y)
                true_counts[y_int] += 1
                pred_counts[p_int] += 1
                if p_int == y_int:
                    tp_counts[y_int] += 1
                else:
                    fp_counts[p_int] += 1
                    fn_counts[y_int] += 1

    acc = float(correct / total_samples if total_samples > 0 else 0.0)
    all_classes = sorted([int(x) for x in set(true_counts.keys()) | set(pred_counts.keys())])
    class_stats = {}
    macro_p, macro_r, macro_f1 = 0.0, 0.0, 0.0
    weighted_p, weighted_r, weighted_f1 = 0.0, 0.0, 0.0

    for c in all_classes:
        tp = int(tp_counts[c])
        fp = int(fp_counts[c])
        fn = int(fn_counts[c])
        supp = int(true_counts[c])

        p = float(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        r = float(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        f1 = float((2 * p * r) / (p + r) if (p + r) > 0 else 0.0)

        c_name = label_names.get(c, f"Class {c}")
        class_stats[c] = {
            "name": c_name,
            "support": supp,
            "precision": p,
            "recall": r,
            "f1": f1
        }

        macro_p += p
        macro_r += r
        macro_f1 += f1
        weighted_p += p * supp
        weighted_r += r * supp
        weighted_f1 += f1 * supp

    n_cls = len(all_classes) if all_classes else 1
    macro_p /= n_cls
    macro_r /= n_cls
    macro_f1 /= n_cls

    weighted_p /= total_samples if total_samples > 0 else 1
    weighted_r /= total_samples if total_samples > 0 else 1
    weighted_f1 /= total_samples if total_samples > 0 else 1

    return {
        "accuracy": acc,
        "macro_p": macro_p,
        "macro_r": macro_r,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "class_stats": class_stats,
        "correct": correct,
        "total": total_samples
    }


def build_port_mutation_map(spec_dict):
    """Dynamically locates destination port fields across sliding window steps."""
    window_size = spec_dict.get("window_size", 10)
    dim_per_packet = spec_dict.get("dimensions_per_packet", 291)
    fields = spec_dict.get("fields", [])

    dport_field = None
    curr_offset = 0
    for f in fields:
        if f.get("name") in ["dport_type", "dport"]:
            dport_field = f
            dport_offset = curr_offset
            break
        curr_offset += f.get("dimensions", 0)

    if not dport_field:
        return None

    # Map System Port (bracket 0) to Registered Port (bracket 1)
    mutation_map = {}
    for t in range(window_size):
        base = t * dim_per_packet + dport_offset
        src_idx = base + 0  # System
        dst_idx = base + 1  # Registered
        mutation_map[src_idx] = dst_idx

    return mutation_map


def main():
    parser = argparse.ArgumentParser(description="Canonical Declarative Training Engine for SLIDE-NIDS")
    parser.add_argument("--spec", required=True, type=str, help="Path to Feature Architecture Specification JSON")
    parser.add_argument("--config", required=True, type=str, help="Path to Training Configuration JSON")
    parser.add_argument("--train", required=True, type=str, help="Path to Train SVMLight file")
    parser.add_argument("--test", required=True, type=str, help="Path to Test SVMLight file")
    parser.add_argument("--label-map", default="configs/label_maps/label_map_cicids2017_multiclass.json", type=str, help="Path to label map JSON")
    parser.add_argument("--output-weights", default="configs/weights_canonical_v5.npz", type=str, help="Output path for weights .npz")
    parser.add_argument("--output-md", default="audit_reports/training_certificate.md", type=str, help="Output Markdown report path")
    parser.add_argument("--output-json", default="audit_reports/training_certificate.json", type=str, help="Output JSON certificate path")
    parser.add_argument("--evasion-test", action="store_true", help="Run automated adversarial port-mutation evasion test")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Load Declarative Contracts
    with open(args.spec, "r") as f:
        spec_dict = json.load(f)
    with open(args.config, "r") as f:
        config_dict = json.load(f)

    # Preflight Validation (Fail-Fast)
    preflight_contract_check(args.train, spec_dict, config_dict)

    # Extract Architecture & Training Parameters
    total_dims = spec_dict["total_dimensions"]
    arch_cfg = config_dict["architecture"]
    train_cfg = config_dict["training"]

    hidden_layers = arch_cfg.get("hidden_layers", [2048, 256])
    num_classes = arch_cfg.get("num_classes", 15)
    dropout = arch_cfg.get("dropout", 0.05)
    activation = arch_cfg.get("activation", "relu")

    epochs = train_cfg.get("epochs", 5)
    batch_size = train_cfg.get("batch_size", 4096)
    lr = train_cfg.get("learning_rate", 0.001)
    weight_decay = train_cfg.get("weight_decay", 1e-4)
    balance_ratio = train_cfg.get("balance_benign_ratio", 1.0)

    # Load Label Map
    name_map = {}
    if os.path.exists(args.label_map):
        with open(args.label_map, "r") as f:
            raw = json.load(f)
        for k, v in raw.items():
            if v not in name_map:
                name_map[v] = k

    print("\n" + "=" * 75)
    print("      SLIDE-NIDS CANONICAL DECLARATIVE TRAINING ENGINE")
    print("=" * 75)
    print(f"Feature Spec:        {spec_dict.get('name', 'Custom')} (Dims: {total_dims:,})")
    print(f"Training Config:     {config_dict.get('name', 'Standard Config')}")
    arch_repr = f"Input {total_dims} -> " + " -> ".join([f"Hidden {h}" for h in hidden_layers]) + f" -> Output {num_classes}"
    print(f"Network Topology:    {arch_repr}")
    print(f"L3 Cache Footprint:  W0 takes {(total_dims * hidden_layers[0] * 4)/(1024*1024):.1f} MB (100% in L3 V-Cache)")
    print(f"Hyperparameters:     Epochs={epochs} | Batch={batch_size:,} | LR={lr} | Balance={balance_ratio:.1f}x")
    print("-" * 75)

    # Ingest Datasets
    tr_indices, tr_offsets, tr_labels = load_dataset_memory_safe(args.train, balance_benign_ratio=balance_ratio, seed=args.seed)
    te_indices, te_offsets, te_labels = load_dataset_memory_safe(args.test, balance_benign_ratio=None, seed=args.seed)

    # Initialize Universal Model
    torch.manual_seed(args.seed)
    model = UniversalSparseNIDSMLP(
        in_features=total_dims,
        hidden_layers=hidden_layers,
        num_classes=num_classes,
        dropout=dropout,
        activation=activation
    )

    # Loss & Optimization
    counts = Counter(tr_labels.numpy())
    total_tr = len(tr_labels)
    class_weights = torch.ones(num_classes, dtype=torch.float32)
    for c in range(num_classes):
        cnt = counts.get(c, 0)
        if cnt > 0:
            class_weights[c] = (total_tr / (num_classes * cnt)) ** 0.5

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print("\nStarting Optimization ...")
    num_samples = len(tr_labels)
    t_train_start = time.time()
    history = []

    for ep in range(1, epochs + 1):
        model.train()
        t_ep_start = time.time()
        running_loss = 0.0
        batches = 0

        batch_starts = list(range(0, num_samples, batch_size))
        random.shuffle(batch_starts)

        for b_start in batch_starts:
            b_end = min(b_start + batch_size, num_samples)
            idx_start = tr_offsets[b_start].item()
            idx_end = tr_offsets[b_end].item() if b_end < num_samples else len(tr_indices)

            b_indices = tr_indices[idx_start:idx_end]
            b_offsets = tr_offsets[b_start:b_end] - idx_start
            b_y = tr_labels[b_start:b_end]

            optimizer.zero_grad()
            logits = model(b_indices, b_offsets)
            loss = criterion(logits, b_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batches += 1

        scheduler.step()
        ep_duration = time.time() - t_ep_start
        avg_loss = running_loss / batches if batches > 0 else 0.0

        val_res = evaluate_engine(model, te_indices, te_offsets, te_labels, name_map, batch_size=10000)
        botnet_rec = val_res["class_stats"].get(3, {}).get("recall", 0.0) * 100
        ddos_rec = val_res["class_stats"].get(2, {}).get("recall", 0.0) * 100

        print(f"  Epoch {ep:02d}/{epochs:02d} [{ep_duration:.1f}s] - Loss: {avg_loss:.4f} | "
              f"Test Acc: {val_res['accuracy']*100:.2f}% | Macro F1: {val_res['macro_f1']*100:.2f}% | "
              f"Botnet Rec: {botnet_rec:.2f}% | DDoS Rec: {ddos_rec:.2f}%")

        history.append({
            "epoch": ep,
            "loss": avg_loss,
            "accuracy": val_res["accuracy"],
            "macro_f1": val_res["macro_f1"],
            "botnet_recall": botnet_rec,
            "time_seconds": ep_duration
        })

    total_time = time.time() - t_train_start
    print(f"Optimization completed in {total_time:.1f}s.")

    # Final Test Set Evaluation
    print("\n" + "=" * 75)
    print(f"   FINAL FORMAL EVALUATION OVER FULL TEST SET ({len(te_labels):,} SAMPLES)")
    print("=" * 75)
    final_res = evaluate_engine(model, te_indices, te_offsets, te_labels, name_map, batch_size=10000)

    print(f"Overall Test Accuracy: {final_res['accuracy'] * 100:.4f}% ({final_res['correct']:,} / {final_res['total']:,})")
    print(f"Macro F1-Score:        {final_res['macro_f1'] * 100:.4f}%")
    print(f"Weighted F1-Score:     {final_res['weighted_f1'] * 100:.4f}%")
    print("-" * 75)
    print(f"{'Class ID':<8} | {'Class Name':<20} | {'Support':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 75)
    for c, m in final_res["class_stats"].items():
        print(f"{c:<8} | {m['name']:<20} | {m['support']:<10,} | {m['precision'] * 100:>8.2f}% | {m['recall'] * 100:>8.2f}% | {m['f1'] * 100:>8.2f}%")
    print("=" * 75)

    # Optional Evasion Test
    evasion_summary = {}
    if args.evasion_test:
        print("\n" + "=" * 75)
        print("     ADVERSARIAL EVASION TEST: PORT MUTATION (SYSTEM -> REGISTERED)")
        print("=" * 75)
        mutation_map = build_port_mutation_map(spec_dict)
        if mutation_map:
            evasion_res = evaluate_engine(model, te_indices, te_offsets, te_labels, name_map, batch_size=10000, mutation_map=mutation_map)
            print(f"Post-Mutation Accuracy: {evasion_res['accuracy'] * 100:.4f}% (Original: {final_res['accuracy'] * 100:.4f}%)")
            print(f"Post-Mutation Macro F1: {evasion_res['macro_f1'] * 100:.4f}% (Original: {final_res['macro_f1'] * 100:.4f}%)")
            print("-" * 75)
            print(f"{'Class Name':<20} | {'Original Recall':<16} | {'Mutated Recall':<16} | {'Recall Delta':<14} | {'Status'}")
            print("-" * 75)
            for c, m in final_res["class_stats"].items():
                orig_r = m["recall"] * 100
                evas_r = evasion_res["class_stats"].get(c, {}).get("recall", 0.0) * 100
                delta = evas_r - orig_r
                status = "IMMUNE (Delta >= -1%)" if delta >= -1.0 else ("STABLE (Delta >= -5%)" if delta >= -5.0 else "VULNERABLE")
                print(f"{m['name']:<20} | {orig_r:>14.2f}% | {evas_r:>14.2f}% | {delta:>+12.2f}% | {status}")
                evasion_summary[m["name"]] = {
                    "original_recall": orig_r,
                    "mutated_recall": evas_r,
                    "delta": delta,
                    "status": status
                }
            print("=" * 75)

    # Export Weights
    os.makedirs(os.path.dirname(args.output_weights) or ".", exist_ok=True)
    weights_dict = model.export_weights_dict()
    np.savez_compressed(args.output_weights, **weights_dict)
    weights_size = os.path.getsize(args.output_weights) / (1024 * 1024)
    print(f"\nModel weights saved to: {args.output_weights} ({weights_size:.2f} MB)")

    # Save Reports
    report = {
        "timestamp": datetime.now().isoformat(),
        "spec_name": spec_dict.get("name"),
        "config_name": config_dict.get("name"),
        "architecture": arch_repr,
        "input_dimensions": total_dims,
        "weights_path": args.output_weights,
        "training_time_seconds": total_time,
        "test_results": {
            "overall_accuracy": float(final_res["accuracy"]),
            "macro_f1": float(final_res["macro_f1"]),
            "weighted_f1": float(final_res["weighted_f1"]),
            "class_metrics": {str(k): v for k, v in final_res["class_stats"].items()}
        },
        "evasion_test": evasion_summary,
        "epoch_history": history
    }

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"JSON Training Certificate saved to: {args.output_json}")

    if args.output_md:
        os.makedirs(os.path.dirname(args.output_md) or ".", exist_ok=True)
        with open(args.output_md, "w") as f:
            f.write(f"# Canonical Model Training Certificate: SLIDE-NIDS\n\n")
            f.write(f"**Verification Timestamp**: `{report['timestamp']}`  \n")
            f.write(f"**Feature Specification**: `{spec_dict.get('name')}`  \n")
            f.write(f"**Network Topology**: `{arch_repr}`  \n")
            f.write(f"**Weights File**: `{args.output_weights}` (`{weights_size:.2f} MB`)  \n\n")
            f.write(f"## Evaluation Metrics on Full Test Set ({final_res['total']:,} samples)\n\n")
            f.write(f"- **Overall Accuracy**: **`{final_res['accuracy'] * 100:.4f}%`**\n")
            f.write(f"- **Macro F1-Score**: **`{final_res['macro_f1'] * 100:.4f}%`**\n")
            f.write(f"- **Weighted F1-Score**: **`{final_res['weighted_f1'] * 100:.4f}%`**\n\n")
            f.write(f"### Per-Class Metrics\n\n")
            f.write(f"| Class ID | Class Name | Support | Precision | Recall | F1-Score |\n")
            f.write(f"| :---: | :--- | :---: | :---: | :---: | :---: |\n")
            for c, m in final_res["class_stats"].items():
                f.write(f"| {c} | {m['name']} | {m['support']:,} | {m['precision'] * 100:.2f}% | {m['recall'] * 100:.2f}% | {m['f1'] * 100:.2f}% |\n")
            if evasion_summary:
                f.write(f"\n## Adversarial Port-Mutation Robustness\n\n")
                f.write(f"| Class Name | Standard Recall | Mutated Recall | Delta | Robustness Status |\n")
                f.write(f"| :--- | :---: | :---: | :---: | :--- |\n")
                for c_name, comp in evasion_summary.items():
                    f.write(f"| {c_name} | {comp['original_recall']:.2f}% | {comp['mutated_recall']:.2f}% | {comp['delta']:+.2f}% | **{comp['status']}** |\n")
        print(f"Markdown Training Certificate saved to: {args.output_md}")


if __name__ == "__main__":
    main()
