#!/usr/bin/env python3
"""
Generalized PCAP Sparsification Engine for SLIDE NIDS.
Reads a declarative JSON feature architecture specification and extracts high-dimensional
sparse vectors (SVMLight) from raw PCAP packet streams over sliding time-windows.

Supports:
- Dynamic feature architectures (V4, V5, or custom user-defined schemas).
- Ground-truth labeling via CICFlowMeter CSVs or attack schedule JSON rules.
- Deterministic canonical 5-tuple flow-disjoint splitting (--train-output / --test-output).
- Stride sampling for uniform temporal coverage over long PCAP traces.
"""

import argparse
import hashlib
import ipaddress
import json
import os
import sys
from collections import deque
from scapy.all import PcapReader, IP, TCP, UDP

class FeatureExtractor:
    """Dynamically compiles and extracts features based on a JSON specification."""
    def __init__(self, spec_dict):
        self.spec = spec_dict
        self.window_size = self.spec.get("window_size", 10)
        self.local_subnet = ipaddress.ip_network(self.spec.get("local_subnet", "192.168.0.0/16"))
        self.fields = self.spec.get("fields", [])
        
        # Calculate field offsets within a single packet's feature block
        self.field_offsets = {}
        curr_offset = 0
        for f in self.fields:
            self.field_offsets[f["name"]] = curr_offset
            curr_offset += f["dimensions"]
            
        self.dim_per_packet = curr_offset
        self.total_dimensions = self.dim_per_packet * self.window_size
        
        # Validate specification internal consistency
        expected_total = self.spec.get("total_dimensions")
        if expected_total and expected_total != self.total_dimensions:
            raise ValueError(f"Specification mismatch: computed {self.total_dimensions} total dims, but spec says {expected_total}")

    def extract_packet_features(self, pkt, flow_state):
        """Extracts active feature indices for a single packet relative to [0, dim_per_packet)."""
        if not pkt.haslayer(IP):
            return None, None, None

        ip_layer = pkt[IP]
        src_ip = ip_layer.src
        dst_ip = ip_layer.dst
        proto = ip_layer.proto
        pkt_len = len(pkt)
        pkt_time = float(pkt.time)

        sport, dport = 0, 0
        tcp_flags_val = 0
        seq_num, ack_num = 0, 0

        if proto == 6 and pkt.haslayer(TCP):
            tcp_layer = pkt[TCP]
            sport = tcp_layer.sport
            dport = tcp_layer.dport
            tcp_flags_val = int(tcp_layer.flags)
            seq_num = int(tcp_layer.seq)
            ack_num = int(tcp_layer.ack)
        elif proto == 17 and pkt.haslayer(UDP):
            udp_layer = pkt[UDP]
            sport = udp_layer.sport
            dport = udp_layer.dport

        # Canonical bidirectional 5-tuple
        flow_key = (min(src_ip, dst_ip), max(src_ip, dst_ip), min(sport, dport), max(sport, dport), proto)
        flow_info = {
            "timestamp": pkt_time,
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "sport": sport,
            "dport": dport,
            "proto": proto
        }

        active_indices = []

        for f in self.fields:
            name = f["name"]
            ftype = f["type"]
            base = self.field_offsets[name]

            if name == "ip_proto":
                active_indices.append(base + (proto % 256))

            elif name == "tcp_flags":
                # Check individual TCP flags: SYN, ACK, FIN, RST, PSH, URG, ECE, CWR
                flag_masks = [0x02, 0x10, 0x01, 0x04, 0x08, 0x20, 0x40, 0x80]
                for bit_idx, mask in enumerate(flag_masks):
                    if tcp_flags_val & mask:
                        active_indices.append(base + bit_idx)

            elif name == "sport_type":
                # System, Registered, Dynamic
                if sport < 1024:
                    active_indices.append(base + 0)
                elif sport < 49152:
                    active_indices.append(base + 1)
                else:
                    active_indices.append(base + 2)

            elif name == "dport":
                # Well-known [0-1023] + dynamic category (1024)
                if dport <= 1023:
                    active_indices.append(base + dport)
                else:
                    active_indices.append(base + 1024)

            elif name == "dport_type":
                # Destination Port Category (System, Registered, Dynamic)
                if dport < 1024:
                    active_indices.append(base + 0)
                elif dport < 49152:
                    active_indices.append(base + 1)
                else:
                    active_indices.append(base + 2)

            elif name == "subnet_roles":
                # Directionality: src is local/ext, dst is local/ext
                try:
                    src_obj = ipaddress.ip_address(src_ip)
                    dst_obj = ipaddress.ip_address(dst_ip)
                    src_local = src_obj in self.local_subnet
                    dst_local = dst_obj in self.local_subnet
                except ValueError:
                    src_local, dst_local = False, False

                active_indices.append(base + (0 if src_local else 1))
                active_indices.append(base + 2 + (0 if dst_local else 1))

            elif name == "pkt_len":
                # Binned packet length
                if pkt_len <= 64:
                    active_indices.append(base + 0)
                elif pkt_len <= 128:
                    active_indices.append(base + 1)
                elif pkt_len <= 512:
                    active_indices.append(base + 2)
                elif pkt_len <= 1024:
                    active_indices.append(base + 3)
                else:
                    active_indices.append(base + 4)

            elif name == "ip_ttl":
                ttl = int(getattr(ip_layer, "ttl", 64))
                if ttl <= 64:
                    active_indices.append(base + 0)
                elif ttl <= 128:
                    active_indices.append(base + 1)
                elif ttl <= 255:
                    active_indices.append(base + 2)
                else:
                    active_indices.append(base + 3)

            elif name == "seq_delta":
                last_seq = flow_state.get("last_seq", seq_num)
                delta = (seq_num - last_seq) & 0xFFFFFFFF
                flow_state["last_seq"] = seq_num
                if delta == 0:
                    active_indices.append(base + 0)
                elif delta <= 1460:
                    active_indices.append(base + 1)
                elif delta <= 65535:
                    active_indices.append(base + 2)
                else:
                    active_indices.append(base + 3)

            elif name == "ack_delta":
                last_ack = flow_state.get("last_ack", ack_num)
                delta = (ack_num - last_ack) & 0xFFFFFFFF
                flow_state["last_ack"] = ack_num
                if delta == 0:
                    active_indices.append(base + 0)
                elif delta <= 1460:
                    active_indices.append(base + 1)
                elif delta <= 65535:
                    active_indices.append(base + 2)
                else:
                    active_indices.append(base + 3)

        return active_indices, flow_key, flow_info


def load_labels_csv(csv_path):
    """Loads 5-tuple labels from pre-labeled CICFlowMeter CSV file."""
    if csv_path.lower().endswith('.json'):
        sys.exit(
            f"\n[Error] Invalid argument for --labels-csv: '{csv_path}' is a JSON file!\n"
            f"  * '--labels-csv' expects a CSV flow record file (e.g., /home/bizio/Repos/HashingDeepLearning/dataset/CICIDS2017_improved/friday.csv).\n"
            f"  * If you meant to provide the class integer mapping, pass it with '--label-map {csv_path}'.\n"
        )
    print(f"Loading flow labels from CSV: {csv_path} ...")
    lookup = {}
    with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
        header_line = f.readline().strip()
        if not header_line:
            sys.exit(f"[Error] Empty CSV file: {csv_path}")
        header = [h.strip() for h in header_line.split(',')]
        
        col_map = {col.lower(): idx for idx, col in enumerate(header)}
        
        def find_col(*candidates):
            for cand in candidates:
                if cand.lower() in col_map:
                    return col_map[cand.lower()]
            return None

        src_ip_idx = find_col('Source IP', 'Src IP', 'src_ip', 'source_ip')
        dst_ip_idx = find_col('Destination IP', 'Dst IP', 'dst_ip', 'destination_ip')
        src_port_idx = find_col('Source Port', 'Src Port', 'src_port', 'source_port')
        dst_port_idx = find_col('Destination Port', 'Dst Port', 'dst_port', 'destination_port')
        proto_idx = find_col('Protocol', 'proto')
        label_idx = find_col('Label', 'label')
        
        missing = []
        if src_ip_idx is None: missing.append("Source IP / Src IP")
        if dst_ip_idx is None: missing.append("Destination IP / Dst IP")
        if src_port_idx is None: missing.append("Source Port / Src Port")
        if dst_port_idx is None: missing.append("Destination Port / Dst Port")
        if proto_idx is None: missing.append("Protocol")
        if label_idx is None: missing.append("Label")
        
        if missing:
            sys.exit(
                f"\n[Error] CSV file '{csv_path}' is missing required columns: {missing}\n"
                f"Detected header columns: {header[:12]}...\n"
            )

        max_idx = max(src_ip_idx, dst_ip_idx, src_port_idx, dst_port_idx, proto_idx, label_idx)
        
        for line in f:
            parts = [p.strip() for p in line.strip().split(',')]
            if len(parts) <= max_idx:
                continue
            try:
                sip = parts[src_ip_idx]
                dip = parts[dst_ip_idx]
                sport = int(parts[src_port_idx])
                dport = int(parts[dst_port_idx])
                proto = int(parts[proto_idx])
                label = parts[label_idx]
                
                key = (min(sip, dip), max(sip, dip), min(sport, dport), max(sport, dport), proto)
                lookup[key] = label
            except Exception:
                continue
    print(f"Loaded {len(lookup):,} unique flow labels from CSV.")
    return lookup


def resolve_label(flow_info, flow_key, csv_lookup, attack_schedule, label_map, default_label):
    """Maps flow metadata to integer class ID."""
    label_str = None
    if csv_lookup and flow_key in csv_lookup:
        label_str = csv_lookup[flow_key]
    elif attack_schedule:
        pkt_time = flow_info["timestamp"]
        src_ip = flow_info["src_ip"]
        dst_ip = flow_info["dst_ip"]
        for rule in attack_schedule:
            if rule.get("start_time", 0.0) <= pkt_time <= rule.get("end_time", float("inf")):
                if (rule.get("src_ip") is None or rule.get("src_ip") == src_ip) and \
                   (rule.get("dst_ip") is None or rule.get("dst_ip") == dst_ip):
                    label_str = rule.get("label")
                    break
    if label_str is not None:
        if label_map and label_str in label_map:
            return label_map[label_str], label_str
        elif str(label_str).isdigit():
            return int(label_str), str(label_str)
        else:
            return default_label, label_str
    return default_label, "BENIGN/Default"


def is_flow_train(flow_key, train_ratio=0.8):
    """Deterministic canonical 5-tuple MD5 hash partition."""
    key_str = f"{flow_key[0]}_{flow_key[1]}_{flow_key[2]}_{flow_key[3]}_{flow_key[4]}"
    hash_digest = hashlib.md5(key_str.encode('utf-8')).hexdigest()
    bucket = int(hash_digest, 16) % 100
    return bucket < int(train_ratio * 100)


def main():
    parser = argparse.ArgumentParser(description="Generalized PCAP Sparsifier for SLIDE NIDS")
    parser.add_argument("--spec", required=True, type=str, help="Path to feature specification JSON file")
    parser.add_argument("--input", required=True, type=str, help="Path to input PCAP/PCAPNG file")
    parser.add_argument("--output", type=str, help="Single output SVMLight path")
    parser.add_argument("--train-output", type=str, help="Flow-disjoint train output SVMLight path")
    parser.add_argument("--test-output", type=str, help="Flow-disjoint test output SVMLight path")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Ratio of flows assigned to train (default: 0.8)")
    parser.add_argument("--labels-csv", type=str, help="Pre-labeled CICFlowMeter CSV file")
    parser.add_argument("--attack-schedule", type=str, help="JSON file containing attack time windows")
    parser.add_argument("--label-map", type=str, help="JSON file mapping string labels to integer IDs")
    parser.add_argument("--default-label", type=int, default=0, help="Default integer label (default: 0)")
    parser.add_argument("--max-packets", type=int, default=None, help="Maximum number of packets to process")
    parser.add_argument("--stride", type=int, default=1, help="Process every N-th packet (default: 1)")
    parser.add_argument("--flow-manifest", type=str, default=None, help="Path to save JSON manifest of partitioned flow keys")
    args = parser.parse_args()

    if not args.output and not (args.train_output and args.test_output):
        sys.exit("Error: Must specify either --output OR both --train-output and --test-output")

    with open(args.spec, 'r') as f:
        spec_dict = json.load(f)
    extractor = FeatureExtractor(spec_dict)

    print(f"Loaded Feature Architecture: {spec_dict.get('name', 'Custom')}")
    print(f"  Dimensions per packet: {extractor.dim_per_packet}")
    print(f"  Window size:           {extractor.window_size} packets")
    print(f"  Total input dimensions:{extractor.total_dimensions}")

    if args.label_map and args.label_map.lower().endswith('.csv'):
        sys.exit(
            f"\n[Error] Invalid argument for --label-map: '{args.label_map}' is a CSV file!\n"
            f"  * '--label-map' expects a JSON file mapping label strings to integer IDs (e.g., scripts/label_map_cicids2017_multiclass.json).\n"
            f"  * If you meant to provide the flow CSV dataset, use '--labels-csv {args.label_map}'.\n"
        )

    csv_lookup = load_labels_csv(args.labels_csv) if args.labels_csv else {}
    attack_schedule = json.load(open(args.attack_schedule)) if args.attack_schedule else []
    label_map = json.load(open(args.label_map)) if args.label_map else {}

    flows = {}
    train_flows_seen = set()
    test_flows_seen = set()

    packet_count = 0
    samples_written = 0
    train_samples_written = 0
    test_samples_written = 0
    label_counts = {}

    out_file = open(args.output, "w") if args.output else None
    train_file = open(args.train_output, "w") if args.train_output else None
    test_file = open(args.test_output, "w") if args.test_output else None

    print(f"\nReading packets from: {args.input} (stride={args.stride}) ...")
    try:
        with PcapReader(args.input) as reader:
            for pkt in reader:
                packet_count += 1
                if args.stride > 1 and packet_count % args.stride != 0:
                    continue
                if args.max_packets and samples_written >= args.max_packets:
                    break
                if packet_count % 50000 == 0:
                    print(f"  Processed {packet_count:,} packets... ({samples_written:,} window samples generated)")

                if not pkt.haslayer(IP):
                    continue

                ip_layer = pkt[IP]
                src_ip, dst_ip, proto = ip_layer.src, ip_layer.dst, ip_layer.proto
                sport, dport = 0, 0
                if proto == 6 and pkt.haslayer(TCP):
                    sport, dport = pkt[TCP].sport, pkt[TCP].dport
                elif proto == 17 and pkt.haslayer(UDP):
                    sport, dport = pkt[UDP].sport, pkt[UDP].dport

                flow_key = (min(src_ip, dst_ip), max(src_ip, dst_ip), min(sport, dport), max(sport, dport), proto)

                if flow_key not in flows:
                    flows[flow_key] = {
                        "queue": deque(maxlen=extractor.window_size),
                        "last_seq": 0,
                        "last_ack": 0
                    }

                active_indices, _, flow_info = extractor.extract_packet_features(pkt, flows[flow_key])
                if active_indices is None:
                    continue

                flows[flow_key]["queue"].append(active_indices)
                if len(flows[flow_key]["queue"]) < extractor.window_size:
                    continue

                # Concatenate sliding window
                window_indices = []
                for step_idx, step_indices in enumerate(flows[flow_key]["queue"]):
                    offset = step_idx * extractor.dim_per_packet
                    for idx in step_indices:
                        window_indices.append(offset + idx)

                window_indices.sort()
                label_id, label_str = resolve_label(flow_info, flow_key, csv_lookup, attack_schedule, label_map, args.default_label)
                label_counts[f"{label_id} ({label_str})"] = label_counts.get(f"{label_id} ({label_str})", 0) + 1

                feature_str = " ".join([f"{idx}:1" for idx in window_indices])
                line = f"{label_id} {feature_str}\n"

                if out_file:
                    out_file.write(line)
                    samples_written += 1
                else:
                    if is_flow_train(flow_key, args.train_ratio):
                        train_file.write(line)
                        train_flows_seen.add(flow_key)
                        train_samples_written += 1
                    else:
                        test_file.write(line)
                        test_flows_seen.add(flow_key)
                        test_samples_written += 1
                    samples_written += 1

    finally:
        if out_file: out_file.close()
        if train_file: train_file.close()
        if test_file: test_file.close()

    print(f"\nProcessing Complete!")
    print(f"Total packets read:             {packet_count:,}")
    print(f"Total window samples generated: {samples_written:,}")
    if train_file:
        print(f"  Train samples: {train_samples_written:,} ({len(train_flows_seen):,} unique flows)")
        print(f"  Test samples:  {test_samples_written:,} ({len(test_flows_seen):,} unique flows)")
        overlap = train_flows_seen.intersection(test_flows_seen)
        assert len(overlap) == 0, f"ERROR: Data leakage detected! {len(overlap)} overlapping flows!"
        print("  Flow Disjointness Check: PASSED (0 overlapping flows).")
        if args.flow_manifest:
            os.makedirs(os.path.dirname(args.flow_manifest) or ".", exist_ok=True)
            manifest_payload = {
                "input_pcap": args.input,
                "train_output": args.train_output,
                "test_output": args.test_output,
                "train_ratio": args.train_ratio,
                "total_flows": len(train_flows_seen) + len(test_flows_seen),
                "train_flows_count": len(train_flows_seen),
                "test_flows_count": len(test_flows_seen),
                "overlapping_flows_count": len(overlap),
                "flow_disjointness_passed": (len(overlap) == 0),
                "train_flows": [f"{f[0]}_{f[1]}_{f[2]}_{f[3]}_{f[4]}" for f in train_flows_seen],
                "test_flows": [f"{f[0]}_{f[1]}_{f[2]}_{f[3]}_{f[4]}" for f in test_flows_seen]
            }
            with open(args.flow_manifest, "w") as mf:
                json.dump(manifest_payload, mf, indent=2)
            print(f"  Flow Partition Manifest saved to: {args.flow_manifest}")

    print("\nLabel Distribution Summary:")
    for lbl, cnt in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (cnt / samples_written) * 100 if samples_written > 0 else 0
        print(f"  Label {lbl}: {cnt:,} samples ({pct:.2f}%)")


if __name__ == "__main__":
    main()
