#!/usr/bin/env python3
"""
Raw PCAP Packet Header Sparsification Script (Version 4)
This script parses raw PCAP/PCAPNG network packet streams, extracts IP and transport headers,
constructs a sliding time-window of packets per flow, and writes the output in SVMLight format.

It implements a zero-copy, streaming output design that writes samples line-by-line
to prevent memory exhaustion when processing very large capture files.

Usage:
  python3 scripts/sparsify_pcap.py \
    --input datasets/capture.pcap \
    --output sparse-datasets/capture_sparse.txt \
    --window-size 10 \
    --default-label 6 \
    --local-subnet 192.168.0.0/16
"""

import argparse
import sys
import json
import ipaddress
from collections import deque
from scapy.all import PcapReader, IP, IPv6, TCP, UDP

# Packet dimensions
FEATURES_PER_PACKET = 1301

def get_packet_features(pkt, local_network):
    """
    Extracts the sparse active features for a single packet.
    Returns a list of integer feature indices (0-indexed relative to a single packet).
    """
    active_indices = []
    
    # 1. Check IP layer
    if pkt.haslayer(IP):
        ip_layer = pkt[IP]
        src_ip = ip_layer.src
        dst_ip = ip_layer.dst
        proto = ip_layer.proto
    elif pkt.haslayer(IPv6):
        ip_layer = pkt[IPv6]
        src_ip = ip_layer.src
        dst_ip = ip_layer.dst
        proto = ip_layer.nh
    else:
        # Non-IP packet, ignore or return empty
        return None, None
        
    # 1.1 IP Protocol OHE (Offset: 0 to 255)
    proto_idx = min(max(int(proto), 0), 255)
    active_indices.append(proto_idx)
    
    # 2. Check Transport layer
    sport = 0
    dport = 0
    tcp_flags_idx = []
    
    if pkt.haslayer(TCP):
        tcp_layer = pkt[TCP]
        sport = tcp_layer.sport
        dport = tcp_layer.dport
        flags = int(tcp_layer.flags)
        
        # 2.1 TCP Flags OHE (Offset: 256 to 263)
        # Flags bitmask: F (1), S (2), R (4), P (8), A (16), U (32), E (64), C (128)
        flag_bits = [1, 2, 4, 8, 16, 32, 64, 128]
        for idx, bit in enumerate(flag_bits):
            if flags & bit:
                tcp_flags_idx.append(256 + idx)
    elif pkt.haslayer(UDP):
        udp_layer = pkt[UDP]
        sport = udp_layer.sport
        dport = udp_layer.dport
        
    active_indices.extend(tcp_flags_idx)
    
    # 3. Source Port Type OHE (Offset: 264 to 266)
    if sport < 1024:
        active_indices.append(264 + 0)  # System
    elif sport < 49152:
        active_indices.append(264 + 1)  # Registered
    else:
        active_indices.append(264 + 2)  # Dynamic
        
    # 4. Destination Port OHE (Offset: 267 to 1291)
    if dport < 1024:
        active_indices.append(267 + dport)
    else:
        active_indices.append(267 + 1024)  # Dynamic/Registered bin
        
    # 5. Local Subnet Roles OHE (Offset: 1292 to 1295)
    try:
        src_is_local = ipaddress.ip_address(src_ip) in local_network
    except ValueError:
        src_is_local = False
        
    try:
        dst_is_local = ipaddress.ip_address(dst_ip) in local_network
    except ValueError:
        dst_is_local = False
        
    active_indices.append(1292 + (0 if src_is_local else 1))
    active_indices.append(1294 + (0 if dst_is_local else 1))
    
    # 6. Packet Length OHE (Offset: 1296 to 1300)
    pkt_len = len(pkt)
    if pkt_len <= 64:
        active_indices.append(1296 + 0)  # Tiny
    elif pkt_len <= 512:
        active_indices.append(1296 + 1)  # Small
    elif pkt_len <= 1000:
        active_indices.append(1296 + 2)  # Medium
    elif pkt_len <= 1500:
        active_indices.append(1296 + 3)  # Large
    else:
        active_indices.append(1296 + 4)  # Jumbo
        
    # 5-tuple flow key (bidirectional)
    flow_key = (
        min(src_ip, dst_ip),
        max(src_ip, dst_ip),
        min(sport, dport),
        max(sport, dport),
        proto
    )
    
    return active_indices, flow_key

def main():
    parser = argparse.ArgumentParser(description="Sparsify raw PCAP packet streams for SLIDE NIDS (Version 4)")
    parser.add_argument("--input", required=True, type=str, help="Path to input PCAP/PCAPNG file")
    parser.add_argument("--output", required=True, type=str, help="Path to save output sparse file (SVMLight)")
    parser.add_argument("--window-size", default=10, type=int, help="Size of sliding time-window (number of packets)")
    parser.add_argument("--default-label", default=6, type=int, help="Default label to assign (e.g. 6 for Normal)")
    parser.add_argument("--local-subnet", default="192.168.0.0/16", type=str, help="Local network IP block (e.g. 192.168.0.0/16)")
    parser.add_argument("--save-mapping", type=str, help="Path to save JSON file with feature column index mapping")
    args = parser.parse_args()
    
    try:
        local_network = ipaddress.ip_network(args.local_subnet)
    except ValueError as e:
        sys.exit(f"Error: Invalid local subnet block: {e}")
        
    print(f"Reading packets from: {args.input}")
    
    # Active flows window state dictionary
    # Key: bidirectional 5-tuple, Value: deque of maximum size window_size
    flows = {}
    
    # Counters
    packet_count = 0
    flow_samples_count = 0
    
    # Write line-by-line in SVMLight format
    with open(args.output, "w") as out_file:
        try:
            with PcapReader(args.input) as pcap_reader:
                for pkt in pcap_reader:
                    packet_count += 1
                    
                    if packet_count % 10000 == 0:
                        print(f"Processed {packet_count} packets...")
                        
                    # Extract single packet features and flow identifier
                    active_indices, flow_key = get_packet_features(pkt, local_network)
                    if active_indices is None:
                        continue  # Skip non-IP packets
                        
                    # Initialize or update flow queue
                    if flow_key not in flows:
                        flows[flow_key] = deque(maxlen=args.window_size)
                    
                    # Push current packet's indices
                    flows[flow_key].append(active_indices)
                    
                    # Construct window features (with zero-padding at the start if size < window_size)
                    window_indices = []
                    current_queue = flows[flow_key]
                    size_diff = args.window_size - len(current_queue)
                    
                    # Loop over steps in the window (0 represents oldest, window_size-1 represents newest)
                    for step_idx, pkt_indices in enumerate(current_queue):
                        # Shift indices forward based on the step position
                        shifted_indices = [idx + (step_idx + size_diff) * FEATURES_PER_PACKET for idx in pkt_indices]
                        window_indices.extend(shifted_indices)
                        
                    # Sort active indices (SVMLight requires indices to be in increasing order)
                    window_indices.sort()
                    
                    # Formulate SVMLight line
                    feature_str = " ".join([f"{idx}:1" for idx in window_indices])
                    out_file.write(f"{args.default_label} {feature_str}\n")
                    flow_samples_count += 1
                    
        except Exception as e:
            sys.exit(f"Error reading PCAP file: {e}")
            
    print(f"\nProcessing Complete!")
    print(f"Total packets read: {packet_count}")
    print(f"Total connection states exported (SVMLight): {flow_samples_count}")
    print(f"Sparsified output saved to: {args.output}")
    
    # Save mapping file if requested
    if args.save_mapping:
        feature_labels = {
            "proto": [0, 256],
            "tcp_flags": [256, 264],
            "sport_type": [264, 267],
            "dport": [267, 1292],
            "src_role": [1292, 1294],
            "dst_role": [1294, 1296],
            "pkt_len": [1296, 1301]
        }
        
        offsets = {}
        for step in range(args.window_size):
            step_offsets = {}
            for feat_name, (start, end) in feature_labels.items():
                step_offsets[feat_name] = [
                    start + step * FEATURES_PER_PACKET,
                    end + step * FEATURES_PER_PACKET
                ]
            offsets[f"step_{step}"] = step_offsets
            
        mapping_dict = {
            "total_features": FEATURES_PER_PACKET * args.window_size,
            "window_size": args.window_size,
            "offsets": offsets
        }
        with open(args.save_mapping, "w") as f:
            json.dump(mapping_dict, f, indent=4)
        print(f"Saved mapping metadata to: {args.save_mapping}")

if __name__ == "__main__":
    main()
