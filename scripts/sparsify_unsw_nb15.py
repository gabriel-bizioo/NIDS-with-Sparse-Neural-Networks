#!/usr/bin/env python3
"""
UNSW-NB15 Dataset Sparsification Script
This script loads the UNSW-NB15 dataset (either raw CSV or split CSV files),
extracts and sparsifies selected features, and dumps the output in SVMLight format.

It supports three baseline configurations:
  - Baseline 1: proto, state, service, spkts, dpkts (178 dimensions without ports)
  - Baseline 2: Baseline 1 + sttl, dttl, swin, dwin (215 dimensions)
  - Baseline 3: Baseline 2 + ct_src_dport_ltm, ct_dst_sport_ltm, ct_dst_src_ltm (515 dimensions)

Usage:
  python3 scripts/sparsify_unsw_nb15.py --input datasets/UNSW-NB15/UNSW_NB15_training-set.csv --output sparse-datasets/UNSW-NB15/v1/sparse_unsw_train.txt --baseline 1
"""

import argparse
import sys
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.datasets import dump_svmlight_file

# Define exact category lists for consistency of dimensions across different splits
PORTS = [str(i) for i in range(65536)]
PROTOCOLS = [
    '3pc', 'a/n', 'aes-sp3-d', 'any', 'argus', 'aris', 'arp', 'ax.25', 'bbn-rcc', 'bna', 
    'br-sat-mon', 'cbt', 'cftp', 'chaos', 'compaq-peer', 'cphb', 'cpnx', 'crtp', 'crudp', 
    'dcn', 'ddp', 'ddx', 'dgp', 'egp', 'eigrp', 'emcon', 'encap', 'etherip', 'fc', 'fire', 
    'ggp', 'gmtp', 'gre', 'hmp', 'i-nlsp', 'iatp', 'ib', 'icmp', 'idpr', 'idpr-cmtp', 
    'idrp', 'ifmp', 'igmp', 'igp', 'il', 'ip', 'ipcomp', 'ipcv', 'ipip', 'iplt', 'ipnip', 
    'ippc', 'ipv6', 'ipv6-frag', 'ipv6-no', 'ipv6-opts', 'ipv6-route', 'ipx-n-ip', 'irtp', 
    'isis', 'iso-ip', 'iso-tp4', 'kryptolan', 'l2tp', 'larp', 'leaf-1', 'leaf-2', 'merit-inp', 
    'mfe-nsp', 'mhrp', 'micp', 'mobile', 'mtp', 'mux', 'narp', 'netblt', 'nsfnet-igp', 
    'nvp', 'ospf', 'pgm', 'pim', 'pipe', 'pnni', 'pri-enc', 'prm', 'ptp', 'pup', 'pvp', 
    'qnx', 'rdp', 'rsvp', 'rtp', 'rvd', 'sat-expak', 'sat-mon', 'sccopmce', 'scps', 'sctp', 
    'sdrp', 'secure-vmtp', 'sep', 'skip', 'sm', 'smp', 'snp', 'sprite-rpc', 'sps', 'srp', 
    'st2', 'stp', 'sun-nd', 'swipe', 'tcf', 'tcp', 'tlsp', 'tp++', 'trunk-1', 'trunk-2', 
    'ttp', 'udp', 'unas', 'uti', 'vines', 'visa', 'vmtp', 'vrrp', 'wb-expak', 'wb-mon', 
    'wsn', 'xnet', 'xns-idp', 'xtp', 'zero'
]
STATES = ['ACC', 'CLO', 'CON', 'ECO', 'FIN', 'INT', 'PAR', 'REQ', 'RST', 'URN', 'no']
SERVICES = ['-', 'dhcp', 'dns', 'ftp', 'ftp-data', 'http', 'irc', 'pop3', 'radius', 'smtp', 'snmp', 'ssh', 'ssl']

# Categories for Baseline 2 (Header details)
STTL_CATEGORIES = [str(i) for i in [0, 1, 29, 30, 31, 32, 60, 62, 63, 64, 252, 254, 255]]
DTTL_CATEGORIES = [str(i) for i in [0, 29, 30, 31, 32, 60, 252, 253, 254]]
SWIN_CATEGORIES = [str(i) for i in [0, 5, 14, 31, 42, 43, 45, 52, 67, 87, 99, 103, 154, 156, 167, 168, 172, 192, 202, 232, 245, 255]]
DWIN_CATEGORIES = [str(i) for i in [0, 12, 27, 33, 37, 40, 46, 48, 70, 77, 81, 125, 137, 160, 164, 171, 209, 244, 255]]

# Categories for Baseline 3 (Rolling window counters, max connections is 100)
CT_CATEGORIES = [str(i) for i in range(1, 101)]

# Define bins and labels for Spkts and Dpkts
BINS_SPKTS = [0, 2, 5, 10, 20, 50, 100, 200, 500, 1000, float('inf')]
LABELS_SPKTS = [
    'spkts_1_2', 'spkts_3_5', 'spkts_6_10', 'spkts_11_20', 'spkts_21_50', 
    'spkts_51_100', 'spkts_101_200', 'spkts_201_500', 'spkts_501_1000', 'spkts_1000plus'
]

BINS_DPKTS = [-1, 0, 2, 5, 10, 20, 50, 100, 200, 500, 1000, float('inf')]
LABELS_DPKTS = [
    'dpkts_0', 'dpkts_1_2', 'dpkts_3_5', 'dpkts_6_10', 'dpkts_11_20', 'dpkts_21_50', 
    'dpkts_51_100', 'dpkts_101_200', 'dpkts_201_500', 'dpkts_501_1000', 'dpkts_1000plus'
]

def main():
    parser = argparse.ArgumentParser(description="Sparsify UNSW-NB15 dataset for SLIDE NIDS classifier")
    parser.add_argument("--input", required=True, type=str, help="Path to input UNSW-NB15 CSV file")
    parser.add_argument("--output", required=True, type=str, help="Path to save output sparse file (SVMLight format)")
    parser.add_argument("--baseline", required=True, type=int, choices=[1, 2, 3], help="Baseline configuration version (1, 2, or 3)")
    parser.add_argument("--window-size", default=1, type=int, help="Number of consecutive flows to include in sliding window")
    parser.add_argument("--label-col", default="label", choices=["label", "attack_cat"], help="Column to use as output label")
    parser.add_argument("--save-mapping", type=str, help="Path to save JSON file with feature column index mapping")
    parser.add_argument("--save-label-mapping", type=str, help="Path to save JSON file with label mapping")
    args = parser.parse_args()

    print(f"Reading dataset: {args.input} (Baseline {args.baseline})")
    
    # Read CSV file header to check column presence
    header = pd.read_csv(args.input, nrows=0)
    col_map = {c.lower().strip(): c for c in header.columns}
    
    # Define required columns based on baseline
    baseline_features = []
    
    # Baseline 1 core features
    baseline_features.extend(['proto', 'state', 'service', 'spkts', 'dpkts'])
    
    # Baseline 2 features
    if args.baseline >= 2:
        baseline_features.extend(['sttl', 'dttl', 'swin', 'dwin'])
        
    # Baseline 3 features
    if args.baseline >= 3:
        baseline_features.extend(['ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm'])

    # Check if 'sport' is present (optional, typically missing in pre-split CSVs)
    has_sport = 'sport' in col_map
    required_cols = []
    if has_sport:
        required_cols.append(col_map['sport'])
        
    # Check other features
    for col_name in baseline_features + [args.label_col]:
        if col_name in col_map:
            required_cols.append(col_map[col_name])
        else:
            sys.exit(f"Error: Required column '{col_name}' (case-insensitive) not found in the dataset.")
            
    # Load dataset with only the needed columns
    df = pd.read_csv(args.input, usecols=required_cols, low_memory=False)
    df.columns = [c.lower().strip() for c in df.columns]
    
    # Clean and fill missing values
    df['proto'] = df['proto'].fillna('-').astype(str)
    df['state'] = df['state'].fillna('-').astype(str)
    df['service'] = df['service'].fillna('-').astype(str)
    df['spkts'] = pd.to_numeric(df['spkts'], errors='coerce').fillna(0).astype(int)
    df['dpkts'] = pd.to_numeric(df['dpkts'], errors='coerce').fillna(0).astype(int)
    
    if has_sport:
        df['sport'] = pd.to_numeric(df['sport'], errors='coerce').fillna(0).astype(int).astype(str)
        
    if args.baseline >= 2:
        df['sttl'] = pd.to_numeric(df['sttl'], errors='coerce').fillna(0).astype(int).astype(str)
        df['dttl'] = pd.to_numeric(df['dttl'], errors='coerce').fillna(0).astype(int).astype(str)
        df['swin'] = pd.to_numeric(df['swin'], errors='coerce').fillna(0).astype(int).astype(str)
        df['dwin'] = pd.to_numeric(df['dwin'], errors='coerce').fillna(0).astype(int).astype(str)
        
    if args.baseline >= 3:
        df['ct_src_dport_ltm'] = pd.to_numeric(df['ct_src_dport_ltm'], errors='coerce').fillna(0).astype(int).astype(str)
        df['ct_dst_sport_ltm'] = pd.to_numeric(df['ct_dst_sport_ltm'], errors='coerce').fillna(0).astype(int).astype(str)
        df['ct_dst_src_ltm'] = pd.to_numeric(df['ct_dst_src_ltm'], errors='coerce').fillna(0).astype(int).astype(str)

    # Binning step for Spkts and Dpkts
    print("Performing binning of packet counts...")
    df['spkts_bin'] = pd.cut(df['spkts'], bins=BINS_SPKTS, labels=LABELS_SPKTS).astype(str)
    df['dpkts_bin'] = pd.cut(df['dpkts'], bins=BINS_DPKTS, labels=LABELS_DPKTS).astype(str)
    
    # Combine feature vectors to one-hot encode dynamically
    cols_to_encode = []
    categories = []
    
    if has_sport:
        cols_to_encode.append('sport')
        categories.append(PORTS)
        
    cols_to_encode.extend(['proto', 'state', 'service', 'spkts_bin', 'dpkts_bin'])
    categories.extend([PROTOCOLS, STATES, SERVICES, LABELS_SPKTS, LABELS_DPKTS])
    
    if args.baseline >= 2:
        cols_to_encode.extend(['sttl', 'dttl', 'swin', 'dwin'])
        categories.extend([STTL_CATEGORIES, DTTL_CATEGORIES, SWIN_CATEGORIES, DWIN_CATEGORIES])
        
    if args.baseline >= 3:
        cols_to_encode.extend(['ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm'])
        categories.extend([CT_CATEGORIES, CT_CATEGORIES, CT_CATEGORIES])
        
    features_to_encode = df[cols_to_encode]
    
    # Configure One-Hot Encoder with fixed categories
    print("Fitting One-Hot Encoder...")
    ohe = OneHotEncoder(categories=categories, sparse_output=True, handle_unknown='ignore')
    sparse_features = ohe.fit_transform(features_to_encode)
    
    # Process Labels
    print(f"Encoding label column '{args.label_col}'...")
    label_col_data = df[args.label_col.lower()].fillna('Normal')
    
    if args.label_col == 'label':
        # Binary labels
        labels = label_col_data.astype(int).values
        label_mapping = {"0": 0, "1": 1}
    else:
        # Multiclass labels
        # Use a fixed list of classes to ensure identical mapping across training and test splits
        attack_classes = [
            'Analysis', 'Backdoor', 'DoS', 'Exploits', 'Fuzzers', 
            'Generic', 'Normal', 'Reconnaissance', 'Shellcode', 'Worms'
        ]
        le = LabelEncoder()
        le.fit(attack_classes)
        cleaned_labels = label_col_data.astype(str).str.strip()
        cleaned_labels = cleaned_labels.apply(lambda x: x if x in attack_classes else 'Normal')
        labels = le.transform(cleaned_labels)
        label_mapping = {str(cls): int(idx) for idx, cls in enumerate(le.classes_)}
        
    # Apply sliding time-window if window_size > 1
    if args.window_size > 1:
        print(f"Applying sliding time-window of size N = {args.window_size}...")
        from scipy.sparse import vstack, hstack, csr_matrix
        num_samples, num_features = sparse_features.shape
        cols = []
        for j in range(args.window_size - 1, -1, -1):
            if j == 0:
                cols.append(sparse_features)
            else:
                pad = csr_matrix((j, num_features))
                shifted = sparse_features[:(num_samples - j)]
                shifted_full = vstack([pad, shifted], format='csr')
                cols.append(shifted_full)
        sparse_features = hstack(cols, format='csr')

    print(f"Features shape: {sparse_features.shape}")
    print(f"Label classes counts: {len(label_mapping)} classes")
    
    # Dump to SVMLight format
    print(f"Saving sparse matrix to SVMLight format: {args.output}")
    dump_svmlight_file(sparse_features, labels, args.output, zero_based=True)
    
    # Save metadata mapping if requested
    if args.save_mapping:
        feature_names = ohe.get_feature_names_out(cols_to_encode)
        
        # Dynamically compute feature offsets per step
        offsets = {}
        features_per_step = len(feature_names)
        
        if args.window_size > 1:
            for step in range(args.window_size):
                step_offsets = {}
                current_offset = step * features_per_step
                for col_name, cat_list in zip(cols_to_encode, categories):
                    step_offsets[col_name] = [current_offset, current_offset + len(cat_list)]
                    current_offset += len(cat_list)
                offsets[f"step_{step}"] = step_offsets
            total_features = features_per_step * args.window_size
        else:
            current_offset = 0
            for col_name, cat_list in zip(cols_to_encode, categories):
                offsets[col_name] = [current_offset, current_offset + len(cat_list)]
                current_offset += len(cat_list)
            total_features = features_per_step
            
        mapping_dict = {
            "total_features": total_features,
            "window_size": args.window_size,
            "offsets": offsets
        }
        with open(args.save_mapping, 'w') as f:
            json.dump(mapping_dict, f, indent=4)
        print(f"Saved feature offsets mapping to: {args.save_mapping}")
        
    if args.save_label_mapping:
        with open(args.save_label_mapping, 'w') as f:
            json.dump(label_mapping, f, indent=4)
        print(f"Saved label mappings to: {args.save_label_mapping}")

    print("Success!")

if __name__ == "__main__":
    main()
