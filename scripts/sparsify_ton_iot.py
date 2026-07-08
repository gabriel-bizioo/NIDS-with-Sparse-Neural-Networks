#!/usr/bin/env python3
"""
ToN-IoT and Linux Dataset Sparsification Script
This script provides a generic and robust pipeline to sparsify any ToN-IoT telemetry 
or Linux OS logs dataset. It automatically bins numerical columns, one-hot encodes 
categorical columns, and outputs the result in SVMLight format.

To ensure consistency between training and testing sets, it saves the fitted encoder 
state (including bin edges and OHE categories) during the training run, and reloads 
it during the testing run.

Usage (Train):
  python3 scripts/sparsify_ton_iot.py --input datasets/Train_Test_IoT_dataset/Train_Test_IoT_Fridge.csv --output sparse-datasets/Train_Test_IoT_Fridge_train.txt --save-encoder sparse-datasets/Train_Test_IoT_Fridge_encoder.pkl --save-mapping sparse-datasets/Train_Test_IoT_Fridge_mapping.json

Usage (Test):
  python3 scripts/sparsify_ton_iot.py --input datasets/Train_Test_IoT_dataset/Train_Test_IoT_Fridge.csv --output sparse-datasets/Train_Test_IoT_Fridge_test.txt --load-encoder sparse-datasets/Train_Test_IoT_Fridge_encoder.pkl
"""

import argparse
import pickle
import sys
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.datasets import dump_svmlight_file

# Fixed list of all possible ToN-IoT attack categories (multiclass) to ensure stable target labels
TON_IOT_CLASSES = ['normal', 'backdoor', 'ddos', 'dos', 'injection', 'mitm', 'password', 'ransomware', 'scanning', 'xss']

def clean_data(df, target_col):
    """
    Standardizes column casing, drops metadata, and separates target labels.
    """
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    col_map = {c.lower(): c for c in df.columns}
    
    # Identify target column case-insensitively
    if target_col.lower() not in col_map:
        sys.exit(f"Error: Target column '{target_col}' not found in dataset columns: {list(df.columns)}")
    actual_target_col = col_map[target_col.lower()]
    
    # Exclude metadata columns
    metadata_cols = ['date', 'time', 'ts', 'id', 'label']  # 'label' is the binary target; we use 'type' for multiclass
    cols_to_exclude = [col_map[m] for m in metadata_cols if m in col_map and col_map[m] != actual_target_col]
    
    feature_df = df.drop(columns=cols_to_exclude, errors='ignore')
    labels = df[actual_target_col].astype(str).str.strip().str.lower()
    
    # Also separate actual_target_col from features
    if actual_target_col in feature_df.columns:
        feature_df = feature_df.drop(columns=[actual_target_col])
        
    return feature_df, labels

def process_features(df, is_train, config=None):
    """
    Bins numerical features and converts categorical features to strings.
    If training, learns the bin edges. If testing, re-uses them.
    """
    processed_df = pd.DataFrame(index=df.index)
    bin_edges_dict = {} if is_train else config.get('bin_edges', {})
    categorical_cols = []
    numerical_cols = []
    
    # Define columns to treat as categorical (strings or low-cardinality integers)
    for col in df.columns:
        # Check if column is non-numeric or low cardinality
        if df[col].dtype == 'object' or df[col].dtype == 'bool' or df[col].nunique() < 15:
            categorical_cols.append(col)
        else:
            # Special case for spatial data like GPS coordinates: round to 3 decimal places and treat as categorical
            if col.lower() in ['latitude', 'longitude']:
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)
                
    # Process categorical columns
    for col in categorical_cols:
        if col.lower() in ['latitude', 'longitude']:
            # Round GPS coordinates to 3 decimals (~110m grid) for OHE
            processed_df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).round(3).astype(str)
        else:
            processed_df[col] = df[col].fillna('Missing').astype(str).str.strip()
            
    # Process numerical columns
    for col in numerical_cols:
        series = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
        if is_train:
            # Determine bin boundaries using quantiles (up to 10 bins)
            try:
                # Add tiny noise to avoid duplicate bin edges on low-variance numeric columns
                noise = np.random.normal(0, 1e-9, size=len(series))
                _, edges = pd.qcut(series + noise, q=10, retbins=True, duplicates='drop')
                # Ensure outer bounds cover -inf and inf
                edges[0] = -float('inf')
                edges[-1] = float('inf')
            except Exception:
                # Fallback to equal-width bins if qcut fails
                min_val = series.min()
                max_val = series.max()
                if min_val == max_val:
                    edges = [-float('inf'), min_val, float('inf')]
                else:
                    edges = list(np.linspace(min_val, max_val, 11))
                    edges[0] = -float('inf')
                    edges[-1] = float('inf')
            bin_edges_dict[col] = edges
        else:
            edges = bin_edges_dict.get(col, [-float('inf'), float('inf')])
            
        # Perform binning
        labels = [f"{col}_bin_{i}" for i in range(len(edges) - 1)]
        processed_df[f"{col}_bin"] = pd.cut(series, bins=edges, labels=labels, include_lowest=True).astype(str)
        
    return processed_df, bin_edges_dict

def main():
    parser = argparse.ArgumentParser(description="Sparsify ToN-IoT / Linux logs datasets dynamically")
    parser.add_argument("--input", required=True, type=str, help="Path to input CSV file")
    parser.add_argument("--output", required=True, type=str, help="Path to save output sparse file (SVMLight)")
    parser.add_argument("--target-col", default="type", help="Column to use as label (e.g., 'type' for multiclass, 'label' for binary)")
    parser.add_argument("--save-encoder", type=str, help="Path to save fitted encoder state (pickle)")
    parser.add_argument("--load-encoder", type=str, help="Path to load fitted encoder state (pickle)")
    parser.add_argument("--save-mapping", type=str, help="Path to save JSON file with feature column index mapping")
    args = parser.parse_args()
    
    if args.save_encoder and args.load_encoder:
        sys.exit("Error: Cannot specify both --save-encoder and --load-encoder.")
        
    print(f"Loading dataset: {args.input}")
    df = pd.read_csv(args.input, low_memory=False)
    
    feature_df, raw_labels = clean_data(df, args.target_col)
    
    is_train = args.load_encoder is None
    config = {}
    
    if not is_train:
        print(f"Loading encoder state from: {args.load_encoder}")
        with open(args.load_encoder, 'rb') as f:
            config = pickle.load(f)
            
    # Process features (handles categorical string conversions and numerical binning)
    print("Processing features (binning numerical values and standardizing)...")
    processed_features, bin_edges = process_features(feature_df, is_train, config)
    
    # One-Hot Encode features
    if is_train:
        print("Fitting One-Hot Encoder...")
        ohe = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
        sparse_features = ohe.fit_transform(processed_features)
        config['ohe'] = ohe
        config['bin_edges'] = bin_edges
        config['feature_columns'] = list(processed_features.columns)
    else:
        ohe = config['ohe']
        # Reorder/align columns to match the training state columns
        aligned_features = pd.DataFrame(index=processed_features.index)
        for col in config['feature_columns']:
            if col in processed_features.columns:
                aligned_features[col] = processed_features[col]
            else:
                aligned_features[col] = 'Missing'
        sparse_features = ohe.transform(aligned_features)
        
    # Encode target labels
    print(f"Encoding label column: {args.target_col}")
    if is_train:
        le = LabelEncoder()
        # Fit on fixed TON_IOT_CLASSES if multiclass, else binary
        if args.target_col.lower() in ['type', 'attack_cat']:
            le.fit(TON_IOT_CLASSES)
            # Fallback unknown classes to 'normal'
            cleaned_labels = raw_labels.apply(lambda x: x if x in TON_IOT_CLASSES else 'normal')
            labels = le.transform(cleaned_labels)
        else:
            labels = pd.to_numeric(raw_labels, errors='coerce').fillna(0).astype(int).values
            le.fit([0, 1])
            
        label_mapping = {str(cls): int(idx) for idx, cls in enumerate(le.classes_)}
        config['le'] = le
        config['label_mapping'] = label_mapping
    else:
        le = config['le']
        if args.target_col.lower() in ['type', 'attack_cat']:
            cleaned_labels = raw_labels.apply(lambda x: x if x in TON_IOT_CLASSES else 'normal')
            labels = le.transform(cleaned_labels)
        else:
            labels = pd.to_numeric(raw_labels, errors='coerce').fillna(0).astype(int).values
            
        label_mapping = config['label_mapping']
        
    print(f"Features matrix shape: {sparse_features.shape}")
    print(f"Sparsity: {100 * (1 - sparse_features.nnz / (sparse_features.shape[0] * sparse_features.shape[1])):.4f}%")
    print(f"Label classes: {len(label_mapping)} classes")
    
    # Save sparse data in SVMLight format
    print(f"Saving sparse matrix to: {args.output}")
    dump_svmlight_file(sparse_features, labels, args.output, zero_based=True)
    
    # Save encoder state if requested
    if args.save_encoder:
        print(f"Saving encoder state to: {args.save_encoder}")
        with open(args.save_encoder, 'wb') as f:
            pickle.dump(config, f)
            
    # Save mapping metadata if requested
    if args.save_mapping and is_train:
        feature_names = ohe.get_feature_names_out(config['feature_columns'])
        mapping_dict = {
            "total_features": len(feature_names),
            "label_mapping": label_mapping,
            "feature_categories": {}
        }
        
        # Determine index offsets for each feature class
        current_offset = 0
        for col in config['feature_columns']:
            num_categories = len(ohe.categories_[config['feature_columns'].index(col)])
            mapping_dict["feature_categories"][col] = [current_offset, current_offset + num_categories]
            current_offset += num_categories
            
        with open(args.save_mapping, 'w') as f:
            json.dump(mapping_dict, f, indent=4)
        print(f"Saved feature mappings metadata to: {args.save_mapping}")
        
    print("Success!")

if __name__ == "__main__":
    main()
