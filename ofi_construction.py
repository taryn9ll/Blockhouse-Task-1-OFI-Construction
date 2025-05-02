import argparse
import logging
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def load_data(path: str) -> pd.DataFrame:
    """Load LOB event data from CSV."""
    logging.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    return df


def compute_ofi_features(df: pd.DataFrame, max_depth: int) -> pd.DataFrame:
    """
    Compute OFI features:
      - Best-Level OFI
      - Multi-Level OFI
      - Integrated OFI via PCA
      - Cross-Asset OFI
    """
    # Sort and reset index
    df = df.sort_values(['symbol', 'ts_event']).reset_index(drop=True)
    
    # Shift previous sizes per symbol and level
    for level in range(max_depth):
        lvl = f"{level:02d}"
        df[f'bid_sz_{lvl}_prev'] = df.groupby('symbol')[f'bid_sz_{lvl}'].shift(1)
        df[f'ask_sz_{lvl}_prev'] = df.groupby('symbol')[f'ask_sz_{lvl}'].shift(1)
    
    # Compute per-level OFI and collect column names
    ofi_cols = []
    for level in range(max_depth):
        lvl = f"{level:02d}"
        bid_diff = df[f'bid_sz_{lvl}'] - df[f'bid_sz_{lvl}_prev']
        ask_diff = df[f'ask_sz_{lvl}'] - df[f'ask_sz_{lvl}_prev']
        col = f'ofi_lvl_{lvl}'
        df[col] = bid_diff.fillna(0) - ask_diff.fillna(0)
        ofi_cols.append(col)
    
    # Best-Level OFI
    df['best_level_ofi'] = df[ofi_cols[0]]

    # Multi-Level OFI
    df['multi_level_ofi'] = df[ofi_cols].sum(axis=1)

    # Integrated OFI via PCA
    scaler = StandardScaler()
    ofi_matrix = scaler.fit_transform(df[ofi_cols].fillna(0))
    pca = PCA(n_components=1)
    df['integrated_ofi'] = pca.fit_transform(ofi_matrix).flatten()

    # Cross-Asset OFI: total minus own per timestamp
    df['total_integrated'] = df.groupby('ts_event')['integrated_ofi'].transform('sum')
    df['cross_asset_ofi'] = df['total_integrated'] - df['integrated_ofi']
    df.drop(columns=['total_integrated'], inplace=True)

    # Deduplicate: one row per (symbol, ts_event)
    agg_funcs = {
        'best_level_ofi': 'first',
        'multi_level_ofi': 'first',
        'integrated_ofi': 'first',
        'cross_asset_ofi': 'first'
    }
    df_out = df.groupby(['symbol', 'ts_event'], as_index=False).agg(agg_funcs)
    return df_out


def save_output(df: pd.DataFrame, path: str):
    """Save features to CSV."""
    logging.info(f"Saving features to {path}")
    df.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Compute OFI features from LOB data.")
    parser.add_argument(
        '--input', '-i', default='first_25000_rows.csv',
        help='Input CSV path with LOB event data.'
    )
    parser.add_argument(
        '--output', '-o', default='ofi_features.csv',
        help='Output CSV path for OFI features.'
    )
    parser.add_argument(
        '--max-depth', '-d', type=int, default=10,
        help='Number of depth levels to use when computing OFI.'
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Resolve paths relative to script directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, args.input)
    output_path = os.path.join(base_dir, args.output)

    df = load_data(input_path)
    features = compute_ofi_features(df, args.max_depth)
    save_output(features, output_path)

    logging.info("OFI feature construction completed.")


if __name__ == '__main__':
    main()
