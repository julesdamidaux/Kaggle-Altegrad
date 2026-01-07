#!/usr/bin/env python3
"""
Generate a CSV file with SMILES for test graph examples.

Usage:
    python scripts/generate_test_smiles_csv.py --pkl test_graphs.pkl --output test_smiles.csv --n 1000
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_baseline.data_utils import PreprocessedGraphDataset, x_map, e_map
from Hamza.chem.pyg_to_smiles import pyg_data_to_smiles


def main():
    parser = argparse.ArgumentParser(description='Generate CSV with SMILES for test set')
    parser.add_argument('--pkl', type=str, required=True,
                        help='Path to .pkl file containing preprocessed graphs')
    parser.add_argument('--output', type=str, default='test_smiles.csv',
                        help='Path to output CSV file (default: test_smiles.csv)')
    parser.add_argument('--n', type=int, default=None,
                        help='Number of examples to process (default: all)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    pkl_path = Path(args.pkl)
    if not pkl_path.exists():
        print(f"Error: File not found: {pkl_path}")
        sys.exit(1)
    
    print(f"Loading dataset from: {pkl_path}")
    print(f"Output will be saved to: {args.output}")
    print("=" * 80)
    
    # Load dataset
    try:
        dataset = PreprocessedGraphDataset(str(pkl_path))
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    
    # Determine number of examples to process
    if args.n is None:
        n = len(dataset)
        print(f"Processing all {n} molecules")
    else:
        n = min(args.n, len(dataset))
        print(f"Processing first {n} molecules")
    print("=" * 80)
    
    # Lists to store data
    ids = []
    smiles_list = []
    
    # Process each molecule
    for i in range(n):
        try:
            # Get graph data
            data = dataset[i]
            
            # Get molecule info
            mol_id = getattr(data, 'id', f'unknown_{i}')
            
            # Convert to SMILES
            smiles = pyg_data_to_smiles(data, x_map, e_map, canonical=True)
            
            # Append to lists
            ids.append(mol_id)
            smiles_list.append(smiles)
            
            # Print progress
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{n} molecules...")
            
        except Exception as e:
            # Log error but continue
            mol_id = getattr(dataset[i], 'id', f'unknown_{i}')
            print(f"  Warning: Failed to process molecule {i} (ID: {mol_id}): {str(e)}")
            ids.append(mol_id)
            smiles_list.append('')
    
    # Create DataFrame
    df = pd.DataFrame({
        'ID': ids,
        'SMILES': smiles_list
    })
    
    # Save to CSV
    try:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print("=" * 80)
        print(f"✓ Successfully saved CSV with {len(df)} molecules")
        print(f"  Output: {output_path}")
        print(f"\nDataFrame shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nFirst 5 rows:")
        print(df.head(5).to_string())
    except Exception as e:
        print(f"Error saving CSV: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
