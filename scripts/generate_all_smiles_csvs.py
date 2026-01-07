#!/usr/bin/env python3
"""
Generate CSV files with SMILES and descriptions for train, validation, and test datasets.

Usage:
    python scripts/generate_all_smiles_csvs.py --data_dir ./data_baseline/data --output_dir ./
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_baseline.data_utils import PreprocessedGraphDataset, x_map, e_map
from Hamza.chem.pyg_to_smiles import pyg_data_to_smiles


def generate_csv_for_split(pkl_path, output_path, split_name):
    """
    Generate CSV for a single dataset split.
    
    Args:
        pkl_path: Path to .pkl file
        output_path: Path to output CSV file
        split_name: Name of the split (train/validation/test)
    """
    print(f"\n{'='*80}")
    print(f"Processing {split_name.upper()} dataset")
    print(f"{'='*80}")
    
    # Check if file exists
    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        print(f"Warning: File not found: {pkl_path}, skipping...")
        return False
    
    print(f"Loading dataset from: {pkl_path}")
    
    # Load dataset
    try:
        dataset = PreprocessedGraphDataset(str(pkl_path))
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False
    
    n = len(dataset)
    print(f"Processing {n} molecules...")
    
    # Lists to store data
    ids = []
    smiles_list = []
    descriptions = []
    
    # Process each molecule
    for i in range(n):
        try:
            # Get graph data
            data = dataset[i]
            
            # Get molecule info
            mol_id = getattr(data, 'id', f'unknown_{i}')
            description = getattr(data, 'description', '')
            
            # Convert to SMILES
            smiles = pyg_data_to_smiles(data, x_map, e_map, canonical=True)
            
            # Append to lists
            ids.append(mol_id)
            smiles_list.append(smiles)
            descriptions.append(description)
            
            # Print progress
            if (i + 1) % 500 == 0:
                print(f"  Processed {i + 1}/{n} molecules...")
            
        except Exception as e:
            # Log error but continue
            mol_id = getattr(dataset[i], 'id', f'unknown_{i}')
            print(f"  Warning: Failed to process molecule {i} (ID: {mol_id}): {str(e)}")
            ids.append(mol_id)
            smiles_list.append('')
            descriptions.append('')
    
    # Create DataFrame
    df = pd.DataFrame({
        'ID': ids,
        'SMILES': smiles_list,
        'Description': descriptions
    })
    
    # Save to CSV
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"✓ Successfully saved CSV with {len(df)} molecules")
        print(f"  Output: {output_path}")
        print(f"  Shape: {df.shape}")
        return True
    except Exception as e:
        print(f"Error saving CSV: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Generate CSVs with SMILES and descriptions for all dataset splits')
    parser.add_argument('--data_dir', type=str, default='./data_baseline/data',
                        help='Directory containing .pkl files (default: ./data_baseline/data)')
    parser.add_argument('--output_dir', type=str, default='./',
                        help='Directory for output CSV files (default: ./)')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    # Define splits
    splits = {
        'train': 'train_graphs.pkl',
        'validation': 'validation_graphs.pkl',
        'test': 'test_graphs.pkl'
    }
    
    print(f"Starting SMILES CSV generation for all dataset splits")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Process each split
    results = {}
    for split_name, pkl_filename in splits.items():
        pkl_path = data_dir / pkl_filename
        output_path = output_dir / f'{split_name}_smiles.csv'
        results[split_name] = generate_csv_for_split(pkl_path, output_path, split_name)
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for split_name, success in results.items():
        status = "✓ COMPLETED" if success else "✗ FAILED"
        print(f"  {split_name.upper():15} {status}")
    
    all_success = all(results.values())
    if all_success:
        print(f"\n✓ All CSVs generated successfully in {output_dir}")
    else:
        print(f"\n⚠ Some CSVs failed to generate. Please check the errors above.")


if __name__ == '__main__':
    main()
