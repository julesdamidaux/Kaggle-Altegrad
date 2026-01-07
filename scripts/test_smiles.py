#!/usr/bin/env python3
"""
Test script for SMILES generation from PyTorch Geometric Data objects.

Usage:
    python scripts/test_smiles.py --pkl train_graphs.pkl --n 10
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_baseline.data_utils import PreprocessedGraphDataset, x_map, e_map
from Hamza.chem.pyg_to_smiles import pyg_data_to_smiles

try:
    from rdkit import Chem
except ImportError:
    print("Error: RDKit is not installed. Please install it with:")
    print("  pip install rdkit")
    sys.exit(1)


def validate_smiles(smiles: str) -> bool:
    """
    Validate a SMILES string using RDKit.
    
    Args:
        smiles: SMILES string to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception:
        return False


def count_unique_bonds(data):
    """
    Count the number of unique bonds in a PyG Data object.
    
    Args:
        data: PyG Data object with edge_index
        
    Returns:
        Number of unique bonds (undirected edges)
    """
    edge_index = data.edge_index
    num_edges = edge_index.size(1)
    
    unique_bonds = set()
    for i in range(num_edges):
        u = int(edge_index[0, i].item())
        v = int(edge_index[1, i].item())
        # Store as sorted tuple to ensure uniqueness
        bond = tuple(sorted([u, v]))
        unique_bonds.add(bond)
    
    return len(unique_bonds)


def main():
    parser = argparse.ArgumentParser(description='Test SMILES generation from PyG graphs')
    parser.add_argument('--pkl', type=str, required=True,
                        help='Path to .pkl file containing preprocessed graphs')
    parser.add_argument('--n', type=int, default=10,
                        help='Number of molecules to test (default: 10)')
    
    args = parser.parse_args()
    
    # Check if file exists
    pkl_path = Path(args.pkl)
    if not pkl_path.exists():
        print(f"Error: File not found: {pkl_path}")
        sys.exit(1)
    
    print(f"Loading dataset from: {pkl_path}")
    print(f"Testing first {args.n} molecules")
    print("=" * 80)
    
    # Load dataset
    try:
        dataset = PreprocessedGraphDataset(str(pkl_path))
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    
    # Limit n to dataset size
    n = min(args.n, len(dataset))
    
    # Statistics
    successful = 0
    failed = 0
    valid = 0
    invalid = 0
    
    # Test each molecule
    for i in range(n):
        try:
            # Get graph data
            data = dataset[i]
            
            # Get molecule info
            mol_id = getattr(data, 'id', f'unknown_{i}')
            num_nodes = data.x.size(0)
            unique_bonds = count_unique_bonds(data)
            description = getattr(data, 'description', 'N/A')
            
            # Convert to SMILES
            smiles = pyg_data_to_smiles(data, x_map, e_map, canonical=True)
            
            # Validate SMILES
            is_valid = validate_smiles(smiles)
            
            # Update statistics
            successful += 1
            if is_valid:
                valid += 1
            else:
                invalid += 1
            
            # Print result
            print(f"\nMolecule {i}:")
            print(f"  ID:           {mol_id}")
            print(f"  Nodes:        {num_nodes}")
            print(f"  Unique Bonds: {unique_bonds}")
            print(f"  SMILES:       {smiles}")
            print(f"  Valid:        {is_valid}")
            print(f"  Description:  {description}")
            
        except Exception as e:
            # Handle error
            failed += 1
            mol_id = getattr(dataset[i], 'id', f'unknown_{i}')
            print(f"\nMolecule {i}:")
            print(f"  ID:    {mol_id}")
            print(f"  ERROR: {str(e)}")
    
    # Print summary
    print("\n" + "=" * 80)
    print(f"Summary:")
    print(f"  Total tested:     {n}")
    print(f"  Successful:       {successful}")
    print(f"  Failed:           {failed}")
    print(f"  Valid SMILES:     {valid}")
    print(f"  Invalid SMILES:   {invalid}")
    
    if successful > 0:
        success_rate = (successful / n) * 100
        validity_rate = (valid / successful) * 100 if successful > 0 else 0
        print(f"  Success rate:     {success_rate:.1f}%")
        print(f"  Validity rate:    {validity_rate:.1f}%")


if __name__ == '__main__':
    main()
