# SMILES Pipeline for PyG Graphs

This module provides functionality to convert PyTorch Geometric Data objects to SMILES strings using RDKit.

## Files Created

### 1. `Hamza/chem/pyg_to_smiles.py`
Main conversion module containing:
- `pyg_data_to_smiles(data, x_map, e_map, canonical=True)`: Converts a PyG Data object to SMILES string

**Key Features:**
- Handles bidirectional edges (only processes u < v to avoid duplicates)
- Decodes categorical features using x_map and e_map from data_utils.py
- Sets atom properties: atomic_num, formal_charge, num_hs, is_aromatic, hybridization, chirality
- Sets bond properties: bond_type, stereo, is_conjugated
- Performs molecule sanitization and stereochemistry assignment
- Returns canonical SMILES by default
- Raises clear exceptions with molecule ID on failure

### 2. `scripts/test_smiles.py`
CLI testing script for validating SMILES conversion.

**Usage:**
```bash
python scripts/test_smiles.py --pkl train_graphs.pkl --n 10
```

**Arguments:**
- `--pkl`: Path to .pkl file containing preprocessed graphs (required)
- `--n`: Number of molecules to test (default: 10)

**Output:**
For each molecule, prints:
- ID
- Number of nodes
- Number of unique bonds (undirected edges)
- Generated SMILES string
- Validation status (whether RDKit can parse the SMILES)

**Summary statistics:**
- Total tested
- Successful conversions
- Failed conversions
- Valid SMILES count
- Success and validity rates

## Dependencies

Required packages:
```bash
pip install torch torch-geometric rdkit
```

## Example Usage

### From Code:

```python
from data_baseline.data_utils import PreprocessedGraphDataset, x_map, e_map
from Hamza.chem.pyg_to_smiles import pyg_data_to_smiles

# Load dataset
dataset = PreprocessedGraphDataset('train_graphs.pkl')

# Convert first molecule
data = dataset[0]
smiles = pyg_data_to_smiles(data, x_map, e_map, canonical=True)
print(f"SMILES: {smiles}")

# Validate with RDKit
from rdkit import Chem
mol = Chem.MolFromSmiles(smiles)
if mol is not None:
    print("Valid SMILES!")
```

### From Command Line:

```bash
# Test first 10 molecules
python scripts/test_smiles.py --pkl train_graphs.pkl --n 10

# Test first 100 molecules
python scripts/test_smiles.py --pkl train_graphs.pkl --n 100

# Test all molecules in dataset
python scripts/test_smiles.py --pkl train_graphs.pkl --n 999999
```

## Implementation Notes

1. **Edge Handling**: The PyG Data object stores edges bidirectionally (both u→v and v→u). The conversion function only processes edges where `u < v` to avoid adding duplicate bonds.

2. **Feature Decoding**: Node and edge features are stored as categorical indices. The x_map and e_map dictionaries from data_utils.py are used to decode these indices back to their original values.

3. **Sanitization**: RDKit's SanitizeMol is called to ensure chemical validity. If this fails, a clear exception is raised with the molecule ID.

4. **Stereochemistry**: AssignStereochemistry is called to properly handle chiral centers and stereochemical bonds.

5. **Error Handling**: All errors include the molecule ID (data.id) to help with debugging specific molecules.

## Directory Structure

```
Hamza/
├── chem/
│   ├── __init__.py
│   └── pyg_to_smiles.py
├── models/
└── train/

scripts/
└── test_smiles.py
```

## Acceptance Criteria

✓ Created in `Hamza/` subdirectory (graph encoder not touched)  
✓ Handles bidirectional edges correctly (u < v filter)  
✓ Decodes features using x_map and e_map  
✓ Sets required atom properties  
✓ Sets required bond properties  
✓ Calls SanitizeMol and AssignStereochemistry  
✓ Raises clear exceptions with molecule ID on failure  
✓ Test script with CLI interface  
✓ Validates SMILES with RDKit  
✓ Prints detailed per-molecule statistics  
✓ Handles errors gracefully and continues  
✓ No checkpoints or caches involved
