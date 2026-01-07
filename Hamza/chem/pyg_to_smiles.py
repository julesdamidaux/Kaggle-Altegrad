"""
Convert PyTorch Geometric Data objects to SMILES strings using RDKit.
"""

from typing import Dict, List, Any
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import rdchem


def pyg_data_to_smiles(
    data: Data,
    x_map: Dict[str, List[Any]],
    e_map: Dict[str, List[Any]],
    canonical: bool = True
) -> str:
    """
    Convert a PyTorch Geometric Data object to a SMILES string.
    
    Args:
        data: PyG Data object with x, edge_index, edge_attr attributes
        x_map: Dictionary mapping atom feature names to their possible values
        e_map: Dictionary mapping edge feature names to their possible values
        canonical: If True, return canonical SMILES (default: True)
        
    Returns:
        SMILES string representation of the molecule
        
    Raises:
        ValueError: If conversion fails or molecule is invalid
        
    Note:
        - edge_index contains bidirectional edges, so we only add each bond once (u < v)
        - Node features are decoded using x_map categorical indices
        - Edge features are decoded using e_map categorical indices
    """
    try:
        # Create RDKit molecule object (editable)
        mol = Chem.RWMol()
        
        # Extract node features
        x = data.x  # Shape: [num_nodes, num_features]
        num_nodes = x.size(0)
        
        # Feature indices (matching the order in x_map)
        # Based on data_utils.py x_map structure:
        # atomic_num, chirality, degree, formal_charge, num_hs, num_radical_electrons, hybridization, is_aromatic, is_in_ring
        ATOMIC_NUM_IDX = 0
        CHIRALITY_IDX = 1
        FORMAL_CHARGE_IDX = 3
        NUM_HS_IDX = 4
        HYBRIDIZATION_IDX = 6
        IS_AROMATIC_IDX = 7
        
        # Add atoms to molecule
        atom_idx_map = {}  # Map from data node idx to RDKit atom idx
        for i in range(num_nodes):
            atom_features = x[i]
            
            # Decode atomic number
            atomic_num_idx = int(atom_features[ATOMIC_NUM_IDX].item())
            atomic_num = x_map['atomic_num'][atomic_num_idx]
            
            # Create atom
            atom = Chem.Atom(atomic_num)
            
            # Set formal charge
            formal_charge_idx = int(atom_features[FORMAL_CHARGE_IDX].item())
            formal_charge = x_map['formal_charge'][formal_charge_idx]
            atom.SetFormalCharge(formal_charge)
            
            # Set number of explicit hydrogens
            num_hs_idx = int(atom_features[NUM_HS_IDX].item())
            num_hs = x_map['num_hs'][num_hs_idx]
            atom.SetNumExplicitHs(num_hs)
            
            # Set aromaticity
            is_aromatic_idx = int(atom_features[IS_AROMATIC_IDX].item())
            is_aromatic = x_map['is_aromatic'][is_aromatic_idx]
            atom.SetIsAromatic(is_aromatic)
            
            # Set hybridization
            hybridization_idx = int(atom_features[HYBRIDIZATION_IDX].item())
            hybridization_str = x_map['hybridization'][hybridization_idx]
            if hybridization_str != 'UNSPECIFIED':
                hybridization = getattr(rdchem.HybridizationType, hybridization_str)
                atom.SetHybridization(hybridization)
            
            # Set chirality
            chirality_idx = int(atom_features[CHIRALITY_IDX].item())
            chirality_str = x_map['chirality'][chirality_idx]
            if chirality_str != 'CHI_UNSPECIFIED':
                chirality = getattr(rdchem.ChiralType, chirality_str)
                atom.SetChiralTag(chirality)
            
            # Add atom to molecule
            rdkit_idx = mol.AddAtom(atom)
            atom_idx_map[i] = rdkit_idx
        
        # Extract edge features
        edge_index = data.edge_index  # Shape: [2, num_edges]
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        
        # Feature indices for edges (matching e_map structure):
        # bond_type, stereo, is_conjugated
        BOND_TYPE_IDX = 0
        STEREO_IDX = 1
        IS_CONJUGATED_IDX = 2
        
        # Add bonds (only once per undirected edge, using u < v)
        added_bonds = set()
        num_edges = edge_index.size(1)
        
        for i in range(num_edges):
            u = int(edge_index[0, i].item())
            v = int(edge_index[1, i].item())
            
            # Skip if already added (bidirectional edges)
            if u >= v:
                continue
            
            bond_key = (u, v)
            if bond_key in added_bonds:
                continue
            added_bonds.add(bond_key)
            
            # Get RDKit atom indices
            u_rdkit = atom_idx_map[u]
            v_rdkit = atom_idx_map[v]
            
            # Decode bond type
            if edge_attr is not None:
                bond_features = edge_attr[i]
                bond_type_idx = int(bond_features[BOND_TYPE_IDX].item())
                bond_type_str = e_map['bond_type'][bond_type_idx]
                
                # Map bond type string to RDKit BondType
                bond_type_mapping = {
                    'SINGLE': rdchem.BondType.SINGLE,
                    'DOUBLE': rdchem.BondType.DOUBLE,
                    'TRIPLE': rdchem.BondType.TRIPLE,
                    'AROMATIC': rdchem.BondType.AROMATIC,
                    'UNSPECIFIED': rdchem.BondType.UNSPECIFIED,
                }
                bond_type = bond_type_mapping.get(bond_type_str, rdchem.BondType.SINGLE)
            else:
                bond_type = rdchem.BondType.SINGLE
            
            # Add bond
            bond_idx = mol.AddBond(u_rdkit, v_rdkit, bond_type) - 1
            bond = mol.GetBondWithIdx(bond_idx)
            
            # Set additional bond properties
            if edge_attr is not None:
                # Set stereo
                stereo_idx = int(bond_features[STEREO_IDX].item())
                stereo_str = e_map['stereo'][stereo_idx]
                if stereo_str != 'STEREONONE':
                    stereo_mapping = {
                        'STEREOANY': rdchem.BondStereo.STEREOANY,
                        'STEREOZ': rdchem.BondStereo.STEREOZ,
                        'STEREOE': rdchem.BondStereo.STEREOE,
                        'STEREOCIS': rdchem.BondStereo.STEREOCIS,
                        'STEREOTRANS': rdchem.BondStereo.STEREOTRANS,
                    }
                    if stereo_str in stereo_mapping:
                        bond.SetStereo(stereo_mapping[stereo_str])
                
                # Set conjugation
                is_conjugated_idx = int(bond_features[IS_CONJUGATED_IDX].item())
                is_conjugated = e_map['is_conjugated'][is_conjugated_idx]
                bond.SetIsConjugated(is_conjugated)
        
        # Convert to regular Mol object
        mol = mol.GetMol()
        
        # Sanitize molecule
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            raise ValueError(f"Failed to sanitize molecule (ID: {getattr(data, 'id', 'unknown')}): {str(e)}")
        
        # Assign stereochemistry
        try:
            Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        except Exception as e:
            # Non-critical, continue without stereochemistry
            pass
        
        # Generate SMILES
        try:
            smiles = Chem.MolToSmiles(mol, canonical=canonical)
            if not smiles:
                raise ValueError(f"Generated empty SMILES for molecule (ID: {getattr(data, 'id', 'unknown')})")
            return smiles
        except Exception as e:
            raise ValueError(f"Failed to generate SMILES (ID: {getattr(data, 'id', 'unknown')}): {str(e)}")
            
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Error converting PyG data to SMILES (ID: {getattr(data, 'id', 'unknown')}): {str(e)}")
