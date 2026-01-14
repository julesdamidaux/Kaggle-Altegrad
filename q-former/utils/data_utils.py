"""Data loading utilities for molecular graphs."""

import pickle
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
import config

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

def graph_to_smiles(data):
    """
    Converts a PyG graph object back to a SMILES string.
    Assumes:
    - data.x contains atomic numbers at index 0
    - data.edge_index and data.edge_attr contain bond information
    - data.edge_attr contains bond types (1=SINGLE, 2=DOUBLE, 3=TRIPLE, 4=AROMATIC)
    """
    try:
        mol = Chem.RWMol()
        
        # Add atoms
        for feat in data.x:
            atomic_num = int(feat[0].item())
            atom = Chem.Atom(atomic_num)
            mol.AddAtom(atom)
            
        # Add bonds
        rows, cols = data.edge_index
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            bond_types = data.edge_attr
            if bond_types.dim() > 1:
                bond_types = bond_types[:, 0]
                
            for i in range(data.edge_index.size(1)):
                src = int(rows[i].item())
                dst = int(cols[i].item())
                
                if src < dst: # Add only once
                    b_type_val = int(bond_types[i].item())
                    if b_type_val == 1: b_type = Chem.BondType.SINGLE
                    elif b_type_val == 2: b_type = Chem.BondType.DOUBLE
                    elif b_type_val == 3: b_type = Chem.BondType.TRIPLE
                    elif b_type_val == 4: b_type = Chem.BondType.AROMATIC
                    else: b_type = Chem.BondType.SINGLE
                    mol.AddBond(src, dst, b_type)
        
        mol = mol.GetMol()
        
        # Heuristic fix for common valence issues (e.g. N+, O+)
        try:
            mol.UpdatePropertyCache(strict=False)
            for atom in mol.GetAtoms():
                # Nitrogen with 4 bonds -> N+
                if atom.GetAtomicNum() == 7 and atom.GetExplicitValence() == 4 and atom.GetFormalCharge() == 0:
                    atom.SetFormalCharge(1)
                # Oxygen with 3 bonds -> O+
                elif atom.GetAtomicNum() == 8 and atom.GetExplicitValence() == 3 and atom.GetFormalCharge() == 0:
                     atom.SetFormalCharge(1)
        except:
            pass # Continue if property cache fails

        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            # If strict sanitization fails, try to return SMILES anyway
            pass
            
        return Chem.MolToSmiles(mol)
        
    except Exception:
        # Fail gracefully
        return "C"


class MoleculeGraphDataset(Dataset):
    """Dataset for molecular graphs with text descriptions."""
    
    def __init__(self, graph_path, tokenizer, max_length=256):
        """
        Args:
            graph_path: Path to .pkl file with molecular graphs
            tokenizer: HuggingFace tokenizer for text
            max_length: Maximum length for tokenized text
        """
        print(f"Loading graphs from: {graph_path}")
        with open(graph_path, 'rb') as f:
            self.graphs = pickle.load(f)
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Filter graphs that have descriptions
        self.graphs = [g for g in self.graphs if hasattr(g, 'description') and g.description]
        print(f"Loaded {len(self.graphs)} graphs with descriptions")
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        graph = self.graphs[idx]
        
        # 1. Prepare Chat Conversation
        # We insert the <graph> token in the user prompt repeated NUM_QUERY_TOKENS times
        # This reserves space in the input_ids for the graph embeddings
        graph_tokens = config.GRAPH_TOKEN * config.NUM_QUERY_TOKENS
        
        user_prompt = f"Describe the following molecule: {graph_tokens}"
        
        conversation = [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": graph.description}
        ]
        
        # 2. Format with Chat Template
        # We need to calculate where to mask. 
        # Strategy: Tokenize "User Input" separately to know its length.
        
        # Note: apply_chat_template returns a string or list of ints. We want tensors.
        # But to be safe with lengths, we'll do it in steps.
        
        # Format user part only (to get length of instruction)
        user_conversation = [{"role": "user", "content": user_prompt}]
        user_text = self.tokenizer.apply_chat_template(
            user_conversation, 
            add_generation_prompt=True, 
            tokenize=False
        )
        
        # Format full conversation
        full_text = self.tokenizer.apply_chat_template(
            conversation, 
            tokenize=False
        )
        
        # 3. Tokenize
        # We add separate truncation for the full text
        tokenized_full = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors='pt',
            add_special_tokens=False # template adds them
        )
        
        input_ids = tokenized_full['input_ids'].squeeze(0)
        attention_mask = tokenized_full['attention_mask'].squeeze(0)
        
        # 4. Create Labels (Mask User Part)
        labels = input_ids.clone()
        
        # Re-tokenize user part to find length
        # NOTE: This is an approximation. Ideally we'd use the offset mapping, 
        # but since chat templates are standard, matching the prefix length usually works.
        # However, Llama 3 tokenizer might add BOS token automatically if we aren't careful.
        # The safest way is to assume the model learns if we just mask everything before the start of assistant answer.
        
        user_tokens = self.tokenizer(
            user_text, 
            add_special_tokens=False,
            return_tensors='pt'
        )['input_ids'].squeeze(0)
        
        user_len = user_tokens.size(0)
        
        # Safe masking
        if user_len < labels.size(0):
            labels[:user_len] = -100
        else:
            # Should not happen unless truncated completely
            labels[:] = -100
            
        # Pad to max_length
        pad_len = self.max_length - input_ids.size(0)
        if pad_len > 0:
            pad_tokens = torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, pad_tokens])
            
            pad_mask = torch.zeros((pad_len,), dtype=attention_mask.dtype)
            attention_mask = torch.cat([attention_mask, pad_mask])
            
            pad_labels = torch.full((pad_len,), -100, dtype=labels.dtype)
            labels = torch.cat([labels, pad_labels])
            
        return {
            'graph': graph,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'text': graph.description,
            'id': graph.id
        }


class TestMoleculeGraphDataset(Dataset):
    """Dataset for test molecular graphs without descriptions."""
    
    def __init__(self, graph_path):
        print(f"Loading test graphs from: {graph_path}")
        with open(graph_path, 'rb') as f:
            self.graphs = pickle.load(f)
        print(f"Loaded {len(self.graphs)} test graphs")
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        graph = self.graphs[idx]
        return {
            'graph': graph,
            'id': graph.id
        }


def collate_fn(batch):
    """Custom collate function for batching graphs."""
    graphs = [item['graph'] for item in batch]
    batched_graph = Batch.from_data_list(graphs)
    
    result = {
        'graph': batched_graph,
        'ids': [item['id'] for item in batch]
    }
    
    # Add text data if available (training)
    if 'input_ids' in batch[0]:
        result['input_ids'] = torch.stack([item['input_ids'] for item in batch])
        result['attention_mask'] = torch.stack([item['attention_mask'] for item in batch])
        result['texts'] = [item['text'] for item in batch]
    
    return result
