"""Data loading utilities for molecular graphs."""

import pickle
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch


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
        
        # Tokenize description
        text = graph.description
        tokenized = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'graph': graph,
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'text': text,
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
