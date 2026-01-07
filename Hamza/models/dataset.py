"""
Dataset classes and data loading utilities for molecular graph captioning.
"""

import pickle
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch


class MoleculeGraphTextDataset(Dataset):
    """Dataset that loads molecule graphs with their text descriptions."""
    
    def __init__(self, graph_path, tokenizer, max_length=128, system_prompt="Generate a detailed chemical description of the molecule: "):
        """
        Args:
            graph_path: Path to pickle file containing graph data
            tokenizer: T5 tokenizer for text encoding
            max_length: Maximum sequence length for tokenization
            system_prompt: Instruction prompt prepended to the target text
        """
        print(f"Loading graphs from: {graph_path}")
        with open(graph_path, 'rb') as f:
            self.graphs = pickle.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_prompt = system_prompt
        print(f"Loaded {len(self.graphs)} graphs")
        print(f"System prompt: '{system_prompt}'")
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        graph = self.graphs[idx]
        description = graph.description if hasattr(graph, 'description') else ""
        
        # Add system prompt to the description
        prompted_description = self.system_prompt + description
        
        # Tokenize the prompted description for T5
        tokenized = self.tokenizer(
            prompted_description,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'graph': graph,
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'description': description
        }


def collate_graph_text(batch):
    """
    Collate function for batching graphs and text.
    
    Args:
        batch: List of samples from MoleculeGraphTextDataset
        
    Returns:
        Dictionary with batched graphs, input_ids, attention_mask, and descriptions
    """
    graphs = [item['graph'] for item in batch]
    batched_graph = Batch.from_data_list(graphs)
    
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    descriptions = [item['description'] for item in batch]
    
    return {
        'graph': batched_graph,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'descriptions': descriptions
    }
