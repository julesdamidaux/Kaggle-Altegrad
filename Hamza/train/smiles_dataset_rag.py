"""
Dataset class for SMILES to description fine-tuning with RAG and graph encodings.
Includes graph encodings and retrieved descriptions as additional context.
"""

import pandas as pd
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Optional
import numpy as np


class SmilesDescriptionRAGDataset(Dataset):
    """Dataset for description generation with RAG and graph encodings."""
    
    def __init__(
        self,
        data_pkl_path: str,
        tokenizer,
        max_length: int = 512,
        prompt_template_fn=None,
        num_retrieved: int = 5,
    ):
        """
        Args:
            data_pkl_path: Path to pickle file with graph encodings and retrieved descriptions
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
            prompt_template_fn: Function to format prompts
            num_retrieved: Number of retrieved descriptions to use
        """
        print(f"Loading RAG dataset from: {data_pkl_path}")
        with open(data_pkl_path, 'rb') as f:
            self.data = pickle.load(f)
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template_fn = prompt_template_fn
        self.num_retrieved = num_retrieved
        
        # Filter out invalid entries
        valid_data = []
        for item in self.data:
            if 'Description' in item:
                if len(item.get('Description', '')) > 0:
                    valid_data.append(item)
        
        self.data = valid_data
        print(f"Loaded {len(self.data)} valid examples with RAG context")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        description = item['Description']
        graph_encoding = np.array(item['graph_encoding'], dtype=np.float32)
        
        # Gather retrieved descriptions
        retrieved_descs = []
        for i in range(1, self.num_retrieved + 1):
            desc = item.get(f'retrieved_desc_{i}', '')
            if desc:
                retrieved_descs.append(desc)
        
        # Format the prompt with retrieved context
        if self.prompt_template_fn:
            text = self.prompt_template_fn(
                retrieved_descriptions=retrieved_descs,
                description=description
            )
        else:
            # Default formatting
            retrieved_text = "\n".join([f"{i+1}. {desc}" for i, desc in enumerate(retrieved_descs)])
            text = f"Similar molecules:\n{retrieved_text}\n\nDescription: {description}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # For causal LM, labels are the same as input_ids
        labels = input_ids.clone()
        
        # Mask padding tokens in labels (set to -100 so they're ignored in loss)
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'graph_encoding': torch.from_numpy(graph_encoding),
        }


class SmilesInferenceRAGDataset(Dataset):
    """Dataset for inference with RAG (no target descriptions)."""
    
    def __init__(
        self,
        data_pkl_path: str,
        tokenizer,
        max_length: int = 512,
        prompt_template_fn=None,
        num_retrieved: int = 5,
    ):
        """
        Args:
            data_pkl_path: Path to pickle file with graph encodings and retrieved descriptions
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
            prompt_template_fn: Function to format prompts
            num_retrieved: Number of retrieved descriptions to use
        """
        print(f"Loading inference dataset from: {data_pkl_path}")
        with open(data_pkl_path, 'rb') as f:
            self.data = pickle.load(f)
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template_fn = prompt_template_fn
        self.num_retrieved = num_retrieved
        
        print(f"Loaded {len(self.data)} examples for inference")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        graph_encoding = np.array(item['graph_encoding'], dtype=np.float32)
        
        # Gather retrieved descriptions
        retrieved_descs = []
        for i in range(1, self.num_retrieved + 1):
            desc = item.get(f'retrieved_desc_{i}', '')
            if desc:
                retrieved_descs.append(desc)
        
        # Format the prompt with retrieved context (no target description)
        if self.prompt_template_fn:
            text = self.prompt_template_fn(
                retrieved_descriptions=retrieved_descs
            )
        else:
            # Default formatting
            retrieved_text = "\n".join([f"{i+1}. {desc}" for i, desc in enumerate(retrieved_descs)])
            text = f"Similar molecules:\n{retrieved_text}\n\nDescription:"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'graph_encoding': torch.from_numpy(graph_encoding),
            'id': item['ID'],
        }
