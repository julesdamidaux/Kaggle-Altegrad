"""
Dataset class for SMILES to description fine-tuning.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Optional


class SmilesDescriptionDataset(Dataset):
    """Dataset for SMILES to description generation."""
    
    def __init__(
        self,
        csv_path: str,
        tokenizer,
        max_length: int = 512,
        prompt_template_fn=None,
    ):
        """
        Args:
            csv_path: Path to CSV file with columns: ID, SMILES, Description
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
            prompt_template_fn: Function to format prompts
        """
        print(f"Loading dataset from: {csv_path}")
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template_fn = prompt_template_fn
        
        # Remove any rows with missing SMILES or Description
        self.data = self.data.dropna(subset=['SMILES', 'Description'])
        self.data = self.data[self.data['SMILES'].str.len() > 0]
        self.data = self.data[self.data['Description'].str.len() > 0]
        
        print(f"Loaded {len(self.data)} valid examples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        smiles = row['SMILES']
        description = row['Description']
        
        # Format the prompt
        if self.prompt_template_fn:
            text = self.prompt_template_fn(smiles, description)
        else:
            text = f"SMILES: {smiles}\n\nDescription: {description}"
        
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
        }


class SmilesInferenceDataset(Dataset):
    """Dataset for inference (no descriptions, only SMILES)."""
    
    def __init__(
        self,
        csv_path: str,
        tokenizer,
        max_length: int = 512,
        prompt_template_fn=None,
    ):
        """
        Args:
            csv_path: Path to CSV file with columns: ID, SMILES
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
            prompt_template_fn: Function to format prompts
        """
        print(f"Loading dataset from: {csv_path}")
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template_fn = prompt_template_fn
        
        # Remove any rows with missing SMILES
        self.data = self.data.dropna(subset=['SMILES'])
        self.data = self.data[self.data['SMILES'].str.len() > 0]
        
        print(f"Loaded {len(self.data)} examples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        smiles = row['SMILES']
        mol_id = row['ID']
        
        # Format the prompt (without description for inference)
        if self.prompt_template_fn:
            text = self.prompt_template_fn(smiles, None)
        else:
            text = f"SMILES: {smiles}\n\nDescription:"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'id': mol_id,
            'smiles': smiles,
        }
