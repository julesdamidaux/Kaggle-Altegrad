"""
Inference script for Qwen model with RAG and graph encodings.

Usage:
    python Hamza/train/inference_qwen_with_rag.py \
        --checkpoint ./Hamza/checkpoints/qwen2.5-rag/final \
        --data ./Hamza/RAG+Predict/embeddings/test_llm_data.pkl \
        --output predictions_rag.csv
"""

import argparse
import sys
from pathlib import Path
import pickle

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Hamza.train.smiles_dataset_rag import SmilesInferenceRAGDataset
from Hamza.train.qwen_with_graph import QwenWithGraphEncoding
from Hamza.train.finetune_config import GENERATE_MAX_LENGTH, SYSTEM_PROMPT


def get_prompt_template_with_rag(retrieved_descriptions: list = None, description: str = None) -> str:
    """
    Format the input/output with RAG context for inference.
    """
    # Format retrieved descriptions
    if retrieved_descriptions:
        retrieved_text = "Here are descriptions of structurally similar molecules:\n"
        for i, desc in enumerate(retrieved_descriptions[:5], 1):
            retrieved_text += f"{i}. {desc}\n"
        retrieved_text += "\n"
    else:
        retrieved_text = ""
    
    user = f"{retrieved_text}Write the ontology-style description for this molecule."
    
    if description is not None:
        return f"""<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
Description: {description}<|im_end|>"""
    else:
        return f"""<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
Description:"""


def load_model(checkpoint_path, device="cuda", graph_encoding_dim=256):
    """Load the fine-tuned model with graph encoding support."""
    print(f"Loading model from: {checkpoint_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model with graph encoding wrapper
    model = QwenWithGraphEncoding.from_pretrained(
        checkpoint_path,
        graph_encoding_dim=graph_encoding_dim,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    
    model.eval()
    
    return model, tokenizer


def custom_collate(batch):
    """Custom collate function for batching with graph encodings."""
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'graph_encoding': torch.stack([item['graph_encoding'] for item in batch]),
        'id': [item['id'] for item in batch],
    }


def generate_descriptions(model, tokenizer, dataset, batch_size=8, device="cuda", max_new_tokens=256):
    """Generate descriptions for all molecules in the dataset."""
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=custom_collate
    )
    
    results = []
    
    print(f"Generating descriptions for {len(dataset)} molecules...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            graph_encodings = batch['graph_encoding'].to(device)
            
            # Generate with graph encodings
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                graph_encodings=graph_encodings,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            # Decode
            # Note: outputs will be longer due to prepended graph tokens
            # We need to account for: num_graph_tokens + num_separator_tokens + original input
            num_prefix_tokens = model.num_graph_tokens + model.num_separator_tokens
            
            for i, output in enumerate(outputs):
                # Remove the prefix (graph + separator) and input prompt
                input_length = input_ids[i].shape[0] + num_prefix_tokens
                generated_ids = output[input_length:]
                
                description = tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                results.append({
                    'ID': batch['id'][i],
                    'Description': description.strip(),
                })
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Generate descriptions using Qwen with RAG and graph encodings'
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to fine-tuned model checkpoint')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to input pickle file with graph encodings and RAG context')
    parser.add_argument('--output', type=str, default='predictions_rag.csv',
                        help='Path to output CSV')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--max_new_tokens', type=int, default=256,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--graph_encoding_dim', type=int, default=256,
                        help='Dimension of graph encodings')
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(
        args.checkpoint, 
        device=args.device,
        graph_encoding_dim=args.graph_encoding_dim
    )
    
    # Load dataset
    dataset = SmilesInferenceRAGDataset(
        data_pkl_path=args.data,
        tokenizer=tokenizer,
        max_length=512,
        prompt_template_fn=get_prompt_template_with_rag,
        num_retrieved=5,
    )
    
    # Generate
    results = generate_descriptions(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        batch_size=args.batch_size,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"\nPredictions saved to: {args.output}")
    print(f"Generated {len(results)} descriptions")


if __name__ == "__main__":
    main()
