"""
Inference script for fine-tuned Qwen2.5-3B-Instruct model.

Usage:
    python Hamza/train/inference_qwen.py --checkpoint ./Hamza/checkpoints/qwen2.5-3b-smiles/final --csv test_smiles.csv --output predictions.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Hamza.train.smiles_dataset import SmilesInferenceDataset
from Hamza.train.finetune_config import get_prompt_template, GENERATE_MAX_LENGTH


def load_model(checkpoint_path, device="cuda"):
    """Load the fine-tuned model."""
    print(f"Loading model from: {checkpoint_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    
    model.eval()
    
    return model, tokenizer


def generate_descriptions(model, tokenizer, dataset, batch_size=8, device="cuda"):
    """Generate descriptions for all molecules in the dataset."""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    results = []
    
    print(f"Generating descriptions for {len(dataset)} molecules...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Generate
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=GENERATE_MAX_LENGTH,
                do_sample=False,  # Greedy decoding
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            # Decode
            for i, output in enumerate(outputs):
                # Remove the input prompt from the output
                input_length = input_ids[i].shape[0]
                generated_ids = output[input_length:]
                
                description = tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                results.append({
                    'ID': batch['id'][i],
                    'SMILES': batch['smiles'][i],
                    'Description': description.strip(),
                })
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Generate descriptions from SMILES using fine-tuned Qwen2.5')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to fine-tuned model checkpoint')
    parser.add_argument('--csv', type=str, required=True,
                        help='Path to input CSV with SMILES')
    parser.add_argument('--output', type=str, default='predictions.csv',
                        help='Path to output CSV')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(args.checkpoint, args.device)
    
    # Load dataset
    dataset = SmilesInferenceDataset(
        csv_path=args.csv,
        tokenizer=tokenizer,
        prompt_template_fn=get_prompt_template,
    )
    
    # Generate descriptions
    results = generate_descriptions(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        batch_size=args.batch_size,
        device=args.device,
    )
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    
    print(f"\n✓ Saved predictions to: {args.output}")
    print(f"  Total molecules: {len(df)}")


if __name__ == "__main__":
    main()
