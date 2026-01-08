"""
Generate predictions using fine-tuned Qwen2.5-3B-Instruct model.

Usage:
    python predict_qwen.py --checkpoint final --input test_smiles.csv --output predictions.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Add Hamza directory to path
sys.path.insert(0, str(Path(__file__).parent))

from Hamza.train.smiles_dataset import SmilesInferenceDataset
from Hamza.train.finetune_config import get_prompt_template, GENERATE_MAX_LENGTH, CACHE_DIR


def load_model(checkpoint_name, cache_dir="./cache", device="cuda"):
    """Load the fine-tuned Qwen model."""
    
    checkpoint_path = f"./Hamza/checkpoints/qwen2.5-3b-smiles/{checkpoint_name}"
    
    print(f"Loading model from: {checkpoint_path}")
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        cache_dir=cache_dir,
    )
    
    model.eval()
    
    return model, tokenizer


def generate_descriptions(model, tokenizer, dataset, batch_size=8, device="cuda"):
    """Generate descriptions for all molecules in the dataset."""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    results = []
    
    print(f"\nGenerating descriptions for {len(dataset)} molecules...")
    print(f"Batch size: {batch_size}")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Generate with greedy decoding for consistency
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=GENERATE_MAX_LENGTH,
                do_sample=False,  # Greedy decoding
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            # Decode outputs
            for i, output in enumerate(outputs):
                # Remove the input prompt from the output
                input_length = input_ids[i].shape[0]

                generated_ids = output[input_length:]
                description = tokenizer.decode(generated_ids, skip_special_tokens=True)
                if i ==0:
                    print("Bellow the first ever desciption")
                    print(description)
                    print('Now the full output decoded:')
                    print(tokenizer.decode(output, skip_special_tokens=True))
                results.append({
                    'ID': batch['id'][i],
                    'SMILES': batch['smiles'][i],
                    'Description': description.strip(),
                })
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Generate predictions using Qwen2.5-3B')
    parser.add_argument('--checkpoint', type=str, default='final',
                        help='Checkpoint name (e.g., "final", "checkpoint-5814")')
    parser.add_argument('--input', type=str, default='test_smiles.csv',
                        help='Input CSV file with SMILES')
    parser.add_argument('--output', type=str, default='test_predictions_qwen.csv',
                        help='Output CSV file for predictions')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--cache_dir', type=str, default='./cache',
                        help='Cache directory for models')
    
    args = parser.parse_args()
    
    print("="*80)
    print("Qwen2.5-3B-Instruct Prediction Pipeline")
    print("="*80)
    
    # Load model
    print("\n[1/4] Loading model...")
    model, tokenizer = load_model(
        checkpoint_name=args.checkpoint,
        cache_dir=args.cache_dir,
        device=args.device
    )
    
    # Load dataset
    print(f"\n[2/4] Loading dataset from {args.input}...")
    dataset = SmilesInferenceDataset(
        csv_path=args.input,
        tokenizer=tokenizer,
        prompt_template_fn=get_prompt_template,
    )
    print(f"Loaded {len(dataset)} molecules")
    
    # Generate descriptions
    print(f"\n[3/4] Generating descriptions...")
    results = generate_descriptions(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        batch_size=args.batch_size,
        device=args.device,
    )
    
    # Save results
    print(f"\n[4/4] Saving predictions...")
    df = pd.DataFrame(results)
    
    # Keep only ID and Description columns for submission
    submission_df = df[['ID', 'Description']].copy()
    submission_df.columns = ['ID', 'description']  # Match required format
    
    submission_df.to_csv(args.output, index=False)
    
    print("\n" + "="*80)
    print(f"✓ Saved {len(submission_df)} predictions to: {args.output}")
    print("="*80)
    
    # Show example predictions
    print("\nExample predictions:")
    print("-"*80)
    for i in range(min(3, len(submission_df))):
        print(f"\nID: {submission_df.iloc[i]['ID']}")
        print(f"Description: {submission_df.iloc[i]['description'][:200]}...")
    print("-"*80)


if __name__ == "__main__":
    main()
