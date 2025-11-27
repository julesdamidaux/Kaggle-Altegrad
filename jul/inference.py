"""Inference script to generate captions for test molecules."""

import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.model import GraphToTextModel
from utils.data_utils import TestMoleculeGraphDataset, collate_fn
import config


def generate_captions(model, dataloader, device):
    """Generate captions for all test molecules."""
    model.eval()
    
    all_ids = []
    all_captions = []
    
    for batch in tqdm(dataloader, desc="Generating captions"):
        batch['graph'] = batch['graph'].to(device)
        
        # Generate
        captions = model.generate(
            batch,
            max_new_tokens=config.GENERATION_MAX_NEW_TOKENS,
            min_new_tokens=config.GENERATION_MIN_NEW_TOKENS,
            num_beams=config.GENERATION_NUM_BEAMS,
            temperature=config.GENERATION_TEMPERATURE,
            top_p=config.GENERATION_TOP_P,
            repetition_penalty=config.GENERATION_REPETITION_PENALTY,
            no_repeat_ngram_size=config.GENERATION_NO_REPEAT_NGRAM_SIZE,
            length_penalty=config.GENERATION_LENGTH_PENALTY,
            use_few_shot=config.USE_FEW_SHOT_PROMPTING
        )

        # print(captions)

        all_ids.extend(batch['ids'])
        all_captions.extend(captions)
    
    return all_ids, all_captions


def main():
    print(f"Device: {config.DEVICE}")
    
    # Load model
    print("\n" + "="*80)
    print("Loading model...")
    print("="*80)
    
    model = GraphToTextModel()
    model = model.to(config.DEVICE)
    
    # Load checkpoint
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using train.py")
        return
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']} with val_loss={checkpoint['val_loss']:.4f}")
    
    # Load test dataset
    print("\n" + "="*80)
    print("Loading test dataset...")
    print("="*80)
    
    test_dataset = TestMoleculeGraphDataset(config.TEST_GRAPHS)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Generate captions
    print("\n" + "="*80)
    print("Generating captions...")
    print("="*80)
    
    ids, captions = generate_captions(model, test_loader, config.DEVICE)
    
    # Create output dataframe
    results_df = pd.DataFrame({
        'ID': ids,
        'description': captions
    })
    
    # Save results
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(config.OUTPUT_DIR, "test_predictions.csv")
    results_df.to_csv(output_path, index=False)
    
    print("\n" + "="*80)
    print(f"Saved {len(results_df)} predictions to: {output_path}")
    print("="*80)
    
    # Print some examples
    print("\nExample predictions:")
    print("="*80)
    for i in range(min(5, len(results_df))):
        print(f"\nID: {results_df.iloc[i]['ID']}")
        print(f"Caption: {results_df.iloc[i]['description']}")
    print("="*80)


if __name__ == "__main__":
    main()
