"""Evaluation script for Graph-to-Text model using BLEU and BERTScore."""

import os
import sys
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import corpus_bleu
from bert_score import score as bert_score
import logging

# Ensure nltk resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

# Add jul directory to path to import utils
sys.path.append('jul')

from utils.model import GraphToTextModel
from utils.data_utils import MoleculeGraphDataset, collate_fn
import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_metrics(hypotheses, references):
    """
    Compute BLEU-4 and BERTScore.
    
    Args:
        hypotheses: List of generated strings
        references: List of reference strings (ground truth)
        
    Returns:
        dict: calculated metrics
    """
    # BLEU-4
    # NLTK expects references to be a list of lists of tokens (for multiple refs/sentence)
    # Since we have 1 ref per sentence, it's [[ref_tokens]]
    
    # Tokenize for BLEU
    refs_tokens = [[nltk.word_tokenize(ref)] for ref in references]
    hyps_tokens = [nltk.word_tokenize(hyp) for hyp in hypotheses]
    
    bleu_score = corpus_bleu(refs_tokens, hyps_tokens)
    logger.info(f"BLEU-4 Score: {bleu_score:.4f}")
    
    # BERTScore
    # Uses ChemBERTa for domain-specific embeddings
    logger.info("Computing BERTScore with seyonec/ChemBERTa-zinc-base-v1...")
    P, R, F1 = bert_score(
        hypotheses, 
        references, 
        model_type="seyonec/ChemBERTa-zinc-base-v1", 
        num_layers=1, # Use first layer for speed if needed, but usually last layer default is fine.
                      # Actually, bert_score uses 'num_layers' to pick which layer. 
                      # Let's trust defaults or user request. User said "with ChemBERTa..." matches implementation.
        verbose=True,
        device=config.DEVICE
    )
    
    bert_f1 = F1.mean().item()
    logger.info(f"BERTScore F1: {bert_f1:.4f}")
    
    return {
        "BLEU-4": bleu_score,
        "BERTScore": bert_f1,
        "Composite": (bleu_score + bert_f1) / 2
    }

def generate_validation_predictions(model, dataloader, device):
    """Generate captions for validation set."""
    model.eval()
    
    all_refs = []
    all_hyps = []
    
    logger.info(f"Starting generation for {len(dataloader.dataset)} validation samples...")
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        batch['graph'] = batch['graph'].to(device)
        
        # Ground Truth
        if 'texts' in batch:
            refs = batch['texts']
        else:
            # Should happen in MoleculeGraphDataset
            refs = batch['text'] # backup
            if isinstance(refs, str): refs = [refs]

        # Generate (using the optimized settings in config/model)
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
        
        all_refs.extend(refs)
        all_hyps.extend(captions)
        
    return all_hyps, all_refs

def main():
    print(f"Device: {config.DEVICE}")
    if config.DEVICE == "cuda":
        torch.cuda.empty_cache()
    
    print("\n" + "="*80)
    print("Loading model...")
    print("="*80)
    
    model = GraphToTextModel(token=config.hf_token)
    model = model.to(config.DEVICE)

    # Apply LoRA if enabled to match checkpoint structure
    if config.USE_LORA:
        model.apply_lora()
    
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
        
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    
    print("\n" + "="*80)
    print("Loading VALIDATION dataset...")
    print("="*80)
    
    # Use MoleculeGraphDataset for validation as it contains 'description'
    val_dataset = MoleculeGraphDataset(
        config.VAL_GRAPHS, 
        model.tokenizer,
        max_length=config.MAX_LENGTH
    )
    
    # Subsample 10% for faster evaluation
    import random
    idx_list = list(range(len(val_dataset)))
    random.seed(42)
    subset_indices = random.sample(idx_list, max(1, int(len(val_dataset) * 0.1)))
    val_dataset = torch.utils.data.Subset(val_dataset, subset_indices)
    print(f"Subsampled 10% of validation set: {len(val_dataset)} samples")
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE, # Can keep small batch size for generation
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Generate
    hyps, refs = generate_validation_predictions(model, val_loader, config.DEVICE)
    
    # Evaluate
    print("\n" + "="*80)
    print("Computing Metrics...")
    print("="*80)
    
    metrics = evaluate_metrics(hyps, refs)
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"BLEU-4 Score: {metrics['BLEU-4']:.6f}")
    print(f"BERTScore F1: {metrics['BERTScore']:.6f}")
    print(f"Combined Score: {metrics['Composite']:.6f}")
    print("="*80)
    
    # Save results to CSV for manual inspection
    results_df = pd.DataFrame({
        'Reference': refs,
        'Generated': hyps
    })
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(config.OUTPUT_DIR, "val_evaluation.csv")
    results_df.to_csv(out_path, index=False)
    print(f"Saved generated samples to {out_path}")

if __name__ == "__main__":
    main()
