"""
Main training script for molecular graph captioning.
Loads configuration, initializes model, and runs training loop.
"""

import os
import sys
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import T5Tokenizer, get_linear_schedule_with_warmup

# Add parent directory to path to import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from models import GraphToTextModel, MoleculeGraphTextDataset, collate_graph_text
from train.trainer import train_epoch, evaluate, save_checkpoint

# =========================================================
# CONFIG
# =========================================================
# Data paths
TRAIN_GRAPHS = "/Data/hamza.azzouzi/kaggle_altegrad/data_baseline/data/train_graphs.pkl"
VAL_GRAPHS = "/Data/hamza.azzouzi/kaggle_altegrad/data_baseline/data/validation_graphs.pkl"

# Model parameters
GRAPH_HIDDEN_DIM = 256
GRAPH_LAYERS = 5
T5_MODEL_NAME = "t5-small"  # T5-small LLM

# Training parameters
BATCH_SIZE = 8
EPOCHS = 50
LR = 1e-4
WARMUP_STEPS = 500
MAX_TEXT_LENGTH = 128
GRADIENT_ACCUMULATION_STEPS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# LoRA parameters
USE_LORA = True
LORA_R = 16  # Rank of LoRA matrices
LORA_ALPHA = 32  # LoRA scaling parameter
LORA_DROPOUT = 0.1  # Dropout for LoRA layers
LORA_TARGET_MODULES = ["q", "v"]  # Apply LoRA to query and value projections

# Checkpoint
CHECKPOINT_DIR = "/Data/hamza.azzouzi/kaggle_altegrad/data_baseline/Hamza/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# System prompt for instruction-based fine-tuning
SYSTEM_PROMPT = "Generate a detailed chemical description of the molecule: "


# =========================================================
# Main Training Loop
# =========================================================
def main():
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    
    # Initialize tokenizer
    tokenizer = T5Tokenizer.from_pretrained(T5_MODEL_NAME)
    
    # Load datasets with system prompt
    train_dataset = MoleculeGraphTextDataset(TRAIN_GRAPHS, tokenizer, MAX_TEXT_LENGTH, SYSTEM_PROMPT)
    val_dataset = MoleculeGraphTextDataset(VAL_GRAPHS, tokenizer, MAX_TEXT_LENGTH, SYSTEM_PROMPT)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_graph_text,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_graph_text,
        num_workers=0
    )
    
    # Get node and edge dimensions from first graph
    sample_graph = train_dataset[0]['graph']
    node_dim = sample_graph.x.size(1)
    edge_dim = sample_graph.edge_attr.size(1)
    
    print(f"Node feature dim: {node_dim}")
    print(f"Edge feature dim: {edge_dim}")
    
    # Initialize model
    model = GraphToTextModel(
        node_dim=node_dim,
        edge_dim=edge_dim,
        graph_hidden=GRAPH_HIDDEN_DIM,
        graph_layers=GRAPH_LAYERS,
        t5_model_name=T5_MODEL_NAME,
        use_lora=USE_LORA,
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        lora_target_modules=LORA_TARGET_MODULES
    ).to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    
    total_steps = len(train_loader) * EPOCHS // GRADIENT_ACCUMULATION_STEPS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"{'='*60}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, DEVICE, epoch, GRADIENT_ACCUMULATION_STEPS
        )
        print(f"Train Loss: {train_loss:.4f}")
        
        # Evaluate
        val_loss, samples = evaluate(model, val_loader, DEVICE)
        print(f"Val Loss: {val_loss:.4f}")
        
        # Print some generated samples
        print("\nGenerated Samples:")
        for i, sample in enumerate(samples[:3]):
            print(f"\nSample {i+1}:")
            print(f"Generated:    {sample['generated']}")
            print(f"Ground Truth: {sample['ground_truth'][:100]}...")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, checkpoint_path)
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pt")
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, checkpoint_path)
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
