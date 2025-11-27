"""Training script for Graph-to-Text model."""

import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

from utils.model import GraphToTextModel
from utils.data_utils import MoleculeGraphDataset, collate_fn
import config


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        batch['graph'] = batch['graph'].to(device)
        batch['input_ids'] = batch['input_ids'].to(device)
        batch['attention_mask'] = batch['attention_mask'].to(device)
        
        # Labels are input_ids shifted (handled in model)
        labels = batch['input_ids'].clone()
        
        # Forward pass
        loss = model(batch, labels=labels)
        
        # Backward pass with gradient accumulation
        loss = loss / config.GRADIENT_ACCUMULATION_STEPS
        loss.backward()
        
        if (batch_idx + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            
            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * config.GRADIENT_ACCUMULATION_STEPS
        num_batches += 1
        
        pbar.set_postfix({'loss': total_loss / num_batches})
    
    return total_loss / num_batches


@torch.no_grad()
def validate(model, dataloader, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Validating"):
        batch['graph'] = batch['graph'].to(device)
        batch['input_ids'] = batch['input_ids'].to(device)
        batch['attention_mask'] = batch['attention_mask'].to(device)
        
        labels = batch['input_ids'].clone()
        
        loss = model(batch, labels=labels)
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def main():
    # Create directories
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    
    print(f"Device: {config.DEVICE}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Gradient accumulation steps: {config.GRADIENT_ACCUMULATION_STEPS}")
    print(f"Effective batch size: {config.BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS}")
    
    # Initialize model
    print("\n" + "="*80)
    print("Initializing model...")
    print("="*80)
    model = GraphToTextModel(freeze_llm=False)
    model = model.to(config.DEVICE)
    
    # Load datasets
    print("\n" + "="*80)
    print("Loading datasets...")
    print("="*80)
    train_dataset = MoleculeGraphDataset(
        config.TRAIN_GRAPHS,
        model.tokenizer,
        max_length=config.MAX_LENGTH
    )
    
    val_dataset = MoleculeGraphDataset(
        config.VAL_GRAPHS,
        model.tokenizer,
        max_length=config.MAX_LENGTH
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.MAX_EPOCHS,
        eta_min=1e-6
    )
    
    # Training loop
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80)
    
    best_val_loss = float('inf')
    training_history = []
    
    for epoch in range(1, config.MAX_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.MAX_EPOCHS}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, config.DEVICE, epoch)
        
        # Validate
        val_loss = validate(model, val_loader, config.DEVICE)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save history
        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss
        })
        
        # # Save checkpoint
        # checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pt")
        # torch.save({
        #     'epoch': epoch,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'train_loss': train_loss,
        #     'val_loss': val_loss,
        # }, checkpoint_path)
        # print(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
            }, best_path)
            print(f"New best model saved! Val Loss: {val_loss:.4f}")
        
        # Step scheduler
        scheduler.step()
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
    
    # Save training history
    history_path = os.path.join(config.LOG_DIR, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print("\n" + "="*80)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("="*80)


if __name__ == "__main__":
    main()
