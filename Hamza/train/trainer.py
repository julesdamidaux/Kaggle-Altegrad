"""
Training and evaluation functions for molecular graph captioning.
"""

import torch
from tqdm import tqdm


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, gradient_accumulation_steps=4):
    """
    Train the model for one epoch.
    
    Args:
        model: GraphToTextModel instance
        dataloader: Training data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        epoch: Current epoch number
        gradient_accumulation_steps: Number of steps to accumulate gradients
        
    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        batch['graph'] = batch['graph'].to(device)
        labels = batch['input_ids'].to(device)
        
        # Forward pass
        outputs = model(batch['graph'], labels=labels)
        loss = outputs['loss']
        
        # Gradient accumulation
        loss = loss / gradient_accumulation_steps
        loss.backward()
        
        # Update weights
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1
        
        pbar.set_postfix({'loss': total_loss / num_batches})
    
    return total_loss / num_batches


@torch.no_grad()
def evaluate(model, dataloader, device, num_sample_batches=3, samples_per_batch=2):
    """
    Evaluate the model on validation data.
    
    Args:
        model: GraphToTextModel instance
        dataloader: Validation data loader
        device: Device to evaluate on
        num_sample_batches: Number of batches to generate samples from
        samples_per_batch: Number of samples to show per batch
        
    Returns:
        Tuple of (average_loss, list_of_generated_samples)
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    
    generated_samples = []
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        batch['graph'] = batch['graph'].to(device)
        labels = batch['input_ids'].to(device)
        
        # Compute loss
        outputs = model(batch['graph'], labels=labels)
        loss = outputs['loss']
        total_loss += loss.item()
        num_batches += 1
        
        # Generate some samples for inspection
        if batch_idx < num_sample_batches:
            generated_texts = model.generate(batch['graph'])
            for i, (gen_text, true_text) in enumerate(zip(generated_texts, batch['descriptions'])):
                if i < samples_per_batch:
                    generated_samples.append({
                        'generated': gen_text,
                        'ground_truth': true_text
                    })
    
    avg_loss = total_loss / num_batches
    return avg_loss, generated_samples


def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, checkpoint_path):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch
        val_loss: Validation loss
        checkpoint_path: Path to save checkpoint
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss,
    }, checkpoint_path)
    print(f"✓ Saved checkpoint: {checkpoint_path}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    """
    Load model checkpoint.
    
    Args:
        model: Model to load weights into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        checkpoint_path: Path to checkpoint file
        device: Device to load tensors to
        
    Returns:
        Dictionary with epoch and val_loss
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
    return {
        'epoch': checkpoint['epoch'],
        'val_loss': checkpoint['val_loss']
    }
