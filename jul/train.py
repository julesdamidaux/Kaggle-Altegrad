"""Training script for Graph-to-Text model."""

import os
import torch
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup
from tqdm import tqdm
import json
import wandb

from utils.model import GraphToTextModel
from utils.data_utils import MoleculeGraphDataset, collate_fn
import config



def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    # Zero gradients at the start of the epoch
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        batch['graph'] = batch['graph'].to(device)
        batch['input_ids'] = batch['input_ids'].to(device)
        batch['attention_mask'] = batch['attention_mask'].to(device)
        
        labels = batch['input_ids'].clone()
        loss = model(batch, labels=labels)
        
        loss = loss / config.GRADIENT_ACCUMULATION_STEPS
        loss.backward()
        
        # Perform optimization step if accumulation is done OR it's the last batch
        is_last_batch = (batch_idx + 1) == len(dataloader)
        if (batch_idx + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0 or is_last_batch:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Log to wandb every optimizer step
            lrs = scheduler.get_last_lr()
            lr_head = lrs[0]
            lr_llm = lrs[1] if len(lrs) > 1 else 0.0
            
            current_avg_loss = total_loss / (num_batches + 1e-6) # Avoid div by zero
            wandb.log({
                'train/loss_step': loss.item() * config.GRADIENT_ACCUMULATION_STEPS,
                'train/loss_avg': current_avg_loss,
                'train/lr_head': lr_head,
                'train/lr_llm': lr_llm,
                'train/step': epoch * len(dataloader) // config.GRADIENT_ACCUMULATION_STEPS + (batch_idx + 1) // config.GRADIENT_ACCUMULATION_STEPS
            })
        
        total_loss += loss.item() * config.GRADIENT_ACCUMULATION_STEPS
        num_batches += 1
        
        lrs = scheduler.get_last_lr()
        lr_head = lrs[0]
        lr_llm = lrs[1] if len(lrs) > 1 else 0.0
        
        pbar.set_postfix({
            'loss': total_loss / num_batches, 
            'lr_head': lr_head,
            'lr_llm': lr_llm
        })

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


def run_training_phase(model, train_loader, val_loader, optimizer, scheduler, num_epochs, start_epoch, stage_name):
    """Run a training phase (Stage 1 or Stage 2)."""
    print(f"\n" + "="*80)
    print(f"Starting {stage_name}")
    print("="*80)
    
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, start_epoch + num_epochs):
        print(f"\nEpoch {epoch}/{start_epoch + num_epochs - 1} ({stage_name})")
        
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, config.DEVICE, epoch)
        val_loss = validate(model, val_loader, config.DEVICE)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Log to wandb
        wandb.log({
            'epoch': epoch,
            'train/loss_epoch': train_loss,
            'val/loss_epoch': val_loss,
            'stage': 1 if "Stage 1" in stage_name else 2
        })
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save best model for this stage
            # stage_suffix = "stage1" if "Stage 1" in stage_name else "stage2"
            # best_path = os.path.join(config.CHECKPOINT_DIR, f"best_model_{stage_suffix}.pt")
            
            # Also update the main "best_model.pt" if it's the best overall (or just overwrite for now)
            # For simplicity, let's keep a "best_model.pt" that is always the latest best
            main_best_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
            
            state_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'stage': stage_name
            }
            
            # torch.save(state_dict, best_path)
            torch.save(state_dict, main_best_path)
            print(f"New best model saved! Val Loss: {val_loss:.4f}")
        
        lrs = scheduler.get_last_lr()
        lr_head = lrs[0]
        lr_llm = lrs[1] if len(lrs) > 1 else 0.0
        print(f"Learning rates - Head: {lr_head:.2e}, LLM: {lr_llm:.2e}")
        
    return best_val_loss


def main():
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    
    # Initialize wandb
    wandb.init(
        project="graph-to-text-altegrad",
        name=f"qwen-{config.LLM_MODEL_NAME.split('/')[-1]}-auto-2stage",
        config={
            "model": config.LLM_MODEL_NAME,
            "epochs_stage1": config.EPOCHS_STAGE1,
            "epochs_stage2": config.EPOCHS_STAGE2,
            "batch_size": config.BATCH_SIZE,
            "gradient_accumulation": config.GRADIENT_ACCUMULATION_STEPS,
            "learning_rate_stage1_head": config.LEARNING_RATE_STAGE1_HEAD,
            "learning_rate_stage2_head": config.LEARNING_RATE_STAGE2_HEAD,
            "learning_rate_stage2_lora": config.LEARNING_RATE_STAGE2_LORA,
        }
    )
    
    print(f"Device: {config.DEVICE}")
    
    print("\n" + "="*80)
    print("Initializing model...")
    print("="*80)
    model = GraphToTextModel(freeze_llm=True) # Initially freeze everything in LLM
    model = model.to(config.DEVICE)
    
    print("\n" + "="*80)
    print("Loading datasets...")
    print("="*80)
    train_dataset = MoleculeGraphDataset(config.TRAIN_GRAPHS, model.tokenizer, max_length=config.MAX_LENGTH)
    val_dataset = MoleculeGraphDataset(config.VAL_GRAPHS, model.tokenizer, max_length=config.MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    # ==================================================================================
    # STAGE 1: Train Q-Former + Projector (LLM Frozen)
    # ==================================================================================
    print("\nConfiguring Stage 1...")
    
    # Ensure LLM (including LoRA) is frozen
    for param in model.llm.parameters():
        param.requires_grad = False
        
    head_params = list(model.qformer.parameters()) + list(model.graph_to_llm_proj.parameters()) + [model.separator_token]
    
    optimizer_s1 = torch.optim.AdamW(
        [{'params': head_params, 'lr': config.LEARNING_RATE_STAGE1_HEAD}],
        weight_decay=config.WEIGHT_DECAY
    )
    
    scheduler_s1 = get_constant_schedule_with_warmup(optimizer_s1, num_warmup_steps=config.WARMUP_STEPS)
    
    run_training_phase(
        model, train_loader, val_loader, optimizer_s1, scheduler_s1, 
        num_epochs=config.EPOCHS_STAGE1, 
        start_epoch=1, 
        stage_name="Stage 1 (Frozen LLM)"
    )
    
    # ==================================================================================
    # STAGE 2: Joint Training (Q-Former + Projector + LLM LoRA)
    # ==================================================================================
    print("\nConfiguring Stage 2...")
    
    # Unfreeze LoRA parameters
    lora_params = []
    for name, param in model.llm.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
            lora_params.append(param)
        else:
            param.requires_grad = False # Ensure base weights stay frozen
            
    print(f"Unfrozen {len(lora_params)} LoRA parameters for Stage 2.")
    
    optimizer_s2 = torch.optim.AdamW(
        [
            {'params': head_params, 'lr': config.LEARNING_RATE_STAGE2_HEAD},
            {'params': lora_params, 'lr': config.LEARNING_RATE_STAGE2_LORA}
        ],
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Reset scheduler for Stage 2? Or continue? Usually reset or new warmup is good.
    # Let's use a new scheduler with warmup.
    scheduler_s2 = get_constant_schedule_with_warmup(optimizer_s2, num_warmup_steps=config.WARMUP_STEPS)
    
    run_training_phase(
        model, train_loader, val_loader, optimizer_s2, scheduler_s2, 
        num_epochs=config.EPOCHS_STAGE2, 
        start_epoch=config.EPOCHS_STAGE1 + 1, 
        stage_name="Stage 2 (Joint Training)"
    )
    
    print("\n" + "="*80)
    print("Training complete!")
    print("="*80)
    
    wandb.finish()


if __name__ == "__main__":
    main()
