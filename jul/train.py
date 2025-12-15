"""Training script for Graph-to-Text model with 3-Stage MolCA Pipeline."""

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


def train_epoch_stage1(model, dataloader, optimizer, scheduler, device, epoch):
    """Stage 1: Alignment (ITC + ITM + ITG) with Frozen LLM."""
    model.train()
    # Graphormer handles its own train/eval mode based on freeze config
    # but we ensure it's in train mode if unfrozen
    if not config.FREEZE_GRAPH_ENCODER:
        model.graph_encoder.graphormer.train()
        
    total_loss = 0
    total_loss_itc = 0
    total_loss_itm = 0
    total_loss_itg = 0
    num_batches = 0
    
    optimizer.zero_grad()
    
    # Initialize lr_head for progress bar
    lr_head_group = max([param_group['lr'] for param_group in optimizer.param_groups])
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Stage 1 - Alignment]")
    for batch_idx, batch in enumerate(pbar):
        batch['graph'] = batch['graph'].to(device)
        batch['input_ids'] = batch['input_ids'].to(device)
        batch['attention_mask'] = batch['attention_mask'].to(device)
        
        # Forward with stage=1 for multi-objective training
        outputs = model(batch, stage=1)
        loss = outputs['loss']
        
        loss = loss / config.GRADIENT_ACCUMULATION_STEPS
        loss.backward()
        
        is_last_batch = (batch_idx + 1) == len(dataloader)
        if (batch_idx + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0 or is_last_batch:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Log to wandb
            current_avg_loss = total_loss / (num_batches + 1e-6)
            wandb.log({
                'train/loss_step': loss.item() * config.GRADIENT_ACCUMULATION_STEPS,
                'train/loss_avg': current_avg_loss,
                'train/loss_itc': outputs['loss_itc'].item(),
                'train/loss_itm': outputs['loss_itm'].item(),
                'train/loss_itg': outputs['loss_itg'].item(),
                'train/lr_max': max([g['lr'] for g in optimizer.param_groups]),
                'train/step': epoch * len(dataloader) // config.GRADIENT_ACCUMULATION_STEPS + (batch_idx + 1) // config.GRADIENT_ACCUMULATION_STEPS
            })
        
        total_loss += loss.item() * config.GRADIENT_ACCUMULATION_STEPS
        total_loss_itc += outputs['loss_itc'].item()
        total_loss_itm += outputs['loss_itm'].item()
        total_loss_itg += outputs['loss_itg'].item()
        num_batches += 1
        
        pbar.set_postfix({
            'loss': total_loss / num_batches,
            'itc': total_loss_itc / num_batches,
            'itm': total_loss_itm / num_batches,
            'itg': total_loss_itg / num_batches,
        })

    return total_loss / num_batches


def train_epoch_generative(model, dataloader, optimizer, scheduler, device, epoch, stage_name):
    """Stages 2 & 3: Generative Training (LLM Loss)."""
    model.train()
    if not config.FREEZE_GRAPH_ENCODER:
        model.graph_encoder.graphormer.train()

    total_loss = 0
    num_batches = 0
    
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [{stage_name}]")
    for batch_idx, batch in enumerate(pbar):
        batch['graph'] = batch['graph'].to(device)
        batch['input_ids'] = batch['input_ids'].to(device)
        batch['attention_mask'] = batch['attention_mask'].to(device)
        
        labels = batch['input_ids'].clone()
        loss = model(batch, labels=labels, stage=2) # Uses stage=2 logic for generation path
        
        loss = loss / config.GRADIENT_ACCUMULATION_STEPS
        loss.backward()
        
        is_last_batch = (batch_idx + 1) == len(dataloader)
        if (batch_idx + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0 or is_last_batch:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Log to wandb
            current_avg_loss = total_loss / (num_batches + 1e-6)
            wandb.log({
                'train/loss_step': loss.item() * config.GRADIENT_ACCUMULATION_STEPS,
                'train/loss_avg': current_avg_loss,
                'train/lr_max': max([g['lr'] for g in optimizer.param_groups]),
                'train/step': epoch * len(dataloader) // config.GRADIENT_ACCUMULATION_STEPS + (batch_idx + 1) // config.GRADIENT_ACCUMULATION_STEPS
            })
        
        total_loss += loss.item() * config.GRADIENT_ACCUMULATION_STEPS
        num_batches += 1
        
        pbar.set_postfix({
            'loss': total_loss / num_batches, 
        })

    return total_loss / num_batches


@torch.no_grad()
def validate_stage1(model, dataloader, device):
    """Validate Stage 1 (multi-objective)."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Validating [Stage 1]"):
        batch['graph'] = batch['graph'].to(device)
        batch['input_ids'] = batch['input_ids'].to(device)
        batch['attention_mask'] = batch['attention_mask'].to(device)
        
        outputs = model(batch, stage=1)
        loss = outputs['loss']
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


@torch.no_grad()
def validate_generative(model, dataloader, device):
    """Validate Stage 2 & 3 (LLM Loss)."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Validating [Generative]"):
        batch['graph'] = batch['graph'].to(device)
        batch['input_ids'] = batch['input_ids'].to(device)
        batch['attention_mask'] = batch['attention_mask'].to(device)
        
        labels = batch['input_ids'].clone()
        loss = model(batch, labels=labels, stage=2)
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


@torch.no_grad()
def generate_sample_captions(model, dataloader, device, num_samples=3):
    """Generate sample captions for logging to wandb."""
    model.eval()
    
    samples = []
    for i, batch in enumerate(dataloader):
        if i >= num_samples:
            break
            
        batch['graph'] = batch['graph'].to(device)
        
        # Generate caption
        captions = model.generate(
            batch,
            max_new_tokens=config.GENERATION_MAX_NEW_TOKENS,
            min_new_tokens=config.GENERATION_MIN_NEW_TOKENS,
            num_beams=1,
            temperature=0.3, # Low temp for check
            top_p=0.9,
            repetition_penalty=1.2
        )
        
        # Get the ground truth caption
        gt_caption = batch['texts'][0] if 'texts' in batch else "N/A"
        
        samples.append({
            'id': batch['ids'][0] if 'ids' in batch else f"sample_{i}",
            'generated': captions[0],
            'ground_truth': gt_caption
        })
    
    return samples


def run_training_phase(model, train_loader, val_loader, optimizer, scheduler, num_epochs, start_epoch, stage_name, mode='generative'):
    """Run a training phase."""
    print(f"\n" + "="*80)
    print(f"Starting {stage_name}")
    print("="*80)
    
    best_val_loss = float('inf')
    
    if mode == 'stage1':
        train_fn = train_epoch_stage1
        val_fn = validate_stage1
    else:
        train_fn = lambda m, dl, opt, sch, dev, ep: train_epoch_generative(m, dl, opt, sch, dev, ep, stage_name)
        val_fn = validate_generative
    
    for epoch in range(start_epoch, start_epoch + num_epochs):
        print(f"\nEpoch {epoch}/{start_epoch + num_epochs - 1} ({stage_name})")
        
        train_loss = train_fn(model, train_loader, optimizer, scheduler, config.DEVICE, epoch)
        val_loss = val_fn(model, val_loader, config.DEVICE)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Generate sample captions for logging
        print("Generating sample captions...")
        sample_captions = generate_sample_captions(model, val_loader, config.DEVICE, num_samples=3)
        
        # Create wandb Table for captions
        caption_table = wandb.Table(columns=["ID", "Generated Caption", "Ground Truth"])
        for sample in sample_captions:
            caption_table.add_data(sample['id'], sample['generated'], sample['ground_truth'])
        
        # Log to wandb
        wandb.log({
            'epoch': epoch,
            'train/loss_epoch': train_loss,
            'val/loss_epoch': val_loss,
            'train_loss': train_loss, # Explicit key as requested
            'val_loss': val_loss,     # Explicit key as requested
            'stage_name': stage_name,
            'sample_captions': caption_table
        })
        
        # Print sample captions to console
        print("\nSample Generated Captions:")
        for i, sample in enumerate(sample_captions, 1):
            print(f"  {i}. {sample['generated'][:100]}...")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            main_best_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
            
            state_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'stage': stage_name
            }
            
            torch.save(state_dict, main_best_path) # Always update main best
            print(f"New best model saved! Val Loss: {val_loss:.4f}")
        
    return best_val_loss


def main():
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    
    # Initialize wandb
    wandb.init(
        project="graph-to-text-altegrad",
        name=f"llama-3b-molca-3stage-unfrozen",
        config={
            "model": config.LLM_MODEL_NAME,
            "epochs_stage1": config.EPOCHS_STAGE1,
            "epochs_stage2": config.EPOCHS_STAGE2,
            "epochs_stage3": config.EPOCHS_STAGE3,
            "batch_size": config.BATCH_SIZE,
            "gradient_accumulation": config.GRADIENT_ACCUMULATION_STEPS,
            "unfrozen_encoder": not config.FREEZE_GRAPH_ENCODER
        }
    )
    # Define metrics to ensure they are plotted against 'epoch'
    wandb.define_metric("epoch")
    wandb.define_metric("train/loss_epoch", step_metric="epoch")
    wandb.define_metric("val/loss_epoch", step_metric="epoch")
    wandb.define_metric("train_loss", step_metric="epoch")
    wandb.define_metric("val_loss", step_metric="epoch")
    
    print(f"Device: {config.DEVICE}")
    
    print("\n" + "="*80)
    print("Initializing model...")
    print("="*80)
    model = GraphToTextModel(freeze_llm=True, token=config.hf_token)
    model = model.to(config.DEVICE)
    
    print("\n" + "="*80)
    print("Loading datasets...")
    print("="*80)
    train_dataset = MoleculeGraphDataset(config.TRAIN_GRAPHS, model.tokenizer, max_length=config.MAX_LENGTH)
    val_dataset = MoleculeGraphDataset(config.VAL_GRAPHS, model.tokenizer, max_length=config.MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    # ==================================================================================
    # STAGE 1: Alignment (Graph Encoder + Q-Former) - Multi-objective
    # ==================================================================================
    print("\nConfiguring Stage 1...")
    
    # Freeze LLM
    for param in model.llm.parameters():
        param.requires_grad = False
        
    # Parameters to optimize: Q-Former, Text Projection, ITG Head, Temp, AND Graph Encoder
    stage1_params = [
        {'params': model.qformer.parameters(), 'lr': config.LEARNING_RATE_STAGE1_HEAD},
        {'params': model.text_proj.parameters(), 'lr': config.LEARNING_RATE_STAGE1_HEAD},
        {'params': model.lm_head.parameters(), 'lr': config.LEARNING_RATE_STAGE1_HEAD},
        {'params': [model.temp], 'lr': config.LEARNING_RATE_STAGE1_HEAD}
    ]
    
    if not config.FREEZE_GRAPH_ENCODER:
        print(f"Adding Graph Encoder to optimizer (LR: {config.LEARNING_RATE_STAGE1_ENCODER})")
        stage1_params.append(
            {'params': model.graph_encoder.parameters(), 'lr': config.LEARNING_RATE_STAGE1_ENCODER}
        )
    
    optimizer_s1 = torch.optim.AdamW(stage1_params, weight_decay=config.WEIGHT_DECAY)
    scheduler_s1 = get_constant_schedule_with_warmup(optimizer_s1, num_warmup_steps=config.WARMUP_STEPS)
    
    run_training_phase(
        model, train_loader, val_loader, optimizer_s1, scheduler_s1, 
        num_epochs=config.EPOCHS_STAGE1, 
        start_epoch=1, 
        stage_name="Stage 1_Alignment",
        mode='stage1'
    )
    
    # ==================================================================================
    # STAGE 2: Pre-finetuning (Graph Encoder + Q-Former + Projector) - Generative
    # ==================================================================================
    print("\nConfiguring Stage 2...")
    
    # LLM still frozen
    # Optimize: Q-Former, Graph-to-LLM Projector, Graph Encoder
    stage2_params = [
        {'params': model.qformer.parameters(), 'lr': config.LEARNING_RATE_STAGE2_HEAD},
        {'params': model.graph_to_llm_proj.parameters(), 'lr': config.LEARNING_RATE_STAGE2_HEAD}
    ]

    if not config.FREEZE_GRAPH_ENCODER:
         stage2_params.append(
            {'params': model.graph_encoder.parameters(), 'lr': config.LEARNING_RATE_STAGE2_ENCODER}
        )
        
    optimizer_s2 = torch.optim.AdamW(stage2_params, weight_decay=config.WEIGHT_DECAY)
    scheduler_s2 = get_constant_schedule_with_warmup(optimizer_s2, num_warmup_steps=config.WARMUP_STEPS)
    
    run_training_phase(
        model, train_loader, val_loader, optimizer_s2, scheduler_s2, 
        num_epochs=config.EPOCHS_STAGE2, 
        start_epoch=config.EPOCHS_STAGE1 + 1, 
        stage_name="Stage 2_PreFinetune",
        mode='generative'
    )
    
    # ==================================================================================
    # STAGE 3: Full Finetuning (Graph Encoder + Q-Former + Projector + LoRA)
    # ==================================================================================
    print("\nConfiguring Stage 3...")
    
    # Unfreeze LoRA layers if they exist
    if config.USE_LORA:
        model.apply_lora()
    else:
        print("LoRA disabled. Skipping Stage 3 LLM tuning (this might be unexpected).")

    # Optimize: Everything
    stage3_params = [
        {'params': model.qformer.parameters(), 'lr': config.LEARNING_RATE_STAGE3_HEAD},
        {'params': model.graph_to_llm_proj.parameters(), 'lr': config.LEARNING_RATE_STAGE3_HEAD}
    ]
    
    if not config.FREEZE_GRAPH_ENCODER:
         stage3_params.append(
            {'params': model.graph_encoder.parameters(), 'lr': config.LEARNING_RATE_STAGE3_ENCODER}
        )
    
    if config.USE_LORA:
         stage3_params.append(
             {'params': filter(lambda p: p.requires_grad, model.llm.parameters()), 'lr': config.LEARNING_RATE_STAGE3_LORA}
         )

    optimizer_s3 = torch.optim.AdamW(stage3_params, weight_decay=config.WEIGHT_DECAY)
    scheduler_s3 = get_constant_schedule_with_warmup(optimizer_s3, num_warmup_steps=config.WARMUP_STEPS)
    
    run_training_phase(
        model, train_loader, val_loader, optimizer_s3, scheduler_s3, 
        num_epochs=config.EPOCHS_STAGE3, 
        start_epoch=config.EPOCHS_STAGE1 + config.EPOCHS_STAGE2 + 1, 
        stage_name="Stage 3_FullFinetune",
        mode='generative'
    )
    
    print("\n" + "="*80)
    print("Training complete!")
    print("="*80)
    
    wandb.finish()


if __name__ == "__main__":
    main()
