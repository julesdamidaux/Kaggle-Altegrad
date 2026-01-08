"""
Fine-tune Qwen2.5-3B-Instruct for SMILES to description generation.

Usage:
    python Hamza/train/finetune_qwen.py
"""

import os
import sys
from pathlib import Path
import random

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import transformers

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Hamza.train.smiles_dataset import SmilesDescriptionDataset
from Hamza.train.finetune_config import (
    MODEL_NAME,
    OUTPUT_DIR,
    CACHE_DIR,
    TRAIN_CSV,
    VAL_CSV,
    BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    EFFECTIVE_BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    MAX_LENGTH,
    WARMUP_STEPS,
    LOGGING_STEPS,
    SAVE_STEPS,
    EVAL_STEPS,
    USE_LORA,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_TARGET_MODULES,
    USE_4BIT,
    USE_8BIT,
    get_prompt_template,
    DEVICE,
    FP16,
    BF16,
    USE_GRADIENT_CHECKPOINTING,
    EVAL_BATCH_SIZE,
    USE_WANDB,
    SAVE_TOTAL_LIMIT,
    LOAD_BEST_MODEL_AT_END,
    METRIC_FOR_BEST_MODEL,
)


def setup_model_and_tokenizer():
    """Load and configure the model and tokenizer."""
    print("=" * 80)
    print("Setting up model and tokenizer")
    print("=" * 80)
    
    # Load tokenizer
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
    )
    
    # Set pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Quantization config for memory efficiency
    quantization_config = None
    if USE_4BIT:
        print("Using 4-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if BF16 else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif USE_8BIT:
        print("Using 8-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    
    # Load model
    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        quantization_config=quantization_config,
        device_map="auto" if DEVICE == "cuda" else None,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if BF16 else torch.float16,
    )
    
    # Prepare model for k-bit training if using quantization
    if USE_4BIT or USE_8BIT:
        print("Preparing model for k-bit training")
        model = prepare_model_for_kbit_training(model)
    
    # Enable gradient checkpointing
    if USE_GRADIENT_CHECKPOINTING:
        print("Enabling gradient checkpointing")
        model.gradient_checkpointing_enable()
    
    # Apply LoRA
    if USE_LORA:
        print("Applying LoRA")
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, tokenizer


def load_datasets(tokenizer):
    """Load training and validation datasets."""
    print("\n" + "=" * 80)
    print("Loading datasets")
    print("=" * 80)
    
    train_dataset = SmilesDescriptionDataset(
        csv_path=TRAIN_CSV,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        prompt_template_fn=get_prompt_template,
    )
    
    val_dataset = SmilesDescriptionDataset(
        csv_path=VAL_CSV,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        prompt_template_fn=get_prompt_template,
    )
    
    return train_dataset, val_dataset


class SampleGenerationCallback(TrainerCallback):
    """Callback to generate and print sample predictions during training."""
    
    def __init__(self, tokenizer, val_dataset, num_samples=3):
        self.tokenizer = tokenizer
        self.val_dataset = val_dataset
        self.num_samples = num_samples
        # Select random samples once
        self.sample_indices = random.sample(range(len(val_dataset)), min(num_samples, len(val_dataset)))
    
    def on_evaluate(self, args, state, control, model, **kwargs):
        """Generate samples after each evaluation."""
        print("\n" + "=" * 80)
        print(f"Sample Generations at Step {state.global_step}")
        print("=" * 80)
        
        model.eval()
        with torch.no_grad():
            for idx in self.sample_indices:
                # Get the sample
                sample = self.val_dataset.data.iloc[idx]
                smiles = sample['SMILES']
                target_description = sample['Description']
                
                # Create inference prompt (without description)
                prompt = get_prompt_template(smiles, None)
                
                # Tokenize
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    max_length=MAX_LENGTH,
                    truncation=True,
                ).to(model.device)
                
                # Generate
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                
                # Decode
                generated_text = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
                
                print(f"\n{'─' * 80}")
                print(f"SMILES: {smiles}")
                print(f"\n🎯 TARGET:")
                print(f"{target_description}")
                print(f"\n🤖 GENERATED:")
                print(f"{generated_text.strip()}")
                print(f"{'─' * 80}")
        
        print("=" * 80 + "\n")
        model.train()


def print_random_examples(dataset, num_samples=5):
    """Print random training examples to understand the data."""
    print("\n" + "=" * 80)
    print("📚 Random Training Examples (Target Descriptions)")
    print("=" * 80)
    
    indices = random.sample(range(len(dataset.data)), min(num_samples, len(dataset.data)))
    
    for i, idx in enumerate(indices, 1):
        sample = dataset.data.iloc[idx]
        print(f"\n{i}. SMILES: {sample['SMILES']}")
        print(f"   🎯 TARGET DESCRIPTION:")
        print(f"   {sample['Description']}")
        print("─" * 80)
    
    print("=" * 80 + "\n")


def main():
    """Main training function."""
    print("=" * 80)
    print("Qwen2.5-0.5B-Instruct Fine-tuning for SMILES to Description")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Training data: {TRAIN_CSV}")
    print(f"Validation data: {VAL_CSV}")
    print(f"Device: {DEVICE}")
    print(f"Use LoRA: {USE_LORA}")
    print(f"Use 4-bit: {USE_4BIT}")
    print(f"Use 8-bit: {USE_8BIT}")
    print(f"Effective batch size: {EFFECTIVE_BATCH_SIZE}")
    print("=" * 80)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer()
    
    # Load datasets
    train_dataset, val_dataset = load_datasets(tokenizer)
    
    # Print random training examples
    print_random_examples(train_dataset, num_samples=5)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
    )
    
    # Training arguments
    print("\n" + "=" * 80)
    print("Setting up training arguments")
    print("=" * 80)
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=LOAD_BEST_MODEL_AT_END,
        metric_for_best_model=METRIC_FOR_BEST_MODEL,
        greater_is_better=False,  # Lower loss is better
        fp16=FP16 and not BF16,
        bf16=BF16,
        gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
        report_to="wandb" if USE_WANDB else "none",
        run_name=f"qwen2.5-3b-smiles-{Path(OUTPUT_DIR).name}",
        logging_dir=f"{OUTPUT_DIR}/logs",
        remove_unused_columns=False,
    )
    
    # Initialize trainer
    print("\n" + "=" * 80)
    print("Initializing trainer")
    print("=" * 80)
    
    # Create sample generation callback
    sample_callback = SampleGenerationCallback(
        tokenizer=tokenizer,
        val_dataset=val_dataset,
        num_samples=3
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[sample_callback],
    )
    
    # Start training
    print("\n" + "=" * 80)
    print("Starting training")
    print("=" * 80)
    
    trainer.train()
    
    # Save final model
    print("\n" + "=" * 80)
    print("Saving final model")
    print("=" * 80)
    
    trainer.save_model(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Model saved to: {OUTPUT_DIR}/final")
    print("=" * 80)


if __name__ == "__main__":
    main()
