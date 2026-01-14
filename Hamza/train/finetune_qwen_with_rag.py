"""
Fine-tune Qwen2.5 with graph encodings and RAG for molecule description generation.

This script integrates:
1. Graph encodings from trained graph encoder
2. Retrieved descriptions from similar molecules (RAG)
3. Learnable separator tokens between graph and text

Usage:
    python Hamza/train/finetune_qwen_with_rag.py
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
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import transformers

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Hamza.train.smiles_dataset_rag import SmilesDescriptionRAGDataset
from Hamza.train.qwen_with_graph import QwenWithGraphEncoding
from Hamza.train.finetune_config import (
    MODEL_NAME,
    OUTPUT_DIR,
    CACHE_DIR,
    BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
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
    DEVICE,
    FP16,
    BF16,
    USE_GRADIENT_CHECKPOINTING,
    EVAL_BATCH_SIZE,
    USE_WANDB,
    SAVE_TOTAL_LIMIT,
    LOAD_BEST_MODEL_AT_END,
    METRIC_FOR_BEST_MODEL,
    SYSTEM_PROMPT,
)

# Override output directory for RAG model
OUTPUT_DIR_RAG = OUTPUT_DIR.replace("qwen2.5", "qwen2.5-rag")


def get_prompt_template_with_rag(retrieved_descriptions: list = None, description: str = None) -> str:
    """
    Format the input/output with RAG context.
    
    Args:
        retrieved_descriptions: List of retrieved descriptions from similar molecules
        description: Expected description (for training, None for inference)
    
    Returns:
        Formatted prompt string
    """
    # Format retrieved descriptions
    if retrieved_descriptions:
        retrieved_text = "Here are descriptions of structurally similar molecules:\n"
        for i, desc in enumerate(retrieved_descriptions[:5], 1):
            retrieved_text += f"{i}. {desc}\n"
        retrieved_text += "\n"
    else:
        retrieved_text = ""
    
    user = f"{retrieved_text}Write the ontology-style description for this molecule."
    
    if description is not None:
        return f"""<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
Description: {description}<|im_end|>"""
    else:
        return f"""<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
Description:"""


def setup_model_and_tokenizer(graph_encoding_dim=256):
    """Load and configure the model with graph encoding support."""
    print("=" * 80)
    print("Setting up model and tokenizer with graph encoding")
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
    
    # Load base model
    print(f"Loading base model: {MODEL_NAME}")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        quantization_config=quantization_config,
        device_map="auto" if DEVICE == "cuda" else None,
        trust_remote_code=True,
        dtype=torch.bfloat16 if BF16 else torch.float16,
    )
    
    # Prepare model for k-bit training if using quantization
    if USE_4BIT or USE_8BIT:
        print("Preparing model for k-bit training")
        base_model = prepare_model_for_kbit_training(base_model)
    
    # Wrap with graph encoding support
    print(f"Wrapping model with graph encoding (dim={graph_encoding_dim})")
    model = QwenWithGraphEncoding(
        base_model=base_model,
        graph_encoding_dim=graph_encoding_dim,
        num_graph_tokens=4,
        num_separator_tokens=2,
    )
    
    # Enable gradient checkpointing
    if USE_GRADIENT_CHECKPOINTING:
        print("Enabling gradient checkpointing")
        model.base_model.gradient_checkpointing_enable()
    
    # Apply LoRA to base model
    if USE_LORA:
        print("Applying LoRA to base model")
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model.base_model = get_peft_model(model.base_model, lora_config)
        model.base_model.print_trainable_parameters()
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    return model, tokenizer


def load_datasets(tokenizer, train_data_path, val_data_path, graph_encoding_dim=256):
    """Load training and validation datasets."""
    print("\n" + "=" * 80)
    print("Loading RAG datasets with 5NN retrieval")
    print("=" * 80)
    
    train_dataset = SmilesDescriptionRAGDataset(
        data_pkl_path=train_data_path,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        prompt_template_fn=get_prompt_template_with_rag,
        num_retrieved=5,  # Using 5 nearest neighbor descriptions
    )
    
    val_dataset = SmilesDescriptionRAGDataset(
        data_pkl_path=val_data_path,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        prompt_template_fn=get_prompt_template_with_rag,
        num_retrieved=5,  # Using 5 nearest neighbor descriptions
    )
    
    # Show example of how 5NN RAG is used
    print("\n" + "-" * 80)
    print("Example training input with 5NN RAG:")
    print("-" * 80)
    sample_item = train_dataset.data[0]
    retrieved_descs = [
        sample_item.get(f'retrieved_desc_{i}', '') 
        for i in range(1, 6) 
        if sample_item.get(f'retrieved_desc_{i}', '')
    ]
    example_prompt = get_prompt_template_with_rag(
        retrieved_descriptions=retrieved_descs,
        description=sample_item['Description'][:100] + "..."
    )
    print(example_prompt[:800] + "...\n" if len(example_prompt) > 800 else example_prompt + "\n")
    print(f"Total samples: {len(train_dataset)}")
    print(f"Each sample includes: Graph encoding (256-dim) + 5NN descriptions + Target")
    print("-" * 80)
    
    return train_dataset, val_dataset


class CustomDataCollator:
    """Custom collator that handles graph encodings."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, features):
        # Separate graph encodings from text features
        graph_encodings = torch.stack([f.pop('graph_encoding') for f in features])
        
        # Standard collation for text
        batch = {
            'input_ids': torch.stack([f['input_ids'] for f in features]),
            'attention_mask': torch.stack([f['attention_mask'] for f in features]),
            'labels': torch.stack([f['labels'] for f in features]),
            'graph_encodings': graph_encodings,
        }
        
        return batch


class CustomTrainer(Trainer):
    """Custom trainer that passes graph encodings to the model."""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        graph_encodings = inputs.pop('graph_encodings')
        
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            graph_encodings=graph_encodings,
            labels=inputs['labels'],
        )
        
        loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss


def main():
    print("\n" + "=" * 80)
    print("FINE-TUNING QWEN WITH GRAPH ENCODINGS + 5NN RAG")
    print("=" * 80)
    print("\nTraining Pipeline:")
    print("  1. Encode molecular graph → 256-dim embedding")
    print("  2. Retrieve 5 most similar molecules (5NN)")
    print("  3. Add 5NN descriptions to system prompt")
    print("  4. Project graph encoding → 4 tokens")
    print("  5. Train LLM: [graph_tokens] + [separator] + [text with 5NN]")
    print("=" * 80)
    
    # Data paths (from RAG pipeline)
    train_data_path = "./Hamza/RAG+Predict/embeddings/train_llm_data.pkl"
    # For validation, we can use a subset of train or create a separate val split
    val_data_path = train_data_path  # For now, use same data (will split internally if needed)
    
    # Check if data exists
    if not os.path.exists(train_data_path):
        print(f"ERROR: Training data not found at {train_data_path}")
        print("Please run the RAG pipeline first: python Hamza/RAG+Predict/complete_rag_pipeline.py")
        return
    
    # Setup
    graph_encoding_dim = 256  # Should match the encoder output dimension
    model, tokenizer = setup_model_and_tokenizer(graph_encoding_dim=graph_encoding_dim)
    train_dataset, val_dataset = load_datasets(
        tokenizer, train_data_path, val_data_path, graph_encoding_dim
    )
    
    # Data collator
    data_collator = CustomDataCollator(tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR_RAG,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        eval_strategy="steps",  # Must match save_strategy for load_best_model_at_end
        save_strategy="steps",
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=LOAD_BEST_MODEL_AT_END,
        metric_for_best_model=METRIC_FOR_BEST_MODEL,
        greater_is_better=False,  # Lower loss is better
        fp16=FP16 and not BF16,
        bf16=BF16,
        report_to="wandb" if USE_WANDB else "none",
        remove_unused_columns=False,  # Keep graph encodings
        gradient_checkpointing=False,  # Disabled for custom wrapper model
    )
    
    # Create trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    trainer.train()
    
    # Save final model
    print("\n" + "=" * 80)
    print("Saving final model...")
    print("=" * 80)
    final_dir = os.path.join(OUTPUT_DIR_RAG, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print(f"Model saved to: {final_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
