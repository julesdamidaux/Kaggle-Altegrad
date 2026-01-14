"""
Fine-tune LLM with Graph Encoder, MLP Projection, and Learnable Prompts

This script:
1. Loads pre-trained graph encoder
2. Adds MLP projection to LLM embedding space
3. Adds learnable prompt tokens
4. Fine-tunes on molecule description generation
"""

import os
import sys
from pathlib import Path
import pickle
from dataclasses import dataclass
from typing import Optional, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Hamza.models.graph_encoder import MoleculeGINE
from data_baseline.data_utils import x_map, e_map


@dataclass
class ModelConfig:
    """Configuration for the graph-to-text model"""
    # LLM settings
    llm_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    cache_dir: str = "./cache"
    
    # Graph encoder settings
    graph_encoder_checkpoint: str = "/Data/hamzaazzouzi/Kaggle-Altegrad/Hamza/RAG+Predict/embeddings/encoder_checkpoints/best_encoder.pt"
    graph_hidden_dim: int = 256
    graph_layers: int = 5
    node_dim: int = 9
    edge_dim: int = 3
    
    # Projection and prompt settings
    num_learnable_prompts: int = 8  # Number of learnable prefix tokens
    num_graph_tokens: int = 4  # How many tokens to represent the graph
    
    # Training settings
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = None
    
    use_4bit: bool = True
    freeze_graph_encoder: bool = True  # Whether to freeze the pre-trained graph encoder
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]


class GraphProjector(nn.Module):
    """MLP that projects graph encodings to LLM embedding space"""
    
    def __init__(self, graph_dim: int, llm_dim: int, num_output_tokens: int = 4):
        super().__init__()
        self.num_output_tokens = num_output_tokens
        
        # Multi-layer projection
        hidden_dim = (graph_dim + llm_dim) // 2
        
        self.projector = nn.Sequential(
            nn.Linear(graph_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, llm_dim * num_output_tokens),
            nn.GELU(),
            nn.LayerNorm(llm_dim * num_output_tokens),
        )
        
        # Initialize with small values
        self._init_weights()
    
    def _init_weights(self):
        """Initialize projection weights"""
        for module in self.projector.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, graph_encodings: torch.Tensor) -> torch.Tensor:
        """
        Project graph encodings to LLM embedding space
        
        Args:
            graph_encodings: [batch_size, graph_dim]
        
        Returns:
            projected: [batch_size, num_output_tokens, llm_dim]
        """
        batch_size = graph_encodings.size(0)
        
        # Project to multiple tokens
        projected = self.projector(graph_encodings)
        # [batch_size, llm_dim * num_output_tokens]
        
        # Reshape to separate tokens
        projected = projected.view(batch_size, self.num_output_tokens, -1)
        # [batch_size, num_output_tokens, llm_dim]
        
        return projected


class GraphLLMModel(nn.Module):
    """
    Complete model: Graph Encoder → MLP Projection → Learnable Prompts → LLM
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.supports_gradient_checkpointing = True
        
        # 1. Load pre-trained graph encoder
        print("Loading pre-trained graph encoder...")
        self.graph_encoder = MoleculeGINE(
            node_dim=config.node_dim,
            edge_dim=config.edge_dim,
            hidden=config.graph_hidden_dim,
            layers=config.graph_layers,
            out_dim=config.graph_hidden_dim,
        )
        
        # Load checkpoint
        checkpoint = torch.load(config.graph_encoder_checkpoint, map_location='cpu')
        self.graph_encoder.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded graph encoder from {config.graph_encoder_checkpoint}")
        
        # Freeze graph encoder if specified
        if config.freeze_graph_encoder:
            for param in self.graph_encoder.parameters():
                param.requires_grad = False
            print("✓ Graph encoder frozen")
        
        # 2. Load LLM
        print(f"Loading LLM: {config.llm_model_name}")
        
        # Quantization config
        if config.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            self.llm = AutoModelForCausalLM.from_pretrained(
                config.llm_model_name,
                quantization_config=bnb_config,
                cache_dir=config.cache_dir,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            self.llm = AutoModelForCausalLM.from_pretrained(
                config.llm_model_name,
                cache_dir=config.cache_dir,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
        
        # Get LLM embedding dimension
        self.llm_embed_dim = self.llm.config.hidden_size
        print(f"✓ LLM embedding dimension: {self.llm_embed_dim}")
        
        # 3. MLP Projector: Graph → LLM embedding space
        print(f"Creating MLP projector: {config.graph_hidden_dim} → {self.llm_embed_dim}")
        self.graph_projector = GraphProjector(
            graph_dim=config.graph_hidden_dim,
            llm_dim=self.llm_embed_dim,
            num_output_tokens=config.num_graph_tokens,
        )
        print(f"✓ Graph will be projected to {config.num_graph_tokens} tokens")
        
        # 4. Learnable prompt tokens
        print(f"Creating {config.num_learnable_prompts} learnable prompt tokens")
        self.learnable_prompts = nn.Parameter(
            torch.randn(config.num_learnable_prompts, self.llm_embed_dim) * 0.02
        )
        
        # 5. Apply LoRA to LLM
        if config.use_lora:
            print("Applying LoRA to LLM...")
            if config.use_4bit:
                self.llm = prepare_model_for_kbit_training(self.llm)
            
            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                target_modules=config.lora_target_modules,
                lora_dropout=config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.llm = get_peft_model(self.llm, lora_config)
            self.llm.print_trainable_parameters()
        
        print("\n" + "="*80)
        print("MODEL ARCHITECTURE:")
        print("="*80)
        print(f"1. Graph Encoder: MoleculeGINE ({config.graph_hidden_dim}D)")
        print(f"   └─ Status: {'Frozen' if config.freeze_graph_encoder else 'Trainable'}")
        print(f"2. MLP Projector: {config.graph_hidden_dim}D → {config.num_graph_tokens} × {self.llm_embed_dim}D")
        print(f"   └─ Status: Trainable")
        print(f"3. Learnable Prompts: {config.num_learnable_prompts} tokens × {self.llm_embed_dim}D")
        print(f"   └─ Status: Trainable")
        print(f"4. LLM: {config.llm_model_name}")
        print(f"   └─ Status: LoRA fine-tuning (r={config.lora_r})")
        print(f"\nTotal sequence: [{config.num_learnable_prompts} prompts] + [{config.num_graph_tokens} graph tokens] + [text tokens]")
        print("="*80 + "\n")
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for the LLM"""
        if hasattr(self.llm, 'gradient_checkpointing_enable'):
            self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for the LLM"""
        if hasattr(self.llm, 'gradient_checkpointing_disable'):
            self.llm.gradient_checkpointing_disable()
    
    def forward(
        self,
        graph_data,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass
        
        Args:
            graph_data: PyG graph batch
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            labels: [batch_size, seq_len]
        """
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # 1. Encode graphs
        with torch.set_grad_enabled(not self.config.freeze_graph_encoder):
            graph_encodings = self.graph_encoder(
                graph_data.x.float(),
                graph_data.edge_index,
                graph_data.edge_attr.float(),
                graph_data.batch,
            )  # [batch_size, graph_hidden_dim]
        
        # 2. Project to LLM embedding space
        graph_embeds = self.graph_projector(graph_encodings)
        # [batch_size, num_graph_tokens, llm_embed_dim]
        
        # 3. Get learnable prompt embeddings
        prompt_embeds = self.learnable_prompts.unsqueeze(0).expand(batch_size, -1, -1)
        # [batch_size, num_learnable_prompts, llm_embed_dim]
        
        # 4. Get text embeddings
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        # [batch_size, seq_len, llm_embed_dim]
        
        # 5. Concatenate: [prompts] + [graph tokens] + [text]
        combined_embeds = torch.cat([prompt_embeds, graph_embeds, text_embeds], dim=1)
        # [batch_size, num_learnable_prompts + num_graph_tokens + seq_len, llm_embed_dim]
        
        # 6. Create attention mask for the combined sequence
        graph_attention = torch.ones(
            batch_size,
            self.config.num_learnable_prompts + self.config.num_graph_tokens,
            dtype=attention_mask.dtype,
            device=device,
        )
        combined_attention_mask = torch.cat([graph_attention, attention_mask], dim=1)
        
        # 7. Adjust labels to account for prefix tokens
        if labels is not None:
            # Prepend -100 (ignore index) for prompt and graph tokens
            prefix_labels = torch.full(
                (batch_size, self.config.num_learnable_prompts + self.config.num_graph_tokens),
                -100,
                dtype=labels.dtype,
                device=device,
            )
            combined_labels = torch.cat([prefix_labels, labels], dim=1)
        else:
            combined_labels = None
        
        # 8. Forward through LLM
        outputs = self.llm(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention_mask,
            labels=combined_labels,
            return_dict=True,
        )
        
        return outputs


class MoleculeGraphDataset(Dataset):
    """Dataset that loads graphs and their descriptions"""
    
    def __init__(self, graph_path: str, tokenizer, max_length: int = 512):
        print(f"Loading graphs from {graph_path}")
        with open(graph_path, 'rb') as f:
            self.graphs = pickle.load(f)
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print(f"✓ Loaded {len(self.graphs)} graphs")
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        graph = self.graphs[idx]
        description = graph.description if hasattr(graph, 'description') else ""
        
        # Tokenize description
        text = f"<|im_start|>user\nDescribe this molecule.<|im_end|>\n<|im_start|>assistant\n{description}<|im_end|>"
        
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            'graph': graph,
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'labels': encoded['input_ids'].squeeze(0).clone(),
        }


def collate_fn(batch):
    """Custom collate function to handle graph batching"""
    from torch_geometric.data import Batch as PyGBatch
    
    # Separate graphs and text data
    graphs = [item['graph'] for item in batch]
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    # Batch graphs using PyG
    graph_batch = PyGBatch.from_data_list(graphs)
    
    return {
        'graph_data': graph_batch,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }


class GraphLLMTrainer(Trainer):
    """Custom trainer that handles graph data"""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Override to handle graph inputs"""
        outputs = model(
            graph_data=inputs['graph_data'],
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            labels=inputs['labels'],
        )
        
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss
    
    def _save(self, output_dir=None, state_dict=None):
        """Override save to handle shared tensors properly"""
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Use save_model method which handles shared tensors
        self.model.llm.save_pretrained(
            output_dir,
            safe_serialization=True,
            max_shard_size="5GB"
        )
        
        # Save the custom components (graph projector and learnable prompts)
        custom_state = {
            'graph_projector': self.model.graph_projector.state_dict(),
            'learnable_prompts': self.model.learnable_prompts,
        }
        torch.save(custom_state, os.path.join(output_dir, 'custom_components.pt'))


def main():
    """Main training function"""
    
    # Import configuration
    try:
        from Hamza.train import graph_llm_config as cfg
        print("✓ Loaded configuration from graph_llm_config.py")
    except:
        print("Warning: Could not load graph_llm_config.py, using defaults")
        class cfg:
            pass
        cfg.LLM_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
        cfg.CACHE_DIR = "./cache"
        cfg.GRAPH_ENCODER_CHECKPOINT = "/Data/hamzaazzouzi/Kaggle-Altegrad/Hamza/RAG+Predict/embeddings/encoder_checkpoints/best_encoder.pt"
        cfg.OUTPUT_DIR = "./Hamza/checkpoints/qwen2.5-0.5b-graph-mlp-prompts"
        cfg.TRAIN_GRAPHS = "/Data/hamzaazzouzi/Kaggle-Altegrad/data_baseline/data/train_graphs.pkl"
        cfg.VAL_GRAPHS = "/Data/hamzaazzouzi/Kaggle-Altegrad/data_baseline/data/validation_graphs.pkl"
        cfg.NUM_LEARNABLE_PROMPTS = 8
        cfg.NUM_GRAPH_TOKENS = 4
        cfg.FREEZE_GRAPH_ENCODER = True
        cfg.USE_LORA = True
        cfg.LORA_R = 16
        cfg.LORA_ALPHA = 32
        cfg.LORA_DROPOUT = 0.05
        cfg.LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
        cfg.USE_4BIT = True
        cfg.GRAPH_HIDDEN_DIM = 256
        cfg.GRAPH_LAYERS = 5
        cfg.NODE_DIM = 9
        cfg.EDGE_DIM = 3
        cfg.NUM_EPOCHS = 1
        cfg.BATCH_SIZE = 2
        cfg.GRADIENT_ACCUMULATION_STEPS = 8
        cfg.LEARNING_RATE = 2e-4
        cfg.WARMUP_STEPS = 100
        cfg.LOGGING_STEPS = 10
        cfg.SAVE_STEPS = 500
        cfg.SAVE_TOTAL_LIMIT = 3
        cfg.BF16 = True
        cfg.USE_GRADIENT_CHECKPOINTING = True
        cfg.MAX_LENGTH = 512
    
    # Configuration
    config = ModelConfig()
    config.llm_model_name = cfg.LLM_MODEL_NAME
    config.cache_dir = cfg.CACHE_DIR
    config.graph_encoder_checkpoint = cfg.GRAPH_ENCODER_CHECKPOINT
    config.graph_hidden_dim = cfg.GRAPH_HIDDEN_DIM
    config.graph_layers = cfg.GRAPH_LAYERS
    config.node_dim = cfg.NODE_DIM
    config.edge_dim = cfg.EDGE_DIM
    config.num_learnable_prompts = cfg.NUM_LEARNABLE_PROMPTS
    config.num_graph_tokens = cfg.NUM_GRAPH_TOKENS
    config.use_lora = cfg.USE_LORA
    config.lora_r = cfg.LORA_R
    config.lora_alpha = cfg.LORA_ALPHA
    config.lora_dropout = cfg.LORA_DROPOUT
    config.lora_target_modules = cfg.LORA_TARGET_MODULES
    config.use_4bit = cfg.USE_4BIT
    config.freeze_graph_encoder = cfg.FREEZE_GRAPH_ENCODER
    
    # Output directory
    output_dir = cfg.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    # Data paths
    train_graph_path = cfg.TRAIN_GRAPHS
    val_graph_path = getattr(cfg, 'VAL_GRAPHS', None)
    
    # Load tokenizer
    print(f"Loading tokenizer: {config.llm_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.llm_model_name,
        cache_dir=config.cache_dir,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create model
    print("\n" + "="*80)
    print("INITIALIZING MODEL")
    print("="*80)
    model = GraphLLMModel(config)
    
    # Create dataset
    print("\n" + "="*80)
    print("LOADING DATASET")
    print("="*80)
    train_dataset = MoleculeGraphDataset(
        graph_path=train_graph_path,
        tokenizer=tokenizer,
        max_length=getattr(cfg, 'MAX_LENGTH', 512),
    )
    
    # Load validation dataset if available
    val_dataset = None
    if val_graph_path and os.path.exists(val_graph_path):
        print(f"Loading validation dataset from {val_graph_path}")
        val_dataset = MoleculeGraphDataset(
            graph_path=val_graph_path,
            tokenizer=tokenizer,
            max_length=getattr(cfg, 'MAX_LENGTH', 512),
        )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=getattr(cfg, 'NUM_EPOCHS', 1),
        per_device_train_batch_size=getattr(cfg, 'BATCH_SIZE', 2),
        per_device_eval_batch_size=getattr(cfg, 'BATCH_SIZE', 2),
        gradient_accumulation_steps=getattr(cfg, 'GRADIENT_ACCUMULATION_STEPS', 8),
        learning_rate=getattr(cfg, 'LEARNING_RATE', 2e-4),
        warmup_steps=getattr(cfg, 'WARMUP_STEPS', 100),
        logging_steps=getattr(cfg, 'LOGGING_STEPS', 10),
        save_steps=getattr(cfg, 'SAVE_STEPS', 500),
        eval_steps=getattr(cfg, 'EVAL_STEPS', 500) if val_dataset else None,
        save_total_limit=getattr(cfg, 'SAVE_TOTAL_LIMIT', 3),
        fp16=getattr(cfg, 'FP16', False),
        bf16=getattr(cfg, 'BF16', True),
        gradient_checkpointing=getattr(cfg, 'USE_GRADIENT_CHECKPOINTING', True),
        eval_strategy="steps" if val_dataset else "no",
        load_best_model_at_end=getattr(cfg, 'LOAD_BEST_MODEL_AT_END', True) if val_dataset else False,
        metric_for_best_model=getattr(cfg, 'METRIC_FOR_BEST_MODEL', 'loss'),
        report_to="wandb" if getattr(cfg, 'USE_WANDB', False) else "none",
        run_name=getattr(cfg, 'WANDB_RUN_NAME', None),
        remove_unused_columns=False,  # Important: don't remove graph data
    )
    
    # Create trainer
    trainer = GraphLLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )
    
    # Train
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    trainer.train()
    
    # Save final model
    print("\n" + "="*80)
    print("SAVING MODEL")
    print("="*80)
    trainer.save_model(os.path.join(output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final"))
    
    print(f"\n✓ Training complete! Model saved to {output_dir}/final")


if __name__ == "__main__":
    main()
