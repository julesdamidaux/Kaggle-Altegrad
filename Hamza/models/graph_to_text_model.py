"""
Graph-to-Text Model combining MoleculeGINE encoder with T5 for molecular captioning.
"""

import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from peft import LoraConfig, get_peft_model, TaskType

from .graph_encoder import MoleculeGINE


class GraphToTextModel(nn.Module):
    """
    Complete model for molecular graph captioning:
    1. Graph Encoder: MoleculeGINE (encodes molecular graphs)
    2. Projection Layer: Projects graph embeddings to T5 hidden dimension
    3. T5 Decoder: Generates text descriptions
    """
    
    def __init__(
        self,
        node_dim=9,
        edge_dim=3,
        graph_hidden=256,
        graph_layers=5,
        t5_model_name="t5-small",
        use_lora=True,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        lora_target_modules=["q", "v"]
    ):
        super().__init__()
        
        # 1. Graph Encoder (MoleculeGINE)
        self.graph_encoder = MoleculeGINE(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden=graph_hidden,
            layers=graph_layers,
            out_dim=graph_hidden
        )
        
        # 2. Load T5-small for text generation
        print(f"Loading T5 model: {t5_model_name}")
        self.tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
        self.t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)
        
        # 3. Projection layer from graph embedding to T5 hidden dimension
        t5_hidden_size = self.t5_model.config.d_model  # 512 for t5-small
        self.graph_to_t5_proj = nn.Sequential(
            nn.Linear(graph_hidden, t5_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(t5_hidden_size, t5_hidden_size)
        )
        
        # Learnable prefix tokens to condition T5 on graph
        self.num_graph_tokens = 8
        self.graph_token_embedding = nn.Parameter(
            torch.randn(1, self.num_graph_tokens, t5_hidden_size)
        )
        
        # Apply LoRA to T5 if enabled
        if use_lora:
            self.apply_lora(lora_r, lora_alpha, lora_dropout, lora_target_modules)
    
    def apply_lora(self, lora_r, lora_alpha, lora_dropout, lora_target_modules):
        """Apply LoRA adapters to T5 model for parameter-efficient fine-tuning."""
        print("Applying LoRA to T5 model...")
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            bias="none"
        )
        self.t5_model = get_peft_model(self.t5_model, lora_config)
        self.t5_model.print_trainable_parameters()
    
    def encode_graph(self, graph):
        """Encode molecular graph using MoleculeGINE."""
        graph_embedding = self.graph_encoder(
            graph.x.float(),
            graph.edge_index,
            graph.edge_attr.float(),
            graph.batch
        )
        return graph_embedding
    
    def forward(self, graph, labels=None):
        """
        Forward pass for training.
        
        Args:
            graph: Batched molecular graph
            labels: Tokenized target descriptions (for computing loss)
        
        Returns:
            Dictionary with loss and logits
        """
        batch_size = graph.num_graphs
        
        # 1. Encode graph to get graph-level embeddings
        graph_embeds = self.encode_graph(graph)  # [batch_size, graph_hidden]
        
        # 2. Project to T5 dimension
        graph_embeds_proj = self.graph_to_t5_proj(graph_embeds)  # [batch_size, t5_hidden]
        graph_embeds_proj = graph_embeds_proj.unsqueeze(1)  # [batch_size, 1, t5_hidden]
        
        # 3. Add learnable graph tokens
        graph_tokens = self.graph_token_embedding.expand(batch_size, -1, -1)
        
        # Combine graph embedding with learnable tokens
        encoder_hidden_states = torch.cat([
            graph_embeds_proj,
            graph_tokens
        ], dim=1)  # [batch_size, num_graph_tokens + 1, t5_hidden]
        
        # Create attention mask for encoder hidden states (all ones, all tokens are valid)
        encoder_attention_mask = torch.ones(
            batch_size,
            self.num_graph_tokens + 1,
            dtype=torch.long,
            device=graph_embeds.device
        )
        
        # 4. Generate text with T5 decoder
        if labels is not None:
            # Training mode: compute loss
            # Replace padding token id's of the labels by -100 so it's ignored by the loss
            labels_masked = labels.clone()
            labels_masked[labels_masked == self.tokenizer.pad_token_id] = -100
            
            outputs = self.t5_model(
                encoder_outputs=(encoder_hidden_states,),
                attention_mask=encoder_attention_mask,
                labels=labels_masked,
                return_dict=True
            )
            
            return {
                'loss': outputs.loss,
                'logits': outputs.logits
            }
        else:
            # Inference mode: generate text
            return encoder_hidden_states, encoder_attention_mask
    
    def generate(self, graph, max_length=128, num_beams=4):
        """
        Generate text description for a molecular graph.
        
        Args:
            graph: Batched molecular graph
            max_length: Maximum length of generated text
            num_beams: Number of beams for beam search
        
        Returns:
            List of generated text descriptions
        """
        self.eval()
        with torch.no_grad():
            # Get encoder hidden states
            encoder_hidden_states, encoder_attention_mask = self.forward(graph, labels=None)
            
            # Wrap encoder outputs in the format expected by T5
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)
            
            # Generate with T5
            outputs = self.t5_model.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=encoder_attention_mask,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
            
            # Decode to text
            descriptions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            return descriptions
