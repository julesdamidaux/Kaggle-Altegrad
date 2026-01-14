"""
Modified Qwen model with graph encoding integration.
Adds learnable tokens to separate graph encodings from text.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from typing import Optional, Tuple


class QwenWithGraphEncoding(nn.Module):
    """
    Wrapper around Qwen model that incorporates graph encodings.
    
    The model takes both text tokens and graph encodings as input.
    Graph encodings are projected to the token embedding space and
    prepended to the text sequence with learnable separator tokens.
    
    Architecture:
    [GRAPH_START] + graph_encoding + [GRAPH_END] + text_tokens
    """
    
    def __init__(
        self,
        base_model: AutoModelForCausalLM,
        graph_encoding_dim: int = 256,
        num_graph_tokens: int = 4,  # Number of tokens to represent graph encoding
        num_separator_tokens: int = 2,  # Number of learnable separator tokens
    ):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.graph_encoding_dim = graph_encoding_dim
        self.num_graph_tokens = num_graph_tokens
        self.num_separator_tokens = num_separator_tokens
        
        # Get embedding dimension from base model
        self.embed_dim = self.config.hidden_size
        
        # Project graph encodings to token embedding space
        # We project to multiple tokens to preserve information
        self.graph_projector = nn.Sequential(
            nn.Linear(graph_encoding_dim, self.embed_dim * num_graph_tokens),
            nn.LayerNorm(self.embed_dim * num_graph_tokens),
            nn.GELU(),
            nn.Linear(self.embed_dim * num_graph_tokens, self.embed_dim * num_graph_tokens),
        )
        
        # Learnable separator tokens
        # These mark the boundaries between graph encoding and text
        self.separator_embeddings = nn.Parameter(
            torch.randn(num_separator_tokens, self.embed_dim) * 0.02
        )
        
        # Initialize projector
        self._init_projector()
    
    def _init_projector(self):
        """Initialize the graph projector with small values."""
        for module in self.graph_projector.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        graph_encodings: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Forward pass with graph encodings.
        
        Args:
            input_ids: [batch_size, seq_len] - text token IDs
            attention_mask: [batch_size, seq_len] - attention mask for text
            graph_encodings: [batch_size, graph_dim] - graph encodings
            labels: [batch_size, seq_len] - labels for language modeling
        """
        batch_size = input_ids.size(0)
        
        if graph_encodings is not None:
            # Get text embeddings from base model
            text_embeds = self.base_model.get_input_embeddings()(input_ids)
            # [batch_size, seq_len, embed_dim]
            
            # Project graph encodings to multiple tokens
            graph_projected = self.graph_projector(graph_encodings)
            # [batch_size, embed_dim * num_graph_tokens]
            
            # Reshape to separate tokens
            graph_embeds = graph_projected.view(
                batch_size, self.num_graph_tokens, self.embed_dim
            )
            # [batch_size, num_graph_tokens, embed_dim]
            
            # Add separator tokens
            separator_embeds = self.separator_embeddings.unsqueeze(0).expand(
                batch_size, -1, -1
            )
            # [batch_size, num_separator_tokens, embed_dim]
            
            # Concatenate: [GRAPH_TOKENS] + [SEPARATOR] + [TEXT]
            combined_embeds = torch.cat([
                graph_embeds,
                separator_embeds,
                text_embeds
            ], dim=1)
            # [batch_size, num_graph_tokens + num_separator_tokens + seq_len, embed_dim]
            
            # Extend attention mask for graph tokens and separators
            prefix_mask = torch.ones(
                batch_size,
                self.num_graph_tokens + self.num_separator_tokens,
                dtype=attention_mask.dtype,
                device=attention_mask.device
            )
            combined_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            # [batch_size, num_graph_tokens + num_separator_tokens + seq_len]
            
            # Extend labels if provided
            if labels is not None:
                # Don't compute loss on graph tokens and separators
                prefix_labels = torch.full(
                    (batch_size, self.num_graph_tokens + self.num_separator_tokens),
                    -100,
                    dtype=labels.dtype,
                    device=labels.device
                )
                combined_labels = torch.cat([prefix_labels, labels], dim=1)
            else:
                combined_labels = None
            
            # Forward through base model with combined embeddings
            outputs = self.base_model(
                inputs_embeds=combined_embeds,
                attention_mask=combined_attention_mask,
                labels=combined_labels,
                **kwargs
            )
            
        else:
            # No graph encodings, just use text
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs
            )
        
        return outputs
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        graph_encodings: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Generate text with graph encodings.
        """
        batch_size = input_ids.size(0)
        
        if graph_encodings is not None:
            # Get text embeddings
            text_embeds = self.base_model.get_input_embeddings()(input_ids)
            
            # Project graph encodings
            graph_projected = self.graph_projector(graph_encodings)
            graph_embeds = graph_projected.view(
                batch_size, self.num_graph_tokens, self.embed_dim
            )
            
            # Add separator tokens
            separator_embeds = self.separator_embeddings.unsqueeze(0).expand(
                batch_size, -1, -1
            )
            
            # Concatenate
            combined_embeds = torch.cat([
                graph_embeds,
                separator_embeds,
                text_embeds
            ], dim=1)
            
            # Extend attention mask
            prefix_mask = torch.ones(
                batch_size,
                self.num_graph_tokens + self.num_separator_tokens,
                dtype=attention_mask.dtype,
                device=attention_mask.device
            )
            combined_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            
            # Generate
            outputs = self.base_model.generate(
                inputs_embeds=combined_embeds,
                attention_mask=combined_attention_mask,
                **kwargs
            )
            
        else:
            outputs = self.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )
        
        return outputs
    
    def save_pretrained(self, save_directory, **kwargs):
        """Save the model."""
        # Save base model
        self.base_model.save_pretrained(save_directory, **kwargs)
        
        # Save graph projector and separator embeddings
        graph_state = {
            'graph_projector': self.graph_projector.state_dict(),
            'separator_embeddings': self.separator_embeddings,
            'config': {
                'graph_encoding_dim': self.graph_encoding_dim,
                'num_graph_tokens': self.num_graph_tokens,
                'num_separator_tokens': self.num_separator_tokens,
            }
        }
        torch.save(graph_state, f"{save_directory}/graph_components.pt")
    
    @classmethod
    def from_pretrained(cls, model_path, graph_encoding_dim=256, **kwargs):
        """Load the model."""
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        
        # Try to load graph components
        graph_components_path = f"{model_path}/graph_components.pt"
        import os
        if os.path.exists(graph_components_path):
            graph_state = torch.load(graph_components_path)
            config = graph_state['config']
            
            # Create wrapper
            model = cls(
                base_model=base_model,
                graph_encoding_dim=config['graph_encoding_dim'],
                num_graph_tokens=config['num_graph_tokens'],
                num_separator_tokens=config['num_separator_tokens'],
            )
            
            # Load graph components
            model.graph_projector.load_state_dict(graph_state['graph_projector'])
            model.separator_embeddings.data = graph_state['separator_embeddings']
            
        else:
            # Create new wrapper
            model = cls(
                base_model=base_model,
                graph_encoding_dim=graph_encoding_dim,
            )
        
        return model
