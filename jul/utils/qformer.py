"""Q-Former: Query Transformer for bridging graph and text modalities."""

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel


class QFormer(nn.Module):
    """
    Query Transformer (Q-Former) inspired by BLIP-2.
    Uses learnable query tokens to extract relevant information from graph embeddings.
    """
    
    def __init__(self, num_queries=32, hidden_dim=768, num_layers=6, num_heads=8):
        super().__init__()
        
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        
        # Learnable query tokens
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, hidden_dim))
        
        # BERT-style transformer for queries
        config = BertConfig(
            hidden_size=hidden_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_dim * 4,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=2048,  # Increased to handle larger graphs
        )
        self.qformer = BertModel(config, add_pooling_layer=False)
        
        # Project graph node features to Q-Former dimension
        self.graph_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, graph_node_features, graph_mask=None):
        """
        Args:
            graph_node_features: [batch_size, num_nodes, hidden_dim]
            graph_mask: [batch_size, num_nodes] attention mask for graph nodes
        
        Returns:
            query_output: [batch_size, num_queries, hidden_dim]
        """
        batch_size = graph_node_features.size(0)
        
        # Project graph features
        graph_features = self.graph_proj(graph_node_features)
        
        # Expand query tokens for batch
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        
        # Concatenate queries and graph features
        # Queries will attend to graph features via cross-attention
        inputs_embeds = torch.cat([query_tokens, graph_features], dim=1)
        
        # Create attention mask
        query_mask = torch.ones(batch_size, self.num_queries, device=query_tokens.device)
        if graph_mask is None:
            graph_mask = torch.ones(batch_size, graph_node_features.size(1), device=query_tokens.device)
        attention_mask = torch.cat([query_mask, graph_mask], dim=1)
        
        # Pass through Q-Former
        outputs = self.qformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Extract query outputs (first num_queries tokens)
        query_output = outputs.last_hidden_state[:, :self.num_queries, :]
        
        return query_output
