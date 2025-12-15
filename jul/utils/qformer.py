"""Q-Former: Query Transformer for bridging graph and text modalities."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel


class QFormer(nn.Module):
    """
    Query Transformer (Q-Former) inspired by BLIP-2.
    Supports three training objectives:
    - ITC: Image-Text Contrastive (unimodal self-attention)
    - ITM: Image-Text Matching (bidirectional self-attention)
    - ITG: Image-Grounded Text Generation (multimodal causal self-attention)
    """
    
    def __init__(self, num_queries=32, hidden_dim=768, num_layers=6, num_heads=8):
        super().__init__()
        
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        
        # Learnable query tokens
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, hidden_dim))
        
        # BERT-style transformer for queries and text
        config = BertConfig(
            hidden_size=hidden_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_dim * 4,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=2048,
        )
        self.qformer = BertModel(config, add_pooling_layer=False)
        
        # Projection heads
        self.graph_proj = nn.Linear(hidden_dim, hidden_dim)
        self.text_proj = nn.Linear(hidden_dim, hidden_dim)  # For ITC
        
        # ITM head (binary classification: matched or not)
        self.itm_head = nn.Linear(hidden_dim, 2)
        
    def forward(
        self, 
        graph_node_features, 
        text_embeds=None,
        text_atts=None,
        graph_mask=None,
        mode='extract'  # 'extract', 'itc', 'itm', 'itg'
    ):
        """
        Args:
            graph_node_features: [B, num_nodes, hidden_dim]
            text_embeds: [B, max_len, hidden_dim] - text token embeddings
            text_atts: [B, max_len] - text attention mask
            graph_mask: [B, num_nodes] - graph attention mask
        """
        batch_size = graph_node_features.size(0)
        
        # Project graph features
        graph_features = self.graph_proj(graph_node_features) 
        
        # Expand query tokens
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        
        # Create query attention mask
        query_mask = torch.ones(batch_size, self.num_queries, device=query_tokens.device, dtype=torch.long)
        if graph_mask is None:
            graph_mask = torch.ones(batch_size, graph_node_features.size(1), device=query_tokens.device, dtype=torch.long)
        
        if mode == 'extract':
            # Simple extraction: concatenate queries + graph, queries attend to everything
            inputs_embeds = torch.cat([query_tokens, graph_features], dim=1)
            attention_mask = torch.cat([query_mask, graph_mask], dim=1)
            
            outputs = self.qformer(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True
            )
            # Return only the query outputs
            return outputs.last_hidden_state[:, :self.num_queries, :]
        
        elif mode == 'itc':
            # Unimodal self-attention: queries and text DON'T see each other
            # Process queries with graph
            query_graph_embeds = torch.cat([query_tokens, graph_features], dim=1)
            query_graph_mask = torch.cat([query_mask, graph_mask], dim=1)
            
            query_outputs = self.qformer(
                inputs_embeds=query_graph_embeds,
                attention_mask=query_graph_mask,
                return_dict=True
            )
            query_feats = query_outputs.last_hidden_state[:, :self.num_queries, :]  # [B, num_queries, D]
            
            # Text encodes itself (no graph)
            text_outputs = self.qformer(
                inputs_embeds=text_embeds,
                attention_mask=text_atts,
                return_dict=True
            )
            text_feat = text_outputs.last_hidden_state[:, 0, :]  # [CLS] token
            
            # Project for contrastive learning
            text_feat = F.normalize(self.text_proj(text_feat), dim=-1)
            
            # For queries, take mean and normalize
            query_feat = F.normalize(query_feats.mean(dim=1), dim=-1)
            
            return query_feat, text_feat
        
        elif mode == 'itm':
            # Bidirectional self-attention: queries, graph, and text ALL interact
            # Concatenate queries + graph + text
            inputs_embeds = torch.cat([query_tokens, graph_features, text_embeds], dim=1)
            attention_mask = torch.cat([query_mask, graph_mask, text_atts], dim=1)
            
            outputs = self.qformer(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            # Use query outputs for classification
            query_output = outputs.last_hidden_state[:, :self.num_queries, :]
            
            # Average pool queries and classify
            pooled = query_output.mean(dim=1)  # [B, D]
            itm_logits = self.itm_head(pooled)  # [B, 2]
            
            return itm_logits
        
        elif mode == 'itg':
            # ----- Multimodal causal self-attention -----
            # Concatenate queries + graph + text
            inputs_embeds = torch.cat([query_tokens, graph_features, text_embeds], dim=1)
            # 1D mask for which positions are "real" tokens
            base_mask_1d = torch.cat([query_mask, graph_mask, text_atts], dim=1)  # [B, L]
            B, L = base_mask_1d.size()
            
            num_queries = self.num_queries
            num_graph = graph_features.size(1)
            text_start = num_queries + num_graph  # index of first text token
            
            # Start from outer product to mask out padded positions
            # keep[i,j] = 1 only if both positions are valid (non-pad)
            keep = base_mask_1d
            attn_mask_3d = (keep[:, :, None] & keep[:, None, :]).long()  # [B, L, L]
            
            # Text-text causal mask (lower triangular) applied only to text block
            causal = torch.tril(torch.ones(L, L, device=inputs_embeds.device, dtype=torch.long))
            attn_mask_3d[:, text_start:, text_start:] = attn_mask_3d[:, text_start:, text_start:] * causal[text_start:, text_start:]
            
            # Queries + graph should NOT see any text tokens
            # (they only attend to queries + graph)
            if text_start < L:
                attn_mask_3d[:, :text_start, text_start:] = 0
            
            # Pass 3D mask to BERT: shape (B, from_seq_len, to_seq_len)
            outputs = self.qformer(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_mask_3d,
                return_dict=True
            )
            
            # Return text features (skip queries and graph)
            text_start_idx = text_start
            text_feats = outputs.last_hidden_state[:, text_start_idx:, :]
            return text_feats
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
