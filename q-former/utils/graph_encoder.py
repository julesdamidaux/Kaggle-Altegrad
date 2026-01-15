"""Pretrained Graphormer encoder wrapper (FROZEN)."""

import torch
import torch.nn as nn
from transformers import GraphormerForGraphClassification
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch


class PretrainedGraphormerEncoder(nn.Module):
    """
    Wrapper for pretrained Graphormer model from HuggingFace.
    Model: clefourrier/graphormer-base-pcqm4mv2
    
    This encoder is FROZEN - we don't train it, only use it for feature extraction.
    """
    
    
    def __init__(self, model_name="clefourrier/graphormer-base-pcqm4mv2", hidden_dim=768, freeze=True):
        super().__init__()
        
        print(f"Loading pretrained Graphormer: {model_name}")
        # Load as Classification model to match checkpoint structure, then extract encoder
        full_model = GraphormerForGraphClassification.from_pretrained(model_name)
        self.graphormer = full_model.encoder
        
        if freeze:
            print("Freezing Graphormer parameters (not trainable)")
            for param in self.graphormer.parameters():
                param.requires_grad = False
            self.graphormer.eval()  # Set to eval mode
        else:
            print("Unfreezing Graphormer parameters (TRAINABLE)")
            self.graphormer.train() # Set to train mode
            # Enable gradients
            for param in self.graphormer.parameters():
                param.requires_grad = True
        
        self.hidden_dim = hidden_dim
        self.graphormer_hidden_dim = self.graphormer.config.hidden_size  # 768
        
        # Projection if dimensions don't match
        if self.graphormer_hidden_dim != hidden_dim:
            self.proj = nn.Linear(self.graphormer_hidden_dim, hidden_dim)
        else:
            self.proj = nn.Identity()
    
    def prepare_graphormer_inputs(self, batch):
        """
        Convert PyG batch to Graphormer input format.
        
        Graphormer expects:
        - input_nodes: [batch_size, num_nodes, node_feat_dim]
        - input_edges: [batch_size, num_nodes, num_nodes, edge_feat_dim]
        - attn_bias: [batch_size, num_nodes, num_nodes] (optional)
        - spatial_pos: [batch_size, num_nodes, num_nodes] (optional)
        """
        device = batch.x.device
        
        # Convert to dense format
        # Convert to dense format (Graphormer expects LongTensor for node features)
        node_features, node_mask = to_dense_batch(batch.x.long(), batch.batch)
        batch_size, max_num_nodes, node_feat_dim = node_features.shape
        
        # Create adjacency matrix (dense)
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr.float() if batch.edge_attr is not None else None
        
        # For each graph in batch, create dense adjacency
        adj_matrices = []
        edge_features = []
        
        for i in range(batch_size):
            # Get nodes for this graph
            mask = (batch.batch == i)
            num_nodes = mask.sum().item()
            
            # Get edges for this graph
            edge_mask = mask[edge_index[0]] & mask[edge_index[1]]
            graph_edge_index = edge_index[:, edge_mask]
            
            # Reindex to 0-based for this graph
            node_idx = torch.where(mask)[0]
            node_mapping = torch.zeros(batch.x.size(0), dtype=torch.long, device=device)
            node_mapping[node_idx] = torch.arange(num_nodes, device=device)
            
            graph_edge_index = node_mapping[graph_edge_index]
            
            # Create dense adjacency [max_num_nodes, max_num_nodes]
            adj = torch.zeros(max_num_nodes, max_num_nodes, device=device)
            if graph_edge_index.size(1) > 0:
                adj[graph_edge_index[0], graph_edge_index[1]] = 1.0
            
            adj_matrices.append(adj)
            
            # Edge features [max_num_nodes, max_num_nodes, edge_feat_dim]
            if edge_attr is not None:
                graph_edge_attr = edge_attr[edge_mask]
                edge_feat_dim = graph_edge_attr.size(1)
                edge_feat = torch.zeros(max_num_nodes, max_num_nodes, edge_feat_dim, device=device)
                if graph_edge_index.size(1) > 0:
                    edge_feat[graph_edge_index[0], graph_edge_index[1]] = graph_edge_attr
                edge_features.append(edge_feat)
        
        adj_matrices = torch.stack(adj_matrices)  # [batch_size, max_num_nodes, max_num_nodes]
        
        if edge_features:
            edge_features = torch.stack(edge_features)  # [batch_size, max_num_nodes, max_num_nodes, edge_feat_dim]
        else:
            edge_features = None
        
        # Prepare inputs for Graphormer
        inputs = {}
        inputs['input_nodes'] = node_features
        
        # attn_bias: [B, N+1, N+1] (Padded for virtual node)
        batch_size, max_num_nodes = adj_matrices.shape[:2]
        attn_bias_padded = torch.zeros(batch_size, max_num_nodes + 1, max_num_nodes + 1, device=device)
        attn_bias_padded[:, 1:, 1:] = adj_matrices
        inputs['attn_bias'] = attn_bias_padded
        
        # spatial_pos: [B, N, N] (Not padded, model handles virtual node logic)
        inputs['spatial_pos'] = torch.zeros(batch_size, max_num_nodes, max_num_nodes, dtype=torch.long, device=device)
        
        # in_degree and out_degree: [B, N] (Not padded)
        inputs['in_degree'] = adj_matrices.sum(dim=2).long()
        inputs['out_degree'] = adj_matrices.sum(dim=1).long()
        
        # Edge features (if any) as input_edges: [B, N, N, 1, E] (Padded for multi-hop)
        if edge_features is not None:
            inputs['input_edges'] = edge_features.long().unsqueeze(3)
        else:
            inputs['input_edges'] = torch.zeros(batch_size, max_num_nodes, max_num_nodes, 1, 1, dtype=torch.long, device=device)
            
        # Edge type for attention: [B, N, N] (Not padded)
        # Take first hop (index 0) and first feature (index 0)
        inputs['attn_edge_type'] = inputs['input_edges'][:, :, :, 0, 0].long()
        
        inputs['node_mask'] = node_mask
        
        return inputs
    
    # Removed @torch.no_grad() to allow training
    def forward(self, batch):
        """
        Forward pass through frozen Graphormer.
        
        Args:
            batch: PyG Batch object
        
        Returns:
            node_features: [batch_size, max_num_nodes + 1, hidden_dim]
            graph_features: [batch_size, hidden_dim]
            node_mask: [batch_size, max_num_nodes + 1]
        """
        # Prepare inputs for Graphormer
        inputs = self.prepare_graphormer_inputs(batch)
        
        # Call Graphormer with required positional arguments
        outputs = self.graphormer(
            input_nodes=inputs['input_nodes'],
            input_edges=inputs['input_edges'],
            attn_bias=inputs['attn_bias'],
            in_degree=inputs['in_degree'],
            out_degree=inputs['out_degree'],
            spatial_pos=inputs['spatial_pos'],
            attn_edge_type=inputs['attn_edge_type']
        )
        
        # Get node-level features [batch_size, num_nodes + 1, hidden_dim]
        # (Includes virtual node at index 0)
        node_features_dense = outputs.last_hidden_state
        
        # Get graph-level features (use first token as graph representation)
        graph_features = node_features_dense[:, 0, :]  # [batch_size, hidden_dim]
        
        # Project if needed
        graph_features = self.proj(graph_features)
        # Project node features as well (optional, keep same dim)
        node_features_dense = self.proj(node_features_dense)
        
        # Create mask for node features (including virtual node)
        node_mask = inputs['node_mask']  # [batch_size, num_nodes]
        batch_size = node_mask.size(0)
        virtual_node_mask = torch.ones(batch_size, 1, device=node_mask.device, dtype=node_mask.dtype)
        full_node_mask = torch.cat([virtual_node_mask, node_mask], dim=1)  # [batch_size, num_nodes + 1]
        
        return node_features_dense, graph_features, full_node_mask

