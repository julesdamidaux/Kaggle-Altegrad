"""
Encode all training graphs and store their embeddings for retrieval.
This script loads a trained graph encoder and generates embeddings for all training graphs.
"""

import os
import sys
import torch
import pickle
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from models.graph_encoder import MoleculeGINE
import config


def load_graph_encoder(checkpoint_path, node_dim, edge_dim):
    """Load trained graph encoder from checkpoint."""
    print(f"Loading graph encoder from {checkpoint_path}")
    
    encoder = MoleculeGINE(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden=config.GRAPH_HIDDEN_DIM,
        layers=config.GRAPH_LAYERS,
        out_dim=config.GRAPH_HIDDEN_DIM
    ).to(config.DEVICE)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    
    # Load model state dict
    encoder.load_state_dict(checkpoint['model_state_dict'])
    encoder.eval()
    
    print("Graph encoder loaded successfully")
    return encoder


def encode_all_graphs(graph_path, checkpoint_path, output_path=None, batch_size=32):
    """
    Encode all graphs from a pickle file using the trained encoder.
    
    Args:
        graph_path: Path to pickle file containing graphs
        checkpoint_path: Path to encoder checkpoint
        output_path: Optional path to save embeddings
        batch_size: Batch size for encoding
    
    Returns:
        embeddings: Numpy array of graph embeddings [num_graphs, hidden_dim]
        descriptions: List of corresponding descriptions
        graph_ids: List of graph IDs
    """
    # Load graphs
    print(f"Loading graphs from {graph_path}")
    with open(graph_path, 'rb') as f:
        graphs_data = pickle.load(f)
    
    if isinstance(graphs_data, dict):
        graphs = graphs_data.get('graphs', [])
        descriptions = graphs_data.get('descriptions', [getattr(g, 'description', '') for g in graphs])
    else:
        graphs = graphs_data
        descriptions = [getattr(g, 'description', '') for g in graphs]
    
    print(f"Loaded {len(graphs)} graphs")
    
    # Get dimensions from first graph
    sample_graph = graphs[0]
    node_dim = sample_graph.x.size(1)
    edge_dim = sample_graph.edge_attr.size(1)
    
    print(f"Node feature dim: {node_dim}")
    print(f"Edge feature dim: {edge_dim}")
    
    # Load encoder
    encoder = load_graph_encoder(checkpoint_path, node_dim, edge_dim)
    
    # Create dataloader
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    
    all_embeddings = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Encoding graphs"):
            batch = batch.to(config.DEVICE)
            
            # Get graph embeddings
            embeddings = encoder(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch
            )  # [batch_size, hidden_dim]
            all_embeddings.append(embeddings.cpu().numpy())
    
    # Concatenate all embeddings
    embeddings = np.vstack(all_embeddings)
    
    print(f"Encoded {len(embeddings)} graphs with embedding dim {embeddings.shape[1]}")
    
    # Create graph IDs
    graph_ids = [getattr(g, 'id', idx) for idx, g in enumerate(graphs)]
    
    # Save if output path provided
    if output_path:
        save_embeddings(embeddings, descriptions, graph_ids, output_path)
    
    return embeddings, descriptions, graph_ids


def save_embeddings(embeddings, descriptions, graph_ids, output_path):
    """Save embeddings and metadata."""
    data = {
        'embeddings': embeddings,
        'descriptions': descriptions,
        'graph_ids': graph_ids
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Saved embeddings to {output_path}")


def main():
    print("="*60)
    print("Graph Encoding for RAG System")
    print("="*60)
    
    # Encode all graphs
    embeddings, descriptions, graph_ids = encode_all_graphs(
        graph_path=config.TRAIN_GRAPHS,
        checkpoint_path=config.GRAPH_ENCODER_CHECKPOINT,
        output_path=os.path.join(config.EMBEDDINGS_DIR, "train_embeddings.pkl"),
        batch_size=32
    )
    
    print("\n" + "="*60)
    print("Encoding completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Number of descriptions: {len(descriptions)}")
    print("="*60)


if __name__ == "__main__":
    main()
