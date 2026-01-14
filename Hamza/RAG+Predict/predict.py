"""
Main prediction script using RAG (Retrieval-Augmented Generation).
Encodes query graphs, retrieves similar graphs, and generates descriptions.
"""

import os
import sys
import torch
import pickle
import numpy as np
from tqdm import tqdm
from torch_geometric.loader import DataLoader

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from models.graph_encoder import MoleculeGINE
from retrieval import GraphRetriever, create_rag_prompt
import config


def load_embeddings(embeddings_path):
    """Load precomputed training embeddings."""
    print(f"Loading embeddings from {embeddings_path}")
    with open(embeddings_path, 'rb') as f:
        data = pickle.load(f)
    return data['embeddings'], data['descriptions'], data['graph_ids']


def load_encoder(checkpoint_path, node_dim, edge_dim):
    """Load trained graph encoder."""
    print(f"Loading encoder from {checkpoint_path}")
    
    # Initialize encoder
    encoder = MoleculeGINE(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden=config.GRAPH_HIDDEN_DIM,
        layers=config.GRAPH_LAYERS,
        out_dim=config.GRAPH_HIDDEN_DIM
    ).to(config.DEVICE)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    encoder.load_state_dict(checkpoint['model_state_dict'])
    encoder.eval()
    
    print("Encoder loaded successfully")
    return encoder


def predict_with_rag(test_graphs_path, encoder, retriever, output_path, batch_size=8):
    """
    Generate predictions using RAG approach.
    
    Args:
        test_graphs_path: Path to test graphs
        encoder: Trained graph encoder
        retriever: GraphRetriever instance
        output_path: Path to save predictions
        batch_size: Batch size for processing
    """
    # Load test graphs
    print(f"\nLoading test graphs from {test_graphs_path}")
    with open(test_graphs_path, 'rb') as f:
        test_data = pickle.load(f)
    
    test_graphs = test_data if isinstance(test_data, list) else test_data.get('graphs', [])
    print(f"Loaded {len(test_graphs)} test graphs")
    
    # Create dataloader
    loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)
    
    predictions = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Generating predictions")):
            batch = batch.to(config.DEVICE)
            
            # 1. Encode query graphs
            query_embeddings = encoder(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch
            ).cpu().numpy()
            
            # 2. Retrieve similar graphs for each query
            batch_results = retriever.batch_retrieve(query_embeddings, k=config.K_NEIGHBORS)
            
            # 3. Generate descriptions with RAG context
            for i, (retrieved_descs, scores, retrieved_ids) in enumerate(batch_results):
                # Create RAG prompt with retrieved context
                rag_prompt = create_rag_prompt(
                    "Generate a detailed chemical description of the molecule:",
                    retrieved_descs,
                    scores
                )
                
                # Option 1: Use retrieved descriptions directly (simple baseline)
                # For now, we'll use the most similar description as a starting point
                # In a full implementation, you'd feed this to the LLM
                
                # Option 2: Generate with the model using RAG context
                # This would require modifying the model to accept context
                # For now, we'll just use the top retrieved description
                prediction = retrieved_descs[0]  # Use most similar as prediction
                
                predictions.append({
                    'graph_idx': batch_idx * batch_size + i,
                    'prediction': prediction,
                    'retrieved_descriptions': retrieved_descs,
                    'retrieval_scores': scores,
                    'retrieved_ids': retrieved_ids,
                    'rag_prompt': rag_prompt
                })
    
    # Save predictions
    print(f"\nSaving predictions to {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(predictions, f)
    
    # Also save as CSV for easy viewing
    csv_path = output_path.replace('.pkl', '.csv')
    import pandas as pd
    df = pd.DataFrame([
        {
            'graph_idx': p['graph_idx'],
            'prediction': p['prediction']
        }
        for p in predictions
    ])
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV predictions to {csv_path}")
    
    return predictions


def main():
    print("="*60)
    print("RAG-Based Molecular Graph Description Prediction")
    print("="*60)
    
    # Load precomputed training embeddings
    embeddings_path = os.path.join(config.EMBEDDINGS_DIR, "train_embeddings.pkl")
    train_embeddings, train_descriptions, train_ids = load_embeddings(embeddings_path)
    
    # Initialize retriever
    retriever = GraphRetriever(
        train_embeddings,
        train_descriptions,
        train_ids,
        metric=config.DISTANCE_METRIC
    )
    
    # Load test graphs to get dimensions
    with open(config.TEST_GRAPHS, 'rb') as f:
        test_data = pickle.load(f)
    
    sample_graph = (test_data[0] if isinstance(test_data, list) else test_data['graphs'][0])
    node_dim = sample_graph.x.size(1)
    edge_dim = sample_graph.edge_attr.size(1)
    
    # Load trained encoder
    encoder = load_encoder(config.GRAPH_ENCODER_CHECKPOINT, node_dim, edge_dim)
    
    # Generate predictions
    output_path = os.path.join(config.PREDICTIONS_DIR, "rag_predictions.pkl")
    predictions = predict_with_rag(
        config.TEST_GRAPHS,
        encoder,
        retriever,
        output_path,
        batch_size=8
    )
    
    # Print some examples
    print("\n" + "="*60)
    print("Sample Predictions:")
    print("="*60)
    for i, pred in enumerate(predictions[:3]):
        print(f"\nGraph {pred['graph_idx']}:")
        print(f"Prediction: {pred['prediction'][:200]}...")
        print(f"\nTop 3 retrieved similar molecules (scores):")
        for j, (score, desc) in enumerate(zip(pred['retrieval_scores'][:3], 
                                               pred['retrieved_descriptions'][:3])):
            print(f"  {j+1}. Score: {score:.3f} - {desc[:100]}...")
    
    print("\n" + "="*60)
    print("Prediction completed!")
    print(f"Total predictions: {len(predictions)}")
    print("="*60)


if __name__ == "__main__":
    main()
