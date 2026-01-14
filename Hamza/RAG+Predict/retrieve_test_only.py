"""
Retrieve only the best neighbor for test dataset and save to CSV.
Output: CSV with columns [ID, description]
"""

import os
import sys
import pickle
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import config
from retrieval import GraphRetriever
from encode_graphs import encode_all_graphs


def main():
    print("="*80)
    print("RETRIEVE BEST NEIGHBOR FOR TEST DATASET")
    print("="*80)
    
    # Check if encoder exists
    checkpoint_path = os.path.join(config.EMBEDDINGS_DIR, "encoder_checkpoints/best_encoder.pt")
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Encoder checkpoint not found at {checkpoint_path}")
        print("Please train the encoder first using train_encoder.py or complete_rag_pipeline.py")
        return
    
    # Step 1: Encode train graphs (for retrieval database)
    print("\n" + "-"*80)
    print("Step 1: Encoding training graphs (retrieval database)")
    print("-"*80)
    
    train_embeddings_path = os.path.join(config.EMBEDDINGS_DIR, "train_embeddings.pkl")
    if os.path.exists(train_embeddings_path):
        print(f"Loading cached embeddings from {train_embeddings_path}")
        with open(train_embeddings_path, 'rb') as f:
            data = pickle.load(f)
            train_embeddings = data['embeddings']
            train_descriptions = data['descriptions']
            train_ids = data['graph_ids']
    else:
        print("Computing train embeddings...")
        train_embeddings, train_descriptions, train_ids = encode_all_graphs(
            graph_path=config.TRAIN_GRAPHS,
            checkpoint_path=checkpoint_path,
            output_path=train_embeddings_path,
            batch_size=128,
        )
    
    print(f"Train database: {len(train_embeddings)} graphs")
    
    # Step 2: Encode test graphs
    print("\n" + "-"*80)
    print("Step 2: Encoding test graphs")
    print("-"*80)
    
    test_embeddings_path = os.path.join(config.EMBEDDINGS_DIR, "test_embeddings.pkl")
    if os.path.exists(test_embeddings_path):
        print(f"Loading cached embeddings from {test_embeddings_path}")
        with open(test_embeddings_path, 'rb') as f:
            data = pickle.load(f)
            test_embeddings = data['embeddings']
            test_descriptions = data['descriptions']
            test_ids = data['graph_ids']
    else:
        print("Computing test embeddings...")
        test_embeddings, test_descriptions, test_ids = encode_all_graphs(
            graph_path=config.TEST_GRAPHS,
            checkpoint_path=checkpoint_path,
            output_path=test_embeddings_path,
            batch_size=128,
        )
    
    print(f"Test set: {len(test_embeddings)} graphs")
    
    # Step 3: Initialize retriever with train data
    print("\n" + "-"*80)
    print("Step 3: Retrieving best neighbor for each test graph")
    print("-"*80)
    
    retriever = GraphRetriever(
        embeddings=train_embeddings,
        descriptions=train_descriptions,
        graph_ids=train_ids,
        metric=config.DISTANCE_METRIC,
    )
    
    # Retrieve only k=1 (best neighbor) for each test graph
    test_results = []
    for i, (emb, test_id) in enumerate(tqdm(zip(test_embeddings, test_ids), 
                                             total=len(test_ids),
                                             desc="Retrieving")):
        # Retrieve only 1 closest neighbor
        retrieved_descs, scores, retrieved_ids = retriever.retrieve(emb, k=1)
        
        test_results.append({
            'ID': test_id,
            'description': retrieved_descs[0],
            'similarity_score': scores[0],
            'retrieved_from_id': retrieved_ids[0]
        })
    
    # Step 4: Save to CSV with only ID and description columns
    print("\n" + "-"*80)
    print("Step 4: Saving results to CSV")
    print("-"*80)
    
    # Create DataFrame
    df = pd.DataFrame(test_results)
    
    # Save with all info (for reference)
    full_output_path = os.path.join(config.PREDICTIONS_DIR, "test_best_neighbor_full.csv")
    df.to_csv(full_output_path, index=False)
    print(f"Full results saved to: {full_output_path}")
    
    # Save with only ID and description (as requested)
    simple_output_path = os.path.join(config.PREDICTIONS_DIR, "test_best_neighbor.csv")
    df[['ID', 'description']].to_csv(simple_output_path, index=False)
    print(f"Simple results (ID, description) saved to: {simple_output_path}")
    
    # Print statistics
    print("\n" + "="*80)
    print("RETRIEVAL COMPLETE!")
    print("="*80)
    print(f"Total test samples: {len(test_results)}")
    print(f"Average similarity score: {df['similarity_score'].mean():.4f}")
    print(f"Min similarity score: {df['similarity_score'].min():.4f}")
    print(f"Max similarity score: {df['similarity_score'].max():.4f}")
    print("\nOutput files:")
    print(f"  1. {simple_output_path}")
    print(f"  2. {full_output_path}")
    print("="*80)


if __name__ == "__main__":
    main()
