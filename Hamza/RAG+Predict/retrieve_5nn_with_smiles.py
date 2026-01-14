"""
Retrieve 5 nearest neighbors for test dataset and save to CSV with SMILES.
Output: CSV with columns [ID, SMILES, NN1_desc, NN2_desc, NN3_desc, NN4_desc, NN5_desc]
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
sys.path.insert(0, os.path.dirname(parent_dir))

import config
from retrieval import GraphRetriever
from encode_graphs import encode_all_graphs
from Hamza.chem.pyg_to_smiles import pyg_data_to_smiles
from data_baseline.data_utils import x_map, e_map


def load_graphs_and_convert_to_smiles(graph_path):
    """Load graphs and convert to SMILES strings."""
    print(f"Loading graphs from {graph_path}")
    with open(graph_path, 'rb') as f:
        graphs = pickle.load(f)
    
    print("Converting graphs to SMILES...")
    smiles_list = []
    ids_list = []
    
    for graph in tqdm(graphs, desc="Converting to SMILES"):
        try:
            smiles = pyg_data_to_smiles(graph, x_map, e_map, canonical=True)
            smiles_list.append(smiles)
            ids_list.append(graph.id)
        except Exception as e:
            print(f"Warning: Failed to convert graph {graph.id} to SMILES: {e}")
            smiles_list.append("CONVERSION_FAILED")
            ids_list.append(graph.id)
    
    return smiles_list, ids_list


def main():
    print("="*80)
    print("RETRIEVE 5 NEAREST NEIGHBORS FOR TEST DATASET WITH SMILES")
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
    
    # Step 3: Convert test graphs to SMILES
    print("\n" + "-"*80)
    print("Step 3: Converting test graphs to SMILES")
    print("-"*80)
    
    test_smiles, test_smiles_ids = load_graphs_and_convert_to_smiles(config.TEST_GRAPHS)
    
    # Create mapping from ID to SMILES
    id_to_smiles = {str(id_): smiles for id_, smiles in zip(test_smiles_ids, test_smiles)}
    
    print(f"Converted {len(test_smiles)} test graphs to SMILES")
    
    # Step 4: Initialize retriever with train data
    print("\n" + "-"*80)
    print("Step 4: Retrieving 5 nearest neighbors for each test graph")
    print("-"*80)
    
    retriever = GraphRetriever(
        embeddings=train_embeddings,
        descriptions=train_descriptions,
        graph_ids=train_ids,
        metric=config.DISTANCE_METRIC,
    )
    
    # Retrieve k=5 nearest neighbors for each test graph
    test_results = []
    for i, (emb, test_id) in enumerate(tqdm(zip(test_embeddings, test_ids), 
                                             total=len(test_ids),
                                             desc="Retrieving 5-NN")):
        # Retrieve 5 closest neighbors
        retrieved_descs, scores, retrieved_ids = retriever.retrieve(emb, k=5)
        
        # Get SMILES for this test graph
        smiles = id_to_smiles.get(str(test_id), "UNKNOWN")
        
        result = {
            'ID': test_id,
            'SMILES': smiles,
            'NN1_desc': retrieved_descs[0] if len(retrieved_descs) > 0 else "",
            'NN2_desc': retrieved_descs[1] if len(retrieved_descs) > 1 else "",
            'NN3_desc': retrieved_descs[2] if len(retrieved_descs) > 2 else "",
            'NN4_desc': retrieved_descs[3] if len(retrieved_descs) > 3 else "",
            'NN5_desc': retrieved_descs[4] if len(retrieved_descs) > 4 else "",
        }
        
        # Also store scores and IDs for reference
        result['NN1_score'] = scores[0] if len(scores) > 0 else 0.0
        result['NN2_score'] = scores[1] if len(scores) > 1 else 0.0
        result['NN3_score'] = scores[2] if len(scores) > 2 else 0.0
        result['NN4_score'] = scores[3] if len(scores) > 3 else 0.0
        result['NN5_score'] = scores[4] if len(scores) > 4 else 0.0
        
        result['NN1_id'] = retrieved_ids[0] if len(retrieved_ids) > 0 else -1
        result['NN2_id'] = retrieved_ids[1] if len(retrieved_ids) > 1 else -1
        result['NN3_id'] = retrieved_ids[2] if len(retrieved_ids) > 2 else -1
        result['NN4_id'] = retrieved_ids[3] if len(retrieved_ids) > 3 else -1
        result['NN5_id'] = retrieved_ids[4] if len(retrieved_ids) > 4 else -1
        
        test_results.append(result)
    
    # Step 5: Save to CSV
    print("\n" + "-"*80)
    print("Step 5: Saving results to CSV")
    print("-"*80)
    
    # Create DataFrame
    df = pd.DataFrame(test_results)
    
    # Save with all info (for reference)
    full_output_path = os.path.join(config.PREDICTIONS_DIR, "test_5nn_full.csv")
    df.to_csv(full_output_path, index=False)
    print(f"Full results saved to: {full_output_path}")
    
    # Save with only ID, SMILES, and 5 neighbor descriptions (as requested)
    simple_columns = ['ID', 'SMILES', 'NN1_desc', 'NN2_desc', 'NN3_desc', 'NN4_desc', 'NN5_desc']
    simple_output_path = os.path.join(config.PREDICTIONS_DIR, "test_5nn.csv")
    df[simple_columns].to_csv(simple_output_path, index=False)
    print(f"Simple results (ID, SMILES, 5-NN descriptions) saved to: {simple_output_path}")
    
    # Print statistics
    print("\n" + "="*80)
    print("RETRIEVAL COMPLETE!")
    print("="*80)
    print(f"Total test samples: {len(test_results)}")
    print(f"Average similarity score (NN1): {df['NN1_score'].mean():.4f}")
    print(f"Min similarity score (NN1): {df['NN1_score'].min():.4f}")
    print(f"Max similarity score (NN1): {df['NN1_score'].max():.4f}")
    print("\nOutput files:")
    print(f"  1. {simple_output_path}")
    print(f"  2. {full_output_path}")
    print("="*80)


if __name__ == "__main__":
    main()
