"""
Complete RAG pipeline:
1. Train graph encoder
2. Encode all graphs
3. Retrieve 5 closest descriptions for each molecule
4. Prepare data for LLM fine-tuning with graph encodings
"""

import os
import sys
import pickle
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import config
from retrieval import GraphRetriever
from encode_graphs import encode_all_graphs


def train_encoder(epochs=50, batch_size=64, lr=1e-3):
    """Step 1: Train the graph encoder."""
    print("\n" + "="*80)
    print("STEP 1: Training Graph Encoder")
    print("="*80)
    
    # Import training function
    from train_encoder import main as train_main
    
    # Run training
    train_main(
        graph_path=config.TRAIN_GRAPHS,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        save_dir=config.EMBEDDINGS_DIR,
    )
    
    print("\n✓ Encoder training complete!")


def encode_graphs():
    """Step 2: Encode all train and test graphs."""
    print("\n" + "="*80)
    print("STEP 2: Encoding Graphs")
    print("="*80)
    
    # Load the trained encoder
    checkpoint_path = os.path.join(config.EMBEDDINGS_DIR, "encoder_checkpoints/best_encoder.pt")
    
    # Encode train graphs
    print("\nEncoding training graphs...")
    train_embeddings, train_descriptions, train_ids = encode_all_graphs(
        graph_path=config.TRAIN_GRAPHS,
        checkpoint_path=checkpoint_path,
        output_path=os.path.join(config.EMBEDDINGS_DIR, "train_embeddings.pkl"),
        batch_size=128,
    )
    
    # Encode test graphs
    print("\nEncoding test graphs...")
    test_embeddings, test_descriptions, test_ids = encode_all_graphs(
        graph_path=config.TEST_GRAPHS,
        checkpoint_path=checkpoint_path,
        output_path=os.path.join(config.EMBEDDINGS_DIR, "test_embeddings.pkl"),
        batch_size=128,
    )
    
    print("\n✓ Graph encoding complete!")
    return train_embeddings, train_descriptions, train_ids, test_embeddings, test_descriptions, test_ids


def retrieve_descriptions(train_embeddings, train_descriptions, train_ids,
                          test_embeddings, test_descriptions, test_ids, k=5):
    """Step 3: Retrieve k closest descriptions for each molecule."""
    print("\n" + "="*80)
    print(f"STEP 3: Retrieving {k} Closest Descriptions")
    print("="*80)
    
    # Initialize retriever with training data
    retriever = GraphRetriever(
        embeddings=train_embeddings,
        descriptions=train_descriptions,
        graph_ids=train_ids,
        metric=config.DISTANCE_METRIC,
    )
    
    # Retrieve for training graphs (self-retrieval, will exclude self)
    print("\nRetrieving for training set...")
    train_retrievals = []
    for i, (emb, desc, gid) in enumerate(tqdm(zip(train_embeddings, train_descriptions, train_ids), 
                                                total=len(train_ids))):
        retrieved_descs, scores, retrieved_ids = retriever.retrieve(emb, k=k+1)  # +1 to exclude self
        
        # Exclude self (first result should be self)
        retrieved_descs = [d for j, d in enumerate(retrieved_descs) if retrieved_ids[j] != gid][:k]
        retrieved_scores = [s for j, s in enumerate(scores) if retrieved_ids[j] != gid][:k]
        retrieved_ids_filtered = [rid for rid in retrieved_ids if rid != gid][:k]
        
        train_retrievals.append({
            'graph_id': gid,
            'description': desc,
            'retrieved_descriptions': retrieved_descs,
            'retrieved_scores': retrieved_scores,
            'retrieved_ids': retrieved_ids_filtered,
        })
    
    # Retrieve for test graphs (only k=1, closest neighbor)
    print("\nRetrieving for test set (k=1, closest neighbor only)...")
    test_retrievals = []
    for i, (emb, desc, gid) in enumerate(tqdm(zip(test_embeddings, test_descriptions, test_ids),
                                               total=len(test_ids))):
        retrieved_descs, scores, retrieved_ids = retriever.retrieve(emb, k=1)
        
        test_retrievals.append({
            'graph_id': gid,
            'description': desc,  # May be empty for test
            'retrieved_descriptions': retrieved_descs,
            'retrieved_scores': scores,
            'retrieved_ids': retrieved_ids,
        })
    
    # Save retrievals
    train_retrieval_path = os.path.join(config.EMBEDDINGS_DIR, "train_retrievals.pkl")
    test_retrieval_path = os.path.join(config.EMBEDDINGS_DIR, "test_retrievals.pkl")
    
    with open(train_retrieval_path, 'wb') as f:
        pickle.dump(train_retrievals, f)
    
    with open(test_retrieval_path, 'wb') as f:
        pickle.dump(test_retrievals, f)
    
    print(f"\n✓ Retrieval complete!")
    print(f"  Train retrievals saved to: {train_retrieval_path}")
    print(f"  Test retrievals saved to: {test_retrieval_path}")
    
    return train_retrievals, test_retrievals


def prepare_llm_data(train_retrievals, test_retrievals, train_embeddings, test_embeddings):
    """Step 4: Prepare data for LLM fine-tuning with graph encodings and retrieved descriptions."""
    print("\n" + "="*80)
    print("STEP 4: Preparing LLM Training Data")
    print("="*80)
    
    # Prepare training data with graph encodings and retrieved descriptions
    train_data = []
    for i, retrieval in enumerate(train_retrievals):
        row = {
            'ID': retrieval['graph_id'],
            'graph_encoding': train_embeddings[i].tolist(),  # Save as list for CSV/JSON
            'Description': retrieval['description'],
            'retrieved_desc_1': retrieval['retrieved_descriptions'][0] if len(retrieval['retrieved_descriptions']) > 0 else "",
            'retrieved_desc_2': retrieval['retrieved_descriptions'][1] if len(retrieval['retrieved_descriptions']) > 1 else "",
            'retrieved_desc_3': retrieval['retrieved_descriptions'][2] if len(retrieval['retrieved_descriptions']) > 2 else "",
            'retrieved_desc_4': retrieval['retrieved_descriptions'][3] if len(retrieval['retrieved_descriptions']) > 3 else "",
            'retrieved_desc_5': retrieval['retrieved_descriptions'][4] if len(retrieval['retrieved_descriptions']) > 4 else "",
        }
        train_data.append(row)
    
    # Prepare test data (only 1 retrieved description - the best match)
    test_data = []
    for i, retrieval in enumerate(test_retrievals):
        row = {
            'ID': retrieval['graph_id'],
            'graph_encoding': test_embeddings[i].tolist(),
            'retrieved_desc_best': retrieval['retrieved_descriptions'][0] if len(retrieval['retrieved_descriptions']) > 0 else "",
            'retrieved_score_best': retrieval['retrieved_scores'][0] if len(retrieval['retrieved_scores']) > 0 else 0.0,
            'retrieved_id_best': retrieval['retrieved_ids'][0] if len(retrieval['retrieved_ids']) > 0 else -1,
        }
        test_data.append(row)
    
    # Save as pickle (preserves numpy arrays better)
    train_output_path = os.path.join(config.EMBEDDINGS_DIR, "train_llm_data.pkl")
    test_output_path = os.path.join(config.EMBEDDINGS_DIR, "test_llm_data.pkl")
    
    with open(train_output_path, 'wb') as f:
        pickle.dump(train_data, f)
    
    with open(test_output_path, 'wb') as f:
        pickle.dump(test_data, f)
    
    print(f"\n✓ LLM data preparation complete!")
    print(f"  Train data saved to: {train_output_path}")
    print(f"  Test data saved to: {test_output_path}")
    
    return train_data, test_data


def main():
    """Run the complete RAG pipeline."""
    print("\n" + "="*80)
    print("RAG PIPELINE FOR MOLECULAR GRAPH CAPTIONING")
    print("="*80)
    
    # Step 1: Train encoder (can skip if already trained)
    if not os.path.exists(os.path.join(config.EMBEDDINGS_DIR, "encoder_checkpoints/best_encoder.pt")):
        train_encoder(epochs=50, batch_size=64, lr=1e-3)
    else:
        print("\n✓ Encoder checkpoint found. Skipping training.")
    
    # Step 2: Encode graphs
    train_embeddings, train_descriptions, train_ids, test_embeddings, test_descriptions, test_ids = encode_graphs()
    
    # Step 3: Retrieve descriptions
    train_retrievals, test_retrievals = retrieve_descriptions(
        train_embeddings, train_descriptions, train_ids,
        test_embeddings, test_descriptions, test_ids,
        k=5
    )
    
    # Step 4: Prepare LLM data
    train_data, test_data = prepare_llm_data(
        train_retrievals, test_retrievals,
        train_embeddings, test_embeddings
    )
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("1. Fine-tune the LLM using the modified finetune_qwen_with_rag.py")
    print("2. Run inference using the modified inference_qwen_with_rag.py")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
