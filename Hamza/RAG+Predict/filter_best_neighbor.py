"""
Filter RAG retrieval results to keep only the best (top-1) neighbor.
This creates simplified data files with only the most similar molecule description.
"""

import os
import sys
import pickle
import pandas as pd
from pathlib import Path

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import config


def filter_to_best_neighbor(input_retrieval_path, output_retrieval_path, 
                            input_llm_path, output_llm_path):
    """
    Filter retrieval results to keep only the best (top-1) neighbor.
    
    Args:
        input_retrieval_path: Path to original retrieval pickle with k=5 neighbors
        output_retrieval_path: Path to save filtered retrieval with k=1 neighbor
        input_llm_path: Path to original LLM data with 5 retrieved descriptions
        output_llm_path: Path to save filtered LLM data with 1 retrieved description
    """
    print(f"\nFiltering retrievals from {input_retrieval_path}")
    
    # Load original retrievals
    with open(input_retrieval_path, 'rb') as f:
        retrievals = pickle.load(f)
    
    # Filter to best neighbor only
    filtered_retrievals = []
    for retrieval in retrievals:
        filtered = {
            'graph_id': retrieval['graph_id'],
            'description': retrieval['description'],
            'retrieved_descriptions': [retrieval['retrieved_descriptions'][0]] if retrieval['retrieved_descriptions'] else [],
            'retrieved_scores': [retrieval['retrieved_scores'][0]] if retrieval['retrieved_scores'] else [],
            'retrieved_ids': [retrieval['retrieved_ids'][0]] if retrieval['retrieved_ids'] else [],
        }
        filtered_retrievals.append(filtered)
    
    # Save filtered retrievals
    with open(output_retrieval_path, 'wb') as f:
        pickle.dump(filtered_retrievals, f)
    print(f"  Saved filtered retrievals to: {output_retrieval_path}")
    
    # Load original LLM data
    print(f"\nFiltering LLM data from {input_llm_path}")
    with open(input_llm_path, 'rb') as f:
        llm_data = pickle.load(f)
    
    # Filter to best neighbor only
    filtered_llm_data = []
    for item in llm_data:
        filtered_item = {
            'ID': item['ID'],
            'graph_encoding': item['graph_encoding'],
            'retrieved_desc_1': item['retrieved_desc_1'],  # Keep only the best one
        }
        
        # Add ground truth description if present (for training data)
        if 'Description' in item:
            filtered_item['Description'] = item['Description']
        
        filtered_llm_data.append(filtered_item)
    
    # Save filtered LLM data
    with open(output_llm_path, 'wb') as f:
        pickle.dump(filtered_llm_data, f)
    print(f"  Saved filtered LLM data to: {output_llm_path}")
    
    return filtered_retrievals, filtered_llm_data


def main():
    """Filter both train and test retrieval results to best neighbor only."""
    print("="*80)
    print("FILTERING RAG RESULTS TO BEST NEIGHBOR ONLY")
    print("="*80)
    
    embeddings_dir = config.EMBEDDINGS_DIR
    
    # Define paths
    train_retrieval_input = os.path.join(embeddings_dir, "train_retrievals.pkl")
    train_retrieval_output = os.path.join(embeddings_dir, "train_retrievals_best.pkl")
    train_llm_input = os.path.join(embeddings_dir, "train_llm_data.pkl")
    train_llm_output = os.path.join(embeddings_dir, "train_llm_data_best.pkl")
    
    test_retrieval_input = os.path.join(embeddings_dir, "test_retrievals.pkl")
    test_retrieval_output = os.path.join(embeddings_dir, "test_retrievals_best.pkl")
    test_llm_input = os.path.join(embeddings_dir, "test_llm_data.pkl")
    test_llm_output = os.path.join(embeddings_dir, "test_llm_data_best.pkl")
    
    # Check if input files exist
    if not os.path.exists(train_retrieval_input):
        print(f"\n❌ Error: {train_retrieval_input} not found!")
        print("Please run the complete RAG pipeline first.")
        return
    
    # Filter training data
    print("\n" + "="*80)
    print("FILTERING TRAINING DATA")
    print("="*80)
    train_filtered_retrievals, train_filtered_llm = filter_to_best_neighbor(
        train_retrieval_input, train_retrieval_output,
        train_llm_input, train_llm_output
    )
    print(f"\n✓ Filtered {len(train_filtered_retrievals)} training examples")
    
    # Filter test data
    print("\n" + "="*80)
    print("FILTERING TEST DATA")
    print("="*80)
    test_filtered_retrievals, test_filtered_llm = filter_to_best_neighbor(
        test_retrieval_input, test_retrieval_output,
        test_llm_input, test_llm_output
    )
    print(f"\n✓ Filtered {len(test_filtered_retrievals)} test examples")
    
    # Summary
    print("\n" + "="*80)
    print("FILTERING COMPLETE!")
    print("="*80)
    print("\nOutput files created:")
    print(f"  1. {train_retrieval_output}")
    print(f"  2. {train_llm_output}")
    print(f"  3. {test_retrieval_output}")
    print(f"  4. {test_llm_output}")
    print("\nThese files contain only the best (most similar) neighbor for each molecule.")
    print("="*80)


if __name__ == "__main__":
    main()
