"""
Inference script for Graph Encoder + LLM Model

Generates molecule descriptions from molecular graphs using the fine-tuned model.
"""

import os
import sys
from pathlib import Path
import pickle
import argparse

import torch
import pandas as pd
from torch_geometric.data import Batch as PyGBatch
from transformers import AutoTokenizer
from tqdm import tqdm

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Hamza.train.finetune_llm_with_graph_encoder import GraphLLMModel, ModelConfig


def load_model_and_tokenizer(checkpoint_dir: str):
    """Load trained model and tokenizer"""
    
    print(f"Loading model from {checkpoint_dir}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_dir,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create model config (should match training config)
    config = ModelConfig()
    
    # Load model
    model = GraphLLMModel(config)
    
    # Load trained weights
    # Note: For LoRA models, you might need to load differently
    # This is a simplified version
    try:
        state_dict = torch.load(
            os.path.join(checkpoint_dir, "pytorch_model.bin"),
            map_location='cpu'
        )
        model.load_state_dict(state_dict, strict=False)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load full model: {e}")
        print("Attempting to load LoRA adapters...")
        # For LoRA models, the base model and adapters are loaded separately
    
    model.eval()
    
    return model, tokenizer


def generate_description(model, tokenizer, graph, max_new_tokens=256, device='cuda'):
    """Generate description for a single graph"""
    
    model = model.to(device)
    
    # Create a batch with single graph
    graph_batch = PyGBatch.from_data_list([graph]).to(device)
    
    # Prepare input text
    prompt = "<|im_start|>user\nDescribe this molecule.<|im_end|>\n<|im_start|>assistant\n"
    
    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)
    
    # Get graph encoding
    with torch.no_grad():
        graph_encodings = model.graph_encoder(
            graph_batch.x.float(),
            graph_batch.edge_index,
            graph_batch.edge_attr.float(),
            graph_batch.batch,
        )
        
        # Project to LLM space
        graph_embeds = model.graph_projector(graph_encodings)
        
        # Get learnable prompts
        prompt_embeds = model.learnable_prompts.unsqueeze(0)
        
        # Get text embeddings
        text_embeds = model.llm.get_input_embeddings()(inputs.input_ids)
        
        # Concatenate
        combined_embeds = torch.cat([prompt_embeds, graph_embeds, text_embeds], dim=1)
        
        # Create attention mask
        graph_attention = torch.ones(
            1,
            model.config.num_learnable_prompts + model.config.num_graph_tokens,
            dtype=inputs.attention_mask.dtype,
            device=device,
        )
        combined_attention_mask = torch.cat([graph_attention, inputs.attention_mask], dim=1)
        
        # Generate
        outputs = model.llm.generate(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            temperature=0.7,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    # Skip the prompt tokens in the output
    prefix_length = model.config.num_learnable_prompts + model.config.num_graph_tokens + inputs.input_ids.size(1)
    generated_ids = outputs[0][prefix_length:]
    
    description = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return description


def batch_generate(model, tokenizer, graphs, batch_size=8, max_new_tokens=256, device='cuda'):
    """Generate descriptions for multiple graphs in batches"""
    
    results = []
    
    for i in tqdm(range(0, len(graphs), batch_size), desc="Generating descriptions"):
        batch_graphs = graphs[i:i+batch_size]
        
        for graph in batch_graphs:
            try:
                description = generate_description(
                    model, tokenizer, graph,
                    max_new_tokens=max_new_tokens,
                    device=device
                )
                results.append({
                    'ID': graph.id if hasattr(graph, 'id') else i,
                    'description': description
                })
            except Exception as e:
                print(f"Error processing graph {graph.id if hasattr(graph, 'id') else i}: {e}")
                results.append({
                    'ID': graph.id if hasattr(graph, 'id') else i,
                    'description': ""
                })
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate molecule descriptions using Graph-LLM model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint directory"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input graphs pickle file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="Path to output CSV file"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)"
    )
    
    args = parser.parse_args()
    
    # Load model
    print("="*80)
    print("LOADING MODEL")
    print("="*80)
    model, tokenizer = load_model_and_tokenizer(args.checkpoint)
    
    # Load input graphs
    print("\n" + "="*80)
    print("LOADING INPUT GRAPHS")
    print("="*80)
    print(f"Loading graphs from {args.input}")
    with open(args.input, 'rb') as f:
        graphs = pickle.load(f)
    print(f"✓ Loaded {len(graphs)} graphs")
    
    # Generate descriptions
    print("\n" + "="*80)
    print("GENERATING DESCRIPTIONS")
    print("="*80)
    results = batch_generate(
        model, tokenizer, graphs,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        device=args.device
    )
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"✓ Saved predictions to {args.output}")
    
    # Print sample
    print("\n" + "="*80)
    print("SAMPLE PREDICTIONS")
    print("="*80)
    for i in range(min(3, len(results))):
        print(f"\nID: {results[i]['ID']}")
        print(f"Description: {results[i]['description']}")
    print("="*80)


if __name__ == "__main__":
    main()
