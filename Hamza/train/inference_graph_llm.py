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
from peft import PeftModel
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

    # For inference we load the base LLM and then attach trained LoRA adapters
    # from the checkpoint directory instead of expecting a full pytorch_model.bin
    # Checkpoints contain:
    #   - adapter_model.safetensors / adapter_config.json (LoRA)
    #   - custom_components.pt (projector + prompts)

    # Do not create fresh LoRA weights inside GraphLLMModel; we'll load them from disk
    config.use_lora = False

    # Initialize model (loads graph encoder, base LLM, projector, prompts)
    model = GraphLLMModel(config)

    # Attach trained LoRA adapters to the base LLM
    try:
        model.llm = PeftModel.from_pretrained(
            model.llm,
            checkpoint_dir,
            is_trainable=False,
        )
        print("✓ Loaded LoRA adapters from checkpoint")
    except Exception as e:
        print(f"Warning: Could not load LoRA adapters from {checkpoint_dir}: {e}")

    # Load custom components (graph projector + learnable prompts)
    custom_path = os.path.join(checkpoint_dir, "custom_components.pt")
    if os.path.exists(custom_path):
        try:
            custom_state = torch.load(custom_path, map_location="cpu")
            if "graph_projector" in custom_state:
                model.graph_projector.load_state_dict(custom_state["graph_projector"])
            if "learnable_prompts" in custom_state:
                with torch.no_grad():
                    lp = custom_state["learnable_prompts"]
                    # lp may be a Parameter or Tensor
                    lp_data = lp.data if hasattr(lp, "data") else lp
                    model.learnable_prompts.copy_(lp_data)
            print("✓ Loaded custom components from custom_components.pt")
        except Exception as e:
            print(f"Warning: Failed to load custom_components.pt: {e}")
    else:
        print("Warning: custom_components.pt not found; using randomly initialized projector and prompts")
    
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

        # Get text embeddings first to determine target dtype
        text_embeds = model.llm.get_input_embeddings()(inputs.input_ids)
        target_dtype = text_embeds.dtype

        # Project to LLM space and align dtype with LLM embeddings
        graph_embeds = model.graph_projector(graph_encodings).to(target_dtype)

        # Get learnable prompts and align dtype
        prompt_embeds = model.learnable_prompts.unsqueeze(0).to(target_dtype)

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
