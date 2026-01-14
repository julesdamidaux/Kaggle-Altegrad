"""
Generate predictions for test graphs using trained Graph-LLM model
"""

import os
import sys
from pathlib import Path
import pickle
import torch
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import Batch as PyGBatch
from transformers import AutoTokenizer
from peft import PeftModel

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Hamza.models.graph_encoder import MoleculeGINE
from Hamza.train.graph_llm_config import (
    GRAPH_ENCODER_CHECKPOINT,
    GRAPH_HIDDEN_DIM,
    GRAPH_LAYERS,
    NODE_DIM,
    EDGE_DIM,
    NUM_LEARNABLE_PROMPTS,
    NUM_GRAPH_TOKENS,
)


def load_model_components(checkpoint_dir, device='cuda'):
    """Load all model components from checkpoint"""
    
    print(f"Loading model from {checkpoint_dir}")
    
    # 1. Load tokenizer
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        cache_dir="./cache",
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    print("✓ Tokenizer loaded")
    
    # 2. Load base LLM with 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        quantization_config=bnb_config,
        cache_dir="./cache",
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    model.eval()
    print("✓ LLM with LoRA loaded")
    
    llm_embed_dim = model.config.hidden_size
    
    # 3. Load graph encoder
    graph_encoder = MoleculeGINE(
        node_dim=NODE_DIM,
        edge_dim=EDGE_DIM,
        hidden=GRAPH_HIDDEN_DIM,
        layers=GRAPH_LAYERS,
        out_dim=GRAPH_HIDDEN_DIM,
    ).to(device)
    
    checkpoint = torch.load(GRAPH_ENCODER_CHECKPOINT, map_location=device)
    graph_encoder.load_state_dict(checkpoint['model_state_dict'])
    graph_encoder.eval()
    print("✓ Graph encoder loaded")
    
    # 4. Load custom components (MLP projector + learnable prompts)
    custom_components_path = os.path.join(checkpoint_dir, 'custom_components.pt')
    custom_state = torch.load(custom_components_path, map_location=device)
    
    # Recreate MLP projector
    from torch import nn
    hidden_dim = (GRAPH_HIDDEN_DIM + llm_embed_dim) // 2
    
    graph_projector = nn.Sequential(
        nn.Linear(GRAPH_HIDDEN_DIM, hidden_dim),
        nn.GELU(),
        nn.LayerNorm(hidden_dim),
        nn.Dropout(0.1),
        nn.Linear(hidden_dim, llm_embed_dim * NUM_GRAPH_TOKENS),
        nn.GELU(),
        nn.LayerNorm(llm_embed_dim * NUM_GRAPH_TOKENS),
    ).to(device)
    
    # Fix state dict keys (remove 'projector.' prefix if present)
    projector_state = custom_state['graph_projector']
    fixed_state = {}
    for key, value in projector_state.items():
        if key.startswith('projector.'):
            fixed_state[key.replace('projector.', '')] = value
        else:
            fixed_state[key] = value
    
    graph_projector.load_state_dict(fixed_state)
    graph_projector.eval()
    
    learnable_prompts = custom_state['learnable_prompts'].to(device)
    
    print("✓ MLP projector and learnable prompts loaded")
    print(f"✓ Model ready on {device}")
    
    return {
        'tokenizer': tokenizer,
        'llm': model,
        'graph_encoder': graph_encoder,
        'graph_projector': graph_projector,
        'learnable_prompts': learnable_prompts,
        'llm_embed_dim': llm_embed_dim,
    }


@torch.no_grad()
def generate_description(components, graph, max_new_tokens=256, device='cuda'):
    """Generate description for a single graph"""
    
    # Create batch with single graph
    graph_batch = PyGBatch.from_data_list([graph]).to(device)
    
    # Prepare prompt
    prompt = "<|im_start|>user\nDescribe this molecule.<|im_end|>\n<|im_start|>assistant\n"
    
    # Tokenize
    inputs = components['tokenizer'](
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)
    
    # 1. Encode graph
    graph_encodings = components['graph_encoder'](
        graph_batch.x.float(),
        graph_batch.edge_index,
        graph_batch.edge_attr.float(),
        graph_batch.batch,
    )
    
    # 2. Project to LLM space
    projected = components['graph_projector'](graph_encodings)
    graph_embeds = projected.view(1, NUM_GRAPH_TOKENS, components['llm_embed_dim'])
    
    # 3. Get learnable prompts
    prompt_embeds = components['learnable_prompts'].unsqueeze(0)
    
    # 4. Get text embeddings
    text_embeds = components['llm'].get_input_embeddings()(inputs.input_ids)
    
    # 5. Convert all embeddings to same dtype (bfloat16 to match LLM)
    target_dtype = text_embeds.dtype
    graph_embeds = graph_embeds.to(target_dtype)
    prompt_embeds = prompt_embeds.to(target_dtype)
    
    # 6. Concatenate all
    combined_embeds = torch.cat([prompt_embeds, graph_embeds, text_embeds], dim=1)
    
    # 7. Create attention mask
    prefix_attention = torch.ones(
        1,
        NUM_LEARNABLE_PROMPTS + NUM_GRAPH_TOKENS,
        dtype=inputs.attention_mask.dtype,
        device=device,
    )
    combined_attention_mask = torch.cat([prefix_attention, inputs.attention_mask], dim=1)
    
    # 8. Generate
    outputs = components['llm'].generate(
        inputs_embeds=combined_embeds,
        attention_mask=combined_attention_mask,
        max_new_tokens=max_new_tokens,
        num_beams=4,
        temperature=0.7,
        do_sample=False,
        pad_token_id=components['tokenizer'].pad_token_id,
        eos_token_id=components['tokenizer'].eos_token_id,
    )
    
    # 9. Decode (skip prompt tokens)
    prefix_length = NUM_LEARNABLE_PROMPTS + NUM_GRAPH_TOKENS + inputs.input_ids.size(1)
    generated_ids = outputs[0][prefix_length:]
    
    description = components['tokenizer'].decode(generated_ids, skip_special_tokens=True)
    
    return description


def main():
    print("="*80)
    print("GENERATE TEST PREDICTIONS WITH GRAPH-LLM")
    print("="*80)
    
    # Configuration
    checkpoint_dir = "./Hamza/checkpoints/qwen2.5-0.5b-graph-mlp-prompts/checkpoint-1938"
    test_graphs_path = "./data_baseline/data/test_graphs.pkl"
    output_csv = "./predictions_graph_llm.csv"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    print("\n" + "-"*80)
    print("LOADING MODEL")
    print("-"*80)
    components = load_model_components(checkpoint_dir, device=device)
    
    # Load test graphs
    print("\n" + "-"*80)
    print("LOADING TEST GRAPHS")
    print("-"*80)
    print(f"Loading from {test_graphs_path}")
    with open(test_graphs_path, 'rb') as f:
        test_graphs = pickle.load(f)
    print(f"✓ Loaded {len(test_graphs)} test graphs")
    
    # Generate predictions
    print("\n" + "-"*80)
    print("GENERATING PREDICTIONS")
    print("-"*80)
    
    results = []
    for graph in tqdm(test_graphs, desc="Generating"):
        try:
            description = generate_description(components, graph, device=device)
            results.append({
                'ID': graph.id if hasattr(graph, 'id') else len(results),
                'description': description.strip()
            })
        except Exception as e:
            print(f"\nError on graph {graph.id if hasattr(graph, 'id') else len(results)}: {e}")
            results.append({
                'ID': graph.id if hasattr(graph, 'id') else len(results),
                'description': ""
            })
    
    # Save results
    print("\n" + "-"*80)
    print("SAVING RESULTS")
    print("-"*80)
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"✓ Saved {len(results)} predictions to {output_csv}")
    
    # Show samples
    print("\n" + "-"*80)
    print("SAMPLE PREDICTIONS")
    print("-"*80)
    for i in range(min(3, len(results))):
        print(f"\nID {results[i]['ID']}:")
        print(f"{results[i]['description']}")
    
    print("\n" + "="*80)
    print("PREDICTION COMPLETE!")
    print("="*80)
    print(f"Output saved to: {output_csv}")


if __name__ == "__main__":
    main()
