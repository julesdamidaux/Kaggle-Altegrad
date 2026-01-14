"""
Test script to verify the Graph-LLM setup

Checks:
1. Dependencies are installed
2. Graph encoder checkpoint exists
3. Data files exist
4. Model can be initialized
5. Forward pass works
"""

import os
import sys
from pathlib import Path
import pickle

import torch
print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ CUDA version: {torch.version.cuda}")
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_dependencies():
    """Test if all required packages are installed"""
    print("\n" + "="*80)
    print("TESTING DEPENDENCIES")
    print("="*80)
    
    try:
        import transformers
        print(f"✓ transformers: {transformers.__version__}")
    except ImportError:
        print("✗ transformers not installed. Run: pip install transformers")
        return False
    
    try:
        import peft
        print(f"✓ peft installed")
    except ImportError:
        print("✗ peft not installed. Run: pip install peft")
        return False
    
    try:
        import torch_geometric
        print(f"✓ torch_geometric: {torch_geometric.__version__}")
    except ImportError:
        print("✗ torch_geometric not installed. Run: pip install torch-geometric")
        return False
    
    try:
        import bitsandbytes
        print(f"✓ bitsandbytes installed")
    except ImportError:
        print("✗ bitsandbytes not installed. Run: pip install bitsandbytes")
        print("  (Optional, but needed for 4-bit quantization)")
    
    return True


def test_files():
    """Test if required files exist"""
    print("\n" + "="*80)
    print("TESTING FILES")
    print("="*80)
    
    all_exist = True
    
    # Graph encoder checkpoint
    checkpoint = "/Data/hamzaazzouzi/Kaggle-Altegrad/Hamza/RAG+Predict/embeddings/encoder_checkpoints/best_encoder.pt"
    if os.path.exists(checkpoint):
        print(f"✓ Graph encoder checkpoint: {checkpoint}")
    else:
        print(f"✗ Graph encoder checkpoint not found: {checkpoint}")
        all_exist = False
    
    # Training data
    train_data = "/Data/hamzaazzouzi/Kaggle-Altegrad/data_baseline/data/train_graphs.pkl"
    if os.path.exists(train_data):
        print(f"✓ Training data: {train_data}")
        # Load a sample
        with open(train_data, 'rb') as f:
            graphs = pickle.load(f)
        print(f"  └─ Contains {len(graphs)} graphs")
    else:
        print(f"✗ Training data not found: {train_data}")
        all_exist = False
    
    # Test data
    test_data = "/Data/hamzaazzouzi/Kaggle-Altegrad/data_baseline/data/test_graphs.pkl"
    if os.path.exists(test_data):
        print(f"✓ Test data: {test_data}")
        with open(test_data, 'rb') as f:
            graphs = pickle.load(f)
        print(f"  └─ Contains {len(graphs)} graphs")
    else:
        print(f"✗ Test data not found: {test_data}")
        all_exist = False
    
    return all_exist


def test_model_initialization():
    """Test if model can be initialized"""
    print("\n" + "="*80)
    print("TESTING MODEL INITIALIZATION")
    print("="*80)
    
    try:
        from Hamza.train.finetune_llm_with_graph_encoder import ModelConfig, GraphLLMModel
        
        config = ModelConfig()
        # Use smaller settings for testing
        config.use_4bit = True
        config.num_learnable_prompts = 4
        config.num_graph_tokens = 2
        
        print("Creating model (this may take a minute)...")
        model = GraphLLMModel(config)
        
        print("✓ Model initialized successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✓ Total parameters: {total_params:,}")
        print(f"✓ Trainable parameters: {trainable_params:,}")
        print(f"✓ Trainable ratio: {100 * trainable_params / total_params:.2f}%")
        
        return True
    
    except Exception as e:
        print(f"✗ Failed to initialize model: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass():
    """Test a forward pass through the model"""
    print("\n" + "="*80)
    print("TESTING FORWARD PASS")
    print("="*80)
    
    try:
        from Hamza.train.finetune_llm_with_graph_encoder import (
            ModelConfig, GraphLLMModel, MoleculeGraphDataset, collate_fn
        )
        from transformers import AutoTokenizer
        
        # Load model
        config = ModelConfig()
        config.use_4bit = True
        config.num_learnable_prompts = 4
        config.num_graph_tokens = 2
        
        print("Loading model...")
        model = GraphLLMModel(config)
        model.eval()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.llm_model_name,
            cache_dir=config.cache_dir,
            trust_remote_code=True,
        )
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load a sample graph
        train_data = "/Data/hamzaazzouzi/Kaggle-Altegrad/data_baseline/data/train_graphs.pkl"
        with open(train_data, 'rb') as f:
            graphs = pickle.load(f)
        
        print(f"Testing with graph {graphs[0].id}")
        
        # Create a mini dataset
        class MiniDataset:
            def __init__(self, graphs, tokenizer):
                self.graphs = graphs[:2]  # Just 2 samples
                self.tokenizer = tokenizer
            
            def __len__(self):
                return len(self.graphs)
            
            def __getitem__(self, idx):
                graph = self.graphs[idx]
                description = graph.description if hasattr(graph, 'description') else ""
                
                text = f"<|im_start|>user\nDescribe this molecule.<|im_end|>\n<|im_start|>assistant\n{description}<|im_end|>"
                encoded = self.tokenizer(
                    text,
                    max_length=128,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                
                return {
                    'graph': graph,
                    'input_ids': encoded['input_ids'].squeeze(0),
                    'attention_mask': encoded['attention_mask'].squeeze(0),
                    'labels': encoded['input_ids'].squeeze(0).clone(),
                }
        
        dataset = MiniDataset(graphs, tokenizer)
        batch = collate_fn([dataset[0], dataset[1]])
        
        # Move to GPU if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        batch['graph_data'] = batch['graph_data'].to(device)
        batch['input_ids'] = batch['input_ids'].to(device)
        batch['attention_mask'] = batch['attention_mask'].to(device)
        batch['labels'] = batch['labels'].to(device)
        
        print(f"Running forward pass on {device}...")
        with torch.no_grad():
            outputs = model(
                graph_data=batch['graph_data'],
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'],
            )
        
        print(f"✓ Forward pass successful")
        print(f"✓ Loss: {outputs.loss.item():.4f}")
        print(f"✓ Output shape: {outputs.logits.shape}")
        
        return True
    
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*80)
    print("GRAPH-LLM SETUP TEST")
    print("="*80)
    
    results = {
        'dependencies': test_dependencies(),
        'files': test_files(),
        'model_init': test_model_initialization(),
        'forward_pass': test_forward_pass(),
    }
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED - Ready to train!")
        print("="*80)
        print("\nRun training with:")
        print("  python Hamza/train/finetune_llm_with_graph_encoder.py")
    else:
        print("\n" + "="*80)
        print("✗ SOME TESTS FAILED - Please fix issues above")
        print("="*80)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
