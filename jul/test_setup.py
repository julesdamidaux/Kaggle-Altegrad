"""Quick test to verify model initialization and data loading."""

import torch
from utils.model import GraphToTextModel
from utils.data_utils import MoleculeGraphDataset, collate_fn
from torch.utils.data import DataLoader
import config


def test_model_init():
    """Test model initialization."""
    print("="*80)
    print("Testing model initialization...")
    print("="*80)
    
    try:
        model = GraphToTextModel()
        print("✓ Model initialized successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
        
        return model
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        raise


def test_data_loading(model):
    """Test data loading."""
    print("\n" + "="*80)
    print("Testing data loading...")
    print("="*80)
    
    try:
        dataset = MoleculeGraphDataset(
            config.TRAIN_GRAPHS,
            model.tokenizer,
            max_length=config.MAX_LENGTH
        )
        print(f"✓ Dataset loaded: {len(dataset)} samples")
        
        # Test single sample
        sample = dataset[0]
        print(f"✓ Sample keys: {sample.keys()}")
        print(f"  - Graph nodes: {sample['graph'].x.shape}")
        print(f"  - Graph edges: {sample['graph'].edge_index.shape}")
        print(f"  - Input IDs shape: {sample['input_ids'].shape}")
        print(f"  - Text preview: {sample['text'][:100]}...")
        
        # Test dataloader
        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
        batch = next(iter(loader))
        print(f"✓ Batch loaded successfully")
        print(f"  - Batch graph nodes: {batch['graph'].x.shape}")
        print(f"  - Batch input_ids: {batch['input_ids'].shape}")
        
        return dataset, loader
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        raise


def test_forward_pass(model, loader):
    """Test forward pass."""
    print("\n" + "="*80)
    print("Testing forward pass...")
    print("="*80)
    
    try:
        model = model.to(config.DEVICE)
        model.train()
        
        batch = next(iter(loader))
        batch['graph'] = batch['graph'].to(config.DEVICE)
        batch['input_ids'] = batch['input_ids'].to(config.DEVICE)
        batch['attention_mask'] = batch['attention_mask'].to(config.DEVICE)
        
        labels = batch['input_ids'].clone()
        
        loss = model(batch, labels=labels)
        print(f"✓ Forward pass successful")
        print(f"  - Loss: {loss.item():.4f}")
        
        # Test backward
        loss.backward()
        print(f"✓ Backward pass successful")
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        raise


def test_generation(model, loader):
    """Test generation."""
    print("\n" + "="*80)
    print("Testing generation...")
    print("="*80)
    
    try:
        model.eval()
        
        batch = next(iter(loader))
        batch['graph'] = batch['graph'].to(config.DEVICE)
        
        with torch.no_grad():
            captions = model.generate(batch, max_length=50, num_beams=2)
        
        print(f"✓ Generation successful")
        print(f"  - Generated {len(captions)} captions")
        for i, caption in enumerate(captions):
            print(f"  - Caption {i+1}: {caption[:100]}...")
        
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        raise


def main():
    print("\n" + "="*80)
    print("GRAPH-TO-TEXT MODEL TEST SUITE")
    print("="*80 + "\n")
    
    try:
        # Test 1: Model initialization
        model = test_model_init()
        
        # Test 2: Data loading
        dataset, loader = test_data_loading(model)
        
        # Test 3: Forward pass
        test_forward_pass(model, loader)
        
        # Test 4: Generation
        test_generation(model, loader)
        
        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED!")
        print("="*80)
        print("\nYou can now run:")
        print("  python train.py      # To train the model")
        print("  python inference.py  # To generate predictions")
        
    except Exception as e:
        print("\n" + "="*80)
        print("✗ TESTS FAILED")
        print("="*80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
