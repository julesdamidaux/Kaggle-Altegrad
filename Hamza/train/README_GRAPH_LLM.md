# Graph Encoder + LLM Fine-tuning

Fine-tune an LLM with molecular graph encodings using MLP projection and learnable prompts.

## Architecture

```
Molecular Graph
    ↓
[1] Graph Encoder (MoleculeGINE)
    - Pre-trained, optionally frozen
    - Output: [batch, 256]
    ↓
[2] MLP Projector
    - [256] → [NUM_GRAPH_TOKENS × LLM_DIM]
    - Multi-layer: Linear → GELU → LayerNorm → Dropout → Linear
    - Trainable
    ↓
[3] Reshape: [batch, NUM_GRAPH_TOKENS, LLM_DIM]
    ↓
[4] Learnable Prompts: [batch, NUM_LEARNABLE_PROMPTS, LLM_DIM]
    - Trainable parameters
    ↓
[5] Concatenate:
    [prompts | graph_tokens | text_tokens]
    ↓
[6] LLM (Qwen + LoRA)
    - Fine-tuned with LoRA adapters
    ↓
Output: Molecule description
```

## Key Components

### 1. **Graph Encoder (Pre-trained)**
- MoleculeGINE with 5 GNN layers
- Encodes molecular graphs to 256D vectors
- Can be frozen or fine-tuned

### 2. **MLP Projector (Trainable)**
- Projects graph encodings to LLM embedding space
- Multi-layer architecture with residual connections
- Outputs multiple tokens to preserve information

### 3. **Learnable Prompts (Trainable)**
- Soft prompts that condition the LLM
- Learned during training
- Help bridge graph and text modalities

### 4. **LLM with LoRA (Fine-tuned)**
- Qwen 2.5 (0.5B or 3B)
- LoRA adapters for efficient fine-tuning
- 4-bit quantization for memory efficiency

## Setup

### Prerequisites

```bash
conda activate kaggle_altegrad

# Install required packages
pip install torch torch-geometric transformers peft accelerate bitsandbytes
```

### Configuration

Edit `graph_llm_config.py` to customize:

```python
# Model sizes
NUM_LEARNABLE_PROMPTS = 8  # Number of soft prompt tokens
NUM_GRAPH_TOKENS = 4       # Tokens to represent graph

# Training
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1

# LoRA
LORA_R = 16
LORA_ALPHA = 32
```

## Training

### Quick Start

```bash
cd /Data/hamzaazzouzi/Kaggle-Altegrad

# Activate environment
conda activate kaggle_altegrad

# Run training
python Hamza/train/finetune_llm_with_graph_encoder.py
```

### Custom Configuration

1. Edit `Hamza/train/graph_llm_config.py`
2. Adjust model architecture and hyperparameters
3. Run training

### Training Output

The model will be saved to:
```
Hamza/checkpoints/qwen2.5-0.5b-graph-mlp-prompts/
├── checkpoint-500/
├── checkpoint-1000/
└── final/
    ├── adapter_config.json
    ├── adapter_model.bin
    ├── tokenizer_config.json
    └── ...
```

## Inference

### Generate Predictions

```bash
python Hamza/train/inference_graph_llm.py \
    --checkpoint Hamza/checkpoints/qwen2.5-0.5b-graph-mlp-prompts/final \
    --input data_baseline/data/test_graphs.pkl \
    --output predictions_graph_llm.csv \
    --batch_size 8
```

### Arguments

- `--checkpoint`: Path to trained model directory
- `--input`: Path to input graphs (.pkl file)
- `--output`: Output CSV file (ID, description)
- `--batch_size`: Inference batch size
- `--max_new_tokens`: Max tokens to generate (default: 256)
- `--device`: cuda or cpu

## Model Details

### Trainable Parameters

1. **MLP Projector**: ~2M parameters
   - Projects graph encodings to LLM space
   - Trained from scratch

2. **Learnable Prompts**: ~32K parameters
   - 8 tokens × 4096D (for 0.5B model)
   - Trained from scratch

3. **LoRA Adapters**: ~10M parameters
   - Rank 16, alpha 32
   - Applied to attention layers

4. **Graph Encoder**: 0 parameters (frozen)
   - Or ~5M if fine-tuned

**Total trainable**: ~12M parameters (with frozen graph encoder)

### Memory Requirements

| Configuration | GPU Memory | Training Speed |
|--------------|------------|----------------|
| 0.5B + 4-bit | ~8 GB      | ~10 samples/sec|
| 0.5B + 8-bit | ~12 GB     | ~12 samples/sec|
| 3B + 4-bit   | ~16 GB     | ~4 samples/sec |

## Advantages

1. **Uses Pre-trained Graph Encoder**
   - Leverages existing graph understanding
   - No need to train GNN from scratch

2. **Efficient Fine-tuning**
   - LoRA: only trains ~12M params
   - 4-bit quantization: low memory
   - Can run on single GPU

3. **Flexible Architecture**
   - Easy to adjust number of graph tokens
   - Learnable prompts adapt to task
   - Can freeze/unfreeze graph encoder

4. **Modular Design**
   - Each component can be modified independently
   - Easy to experiment with different LLMs
   - Configurable via config file

## Tips

### For Better Performance

1. **More Graph Tokens**: Increase `NUM_GRAPH_TOKENS` (4 → 8)
   - Preserves more graph information
   - Requires more memory

2. **More Learnable Prompts**: Increase `NUM_LEARNABLE_PROMPTS` (8 → 16)
   - Better conditioning
   - Slightly more parameters

3. **Fine-tune Graph Encoder**: Set `FREEZE_GRAPH_ENCODER = False`
   - Better adaptation
   - More trainable parameters

4. **Larger LoRA Rank**: Increase `LORA_R` (16 → 32)
   - More expressive
   - More parameters to train

### For Faster Training

1. **Smaller Model**: Use Qwen2.5-0.5B instead of 3B
2. **Larger Batch Size**: Increase if GPU allows
3. **Gradient Accumulation**: Simulate larger batches
4. **Mixed Precision**: Use BF16 (enabled by default)

### For Lower Memory

1. **4-bit Quantization**: Already enabled
2. **Gradient Checkpointing**: Already enabled
3. **Smaller Batch Size**: Reduce to 1 if needed
4. **Freeze Graph Encoder**: Already recommended

## Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce batch size
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16

# Or use smaller model
LLM_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
```

### Slow Training

```bash
# Increase batch size if GPU allows
BATCH_SIZE = 4

# Reduce max length
MAX_LENGTH = 384

# Use FP16 instead of BF16 (if GPU supports)
FP16 = True
BF16 = False
```

### Bad Predictions

```bash
# Unfreeze graph encoder
FREEZE_GRAPH_ENCODER = False

# Train longer
NUM_EPOCHS = 2

# Use larger model
LLM_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
```

## Files

- `finetune_llm_with_graph_encoder.py`: Main training script
- `graph_llm_config.py`: Configuration file
- `inference_graph_llm.py`: Inference script
- `README_GRAPH_LLM.md`: This file

## Citation

If you use this code, please cite:
- Graph Encoder: MoleculeGINE
- LLM: Qwen 2.5
- LoRA: Hu et al., 2021
