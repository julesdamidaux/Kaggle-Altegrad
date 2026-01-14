# Graph Encoder + LLM Fine-tuning - Quick Start

## ✅ What's Been Created

### 1. **5-NN CSV Files** (in this directory)
- `predictions/test_5nn.csv` - Test graphs with 5 nearest neighbors
- `predictions/test_5nn_full.csv` - Same with scores and IDs

**Format**: `ID, SMILES, NN1_desc, NN2_desc, NN3_desc, NN4_desc, NN5_desc`

### 2. **Graph-LLM Training System** (in `../train/`)
- `finetune_llm_with_graph_encoder.py` - Main training script
- `graph_llm_config.py` - Configuration file
- `inference_graph_llm.py` - Inference script
- `README_GRAPH_LLM.md` - Detailed documentation
- `test_graph_llm_setup.py` - Setup verification script

## 🚀 Quick Start (From Current Directory)

### Step 1: Verify Setup

```bash
# Make sure you're in the kaggle_altegrad conda environment
conda activate kaggle_altegrad

# Test that everything is ready
cd ../train
python test_graph_llm_setup.py
cd -
```

### Step 2: Train the Model

```bash
# Option A: Use the convenience script (creates one for you below)
./train_graph_llm.sh

# Option B: Run directly
cd /Data/hamzaazzouzi/Kaggle-Altegrad
python Hamza/train/finetune_llm_with_graph_encoder.py
```

### Step 3: Generate Predictions

```bash
cd /Data/hamzaazzouzi/Kaggle-Altegrad

# After training completes
python Hamza/train/inference_graph_llm.py \
    --checkpoint Hamza/checkpoints/qwen2.5-0.5b-graph-mlp-prompts/final \
    --input data_baseline/data/test_graphs.pkl \
    --output predictions_graph_llm.csv \
    --batch_size 8
```

## 📊 Architecture

```
Molecular Graph
    ↓
[Pre-trained Graph Encoder] 256D
    ↓
[MLP Projector] → 4 tokens × LLM_dim
    ↓
[8 Learnable Prompts] (trainable)
    ↓
[Concatenate with Text] → [prompts|graph|text]
    ↓
[Qwen LLM + LoRA]
    ↓
Molecule Description
```

## ⚙️ Configuration

Edit `../train/graph_llm_config.py` to customize:

```python
# Model architecture
NUM_LEARNABLE_PROMPTS = 8    # Soft prompt tokens
NUM_GRAPH_TOKENS = 4         # Tokens for graph representation

# Training
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1

# LoRA settings
LORA_R = 16
LORA_ALPHA = 32

# Memory optimization
USE_4BIT = True              # 4-bit quantization
FREEZE_GRAPH_ENCODER = True  # Don't fine-tune graph encoder
```

## 📈 Training Output

Model checkpoints will be saved to:
```
Hamza/checkpoints/qwen2.5-0.5b-graph-mlp-prompts/
├── checkpoint-500/
├── checkpoint-1000/
└── final/
```

## 💾 What Gets Trained

| Component | Parameters | Status |
|-----------|-----------|---------|
| Graph Encoder | ~5M | Frozen (pre-trained) |
| MLP Projector | ~2M | Trained from scratch |
| Learnable Prompts | ~32K | Trained from scratch |
| LLM LoRA | ~10M | Fine-tuned |
| **Total Trainable** | **~12M** | |

## 🔍 Check Your 5-NN CSV

```bash
# View structure
head -2 predictions/test_5nn.csv

# Count rows (should be 1001: header + 1000 test samples)
wc -l predictions/test_5nn.csv

# View a specific molecule's neighbors
head -10 predictions/test_5nn.csv | tail -1
```

## 📝 Files in This Directory

- `retrieve_5nn_with_smiles.py` - Script that generated the 5-NN CSV
- `predictions/test_5nn.csv` - Your 5-NN results
- `predictions/test_5nn_full.csv` - Extended version with scores

## 🆘 Troubleshooting

### Out of Memory
```bash
# In ../train/graph_llm_config.py, reduce:
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
```

### Slow Training
```bash
# Increase batch size if GPU allows
BATCH_SIZE = 4

# Or reduce max sequence length
MAX_LENGTH = 384
```

### Want Better Results
```bash
# Train graph encoder too
FREEZE_GRAPH_ENCODER = False

# Use larger model
LLM_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# More epochs
NUM_EPOCHS = 2
```

## 📚 More Information

See `../train/README_GRAPH_LLM.md` for detailed documentation.
