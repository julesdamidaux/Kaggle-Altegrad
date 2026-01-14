# Graph-to-Text Model for Molecular Captioning

BLIP-inspired architecture using **Pretrained Graphormer (FROZEN)** + **Q-Former** + **Qwen-0.5B LLM** with LoRA fine-tuning.

## Architecture

```
Molecular Graph → Pretrained Graphormer (FROZEN) → Q-Former (32 learnable queries) → Projection → Qwen LLM → Caption
```

### Components:
- **Graph Encoder**: Pretrained Graphormer from HuggingFace (`clefourrier/graphormer-base-pcqm4mv2`) - **FROZEN**
- **Q-Former**: 32 learnable queries extract graph features via cross-attention - **TRAINABLE**
- **Projection**: Linear layer to match LLM dimension - **TRAINABLE**
- **LLM**: Qwen2-0.5B-Instruct with LoRA (r=16) - **LoRA TRAINABLE**

### What Gets Trained:
| Component | Status | Parameters |
|-----------|--------|------------|
| Graphormer | ❄️ **FROZEN** | 0 (pretrained) |
| Q-Former | ✅ **Trainable** | ~100M |
| Projection | ✅ **Trainable** | ~0.7M |
| Qwen LoRA | ✅ **Trainable** | ~5M |
| **Total** | | **~106M params** |

## Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Train Model
```bash
python train.py
```
- Trains for 10 epochs (~2-3 hours on GPU)
- Saves checkpoints to `./checkpoints/`
- Best model: `./checkpoints/best_model.pt`
- Training history: `./logs/training_history.json`

### 2. Generate Predictions
```bash
python inference.py
```
- Loads best model checkpoint
- Generates captions for test set
- Saves to `./outputs/test_predictions.csv`

### 3. Test Setup (Optional)
```bash
python test_setup.py
```
- Verifies model initialization
- Tests data loading
- Tests forward pass and generation

## Configuration

Edit `config.py` to adjust:
- **Q-Former**: Number of queries, layers, heads
- **LLM**: LoRA rank, alpha, dropout
- **Training**: Batch size, learning rate, epochs
- **Generation**: Beam search, temperature, max length

## Key Parameters

- **Batch size**: 4 (effective: 32 with grad accumulation)
- **Learning rate**: 2e-5
- **LoRA rank**: 16
- **Query tokens**: 32
- **Max caption length**: 256 tokens

## Advantages

✅ **Pretrained graph understanding**: Graphormer trained on molecular data  
✅ **Frozen encoder**: No need to train graph encoder from scratch  
✅ **Parameter efficient**: Only ~106M trainable parameters  
✅ **Memory efficient**: Fits in 24GB GPU  
✅ **Fast training**: Converges faster with pretrained components  

## Output Format

CSV with columns: `ID`, `description`

## File Structure

```
jul/
├── config.py              # All hyperparameters
├── train.py              # Training script (EXECUTE)
├── inference.py          # Inference script (EXECUTE)
├── test_setup.py         # Setup verification (EXECUTE)
├── requirements.txt      # Dependencies
├── README.md            # This file
└── utils/               # Helper modules (never executed directly)
    ├── __init__.py
    ├── data_utils.py    # Data loading
    ├── graph_encoder.py # Pretrained Graphormer wrapper
    ├── qformer.py       # Q-Former module
    └── model.py         # Complete model
```

### Executable Scripts
- `train.py` - Train the model
- `inference.py` - Generate predictions
- `test_setup.py` - Verify setup

### Helper Modules (in utils/)
- `data_utils.py` - Dataset classes and data loading
- `graph_encoder.py` - Frozen pretrained Graphormer
- `qformer.py` - Q-Former implementation
- `model.py` - Complete Graph-to-Text model
