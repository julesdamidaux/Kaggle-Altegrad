# Quick Reference Guide - RAG Pipeline

## 🚀 Quick Start

### Option 1: Automated (Recommended)
```bash
# Start training in tmux session (runs in background)
./start_rag_training.sh

# View progress
tmux attach -t rag

# Detach (keeps running)
# Press: Ctrl+b, then d
```

### Option 2: Manual
```bash
# Attach to tmux session
tmux attach -t rag

# Run pipeline
./run_complete_rag_pipeline.sh
```

## 📁 Key Files Created

### Scripts
- ✅ `run_complete_rag_pipeline.sh` - Master pipeline script
- ✅ `start_rag_training.sh` - Helper to start in tmux
- ✅ `Hamza/RAG+Predict/complete_rag_pipeline.py` - RAG pipeline (train encoder, retrieve, prepare data)

### Models
- ✅ `Hamza/train/qwen_with_graph.py` - Qwen + graph encoding wrapper
- ✅ `Hamza/train/finetune_qwen_with_rag.py` - Fine-tuning script with RAG
- ✅ `Hamza/train/inference_qwen_with_rag.py` - Inference script

### Datasets
- ✅ `Hamza/train/smiles_dataset_rag.py` - Dataset classes for RAG training/inference

## 🔄 Pipeline Steps

### Step 1: RAG Pipeline (Encoder + Retrieval)
```bash
python Hamza/RAG+Predict/complete_rag_pipeline.py
```
**What it does:**
- Trains graph encoder with contrastive learning (InfoNCE loss)
- Encodes all train/test molecules to 256-dim vectors
- Retrieves k=5 closest descriptions for each molecule
- Prepares data with graph encodings + retrieved context

**Outputs:**
- `embeddings/encoder_checkpoints/best_encoder.pt`
- `embeddings/train_embeddings.pkl`
- `embeddings/test_embeddings.pkl`
- `embeddings/train_retrievals.pkl`
- `embeddings/test_retrievals.pkl`
- `embeddings/train_llm_data.pkl` ← Used for LLM training
- `embeddings/test_llm_data.pkl` ← Used for inference

### Step 2: Fine-tune LLM with RAG + Graph Encodings
```bash
python Hamza/train/finetune_qwen_with_rag.py
```
**What it does:**
- Loads Qwen2.5-0.5B-Instruct
- Adds graph projector (256→4 tokens) + 2 learnable separator tokens
- Trains with LoRA + 4-bit quantization
- Input: [GRAPH] + [SEP] + [SMILES + 5 retrieved descriptions]

**Outputs:**
- `checkpoints/qwen2.5-rag/final/` ← Final model
- `checkpoints/qwen2.5-rag/checkpoint-*/` ← Intermediate checkpoints

### Step 3: Generate Predictions
```bash
python Hamza/train/inference_qwen_with_rag.py \
    --checkpoint ./Hamza/checkpoints/qwen2.5-rag/final \
    --data ./Hamza/RAG+Predict/embeddings/test_llm_data.pkl \
    --output predictions_rag.csv
```

**Outputs:**
- `predictions_rag.csv` ← Final predictions (ID, SMILES, Description)

## 🎯 Key Features

### 1. Graph Encoder
- **Architecture:** MoleculeGINE (Graph Isomorphism Network with Edge features)
- **Training:** InfoNCE contrastive loss
- **Output:** 256-dimensional graph embeddings
- **Purpose:** Learn structural representations aligned with text

### 2. RAG Retrieval
- **Method:** Cosine similarity in embedding space
- **K:** 5 closest descriptions
- **Purpose:** Provide relevant vocabulary and patterns from similar molecules

### 3. Graph Integration in LLM
```
Input sequence:
[GRAPH_TOK_1][GRAPH_TOK_2][GRAPH_TOK_3][GRAPH_TOK_4]  ← 4 tokens from graph
[SEP_1][SEP_2]                                          ← 2 learnable separators
<|im_start|>system...SMILES...Retrieved...text...<|im_end|>  ← Normal text
```

**Graph Projector:**
```python
graph_encoding (256) → Linear(256→1024) → LayerNorm → GELU → Linear(1024→1024)
                    → Reshape to (4, 256) → 4 token embeddings
```

**Learnable Separators:**
- 2 trainable embedding vectors (256-dim each)
- Mark boundary between graph and text modalities
- Learned during fine-tuning

## 📊 Monitoring

### View Training Progress
```bash
# Attach to tmux
tmux attach -t rag

# View logs in real-time
tail -f logs/rag_pipeline_*.log

# Check GPU usage
nvidia-smi -l 1
```

### Check Outputs
```bash
# List checkpoints
ls -lh Hamza/checkpoints/qwen2.5-rag/

# Check embeddings
ls -lh Hamza/RAG+Predict/embeddings/

# View predictions
head -20 predictions_rag.csv
```

## ⚙️ Configuration

### Graph Encoder (Hamza/RAG+Predict/config.py)
```python
GRAPH_HIDDEN_DIM = 256        # Graph embedding dimension
GRAPH_LAYERS = 5              # Number of GNN layers
K_NEIGHBORS = 5               # Number of descriptions to retrieve
DISTANCE_METRIC = "cosine"    # Similarity metric
```

### LLM Training (Hamza/train/finetune_config.py)
```python
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # Base model
BATCH_SIZE = 4                  # Per-device batch size
GRADIENT_ACCUMULATION_STEPS = 4 # Effective batch size = 16
LEARNING_RATE = 2e-4           # Learning rate
NUM_EPOCHS = 1                 # Training epochs
USE_LORA = True                # Use LoRA
LORA_R = 16                    # LoRA rank
USE_4BIT = True                # 4-bit quantization
```

### Graph Integration (Hamza/train/qwen_with_graph.py)
```python
graph_encoding_dim = 256      # Input from encoder
num_graph_tokens = 4          # Number of tokens for graph
num_separator_tokens = 2      # Learnable boundary tokens
```

## 🛠️ Tmux Commands

```bash
# Create session (already created)
tmux new-session -s rag

# Attach to session
tmux attach -t rag

# List all sessions
tmux list-sessions

# Detach from inside tmux
Ctrl+b, then d

# Kill session
tmux kill-session -t rag

# Split panes (inside tmux)
Ctrl+b, then "  # Horizontal split
Ctrl+b, then %  # Vertical split

# Navigate panes
Ctrl+b, then arrow keys
```

## 🐛 Troubleshooting

### Out of Memory (OOM)
**Solutions:**
1. Reduce batch size in `finetune_config.py`:
   ```python
   BATCH_SIZE = 2
   GRADIENT_ACCUMULATION_STEPS = 8
   ```
2. Use smaller model:
   ```python
   MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
   ```
3. Reduce graph tokens:
   ```python
   num_graph_tokens = 2  # in qwen_with_graph.py
   ```

### Slow Training
**Solutions:**
1. Ensure 4-bit quantization is enabled:
   ```python
   USE_4BIT = True
   ```
2. Reduce epochs:
   ```python
   NUM_EPOCHS = 1
   ```
3. Use gradient checkpointing:
   ```python
   USE_GRADIENT_CHECKPOINTING = True
   ```

### Data Not Found
**Check:**
```bash
ls -lh data_baseline/data/train_graphs.pkl
ls -lh data_baseline/data/test_graphs.pkl
ls -lh train_smiles.csv
ls -lh test_smiles.csv
```

### Encoder Training Fails
**Check:**
- CUDA available: `python -c "import torch; print(torch.cuda.is_available())"`
- Sentence transformers installed: `pip install sentence-transformers`
- Graph data format: `python -c "import pickle; print(pickle.load(open('data_baseline/data/train_graphs.pkl', 'rb')))"`

## 📈 Expected Timeline

| Step | Task | Time (GPU) |
|------|------|-----------|
| 1 | Train Graph Encoder (30 epochs) | ~2-3 hours |
| 2 | Encode All Graphs | ~10 minutes |
| 3 | Retrieve Descriptions | ~5 minutes |
| 4 | Fine-tune Qwen (1 epoch) | ~3-4 hours |
| 5 | Generate Predictions | ~20 minutes |
| **Total** | | **~6-8 hours** |

## 📚 Additional Documentation

- Full details: `Hamza/RAG+Predict/README_RAG_PIPELINE.md`
- Original encoder: `Hamza/RAG+Predict/README.md`
- LLM training: `Hamza/train/README_QWEN.md`

## ✅ Final Checklist

Before running:
- [ ] Tmux session 'rag' created
- [ ] All scripts executable (`chmod +x *.sh`)
- [ ] Data files present (train_graphs.pkl, test_graphs.pkl, train_smiles.csv)
- [ ] GPU available (`nvidia-smi`)
- [ ] Dependencies installed

To start:
```bash
./start_rag_training.sh
```

Good luck! 🚀
