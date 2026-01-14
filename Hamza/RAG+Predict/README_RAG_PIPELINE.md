# RAG Pipeline for Molecular Graph Captioning

This directory contains a complete pipeline for molecular graph captioning using:
1. **Graph Encoder** - Contrastive learning on molecular graphs
2. **RAG (Retrieval Augmented Generation)** - Retrieve descriptions from similar molecules
3. **LLM with Graph Encodings** - Fine-tune Qwen with graph encodings and retrieved context

## Architecture Overview

### 1. Graph Encoder Training
- Uses **MoleculeGINE** (Graph Isomorphism Network with Edge features)
- Trained with **InfoNCE contrastive loss** to align graph and text embeddings
- Learns to map molecular graphs to a 256-dimensional embedding space

### 2. Retrieval System
- Encodes all training molecules into embedding space
- For each query molecule, retrieves k=5 most similar descriptions using cosine similarity
- Provides relevant context from structurally similar molecules

### 3. LLM with Graph Integration
- **Base Model**: Qwen2.5-0.5B-Instruct (or 3B)
- **Graph Integration**: 
  - Graph encodings projected to token embedding space (4 tokens)
  - Learnable separator tokens (2 tokens) between graph and text
  - Architecture: `[GRAPH_TOKENS] + [SEPARATOR] + [TEXT]`
- **Training**: LoRA fine-tuning with 4-bit quantization
- **Input**: SMILES + 5 retrieved descriptions + graph encoding
- **Output**: Ontology-style molecular description

## Key Features

### Learnable Separation
The model uses learnable separator tokens to distinguish between:
- Graph-based features (from graph encoder)
- Text-based context (SMILES and retrieved descriptions)

This allows the model to effectively integrate both modalities.

### Multi-Modal Input
```
Input = [Graph Encoding (4 tokens)] + [Separator (2 tokens)] + [Text Tokens]
       |                           |                        |
       Graph structure info        Learnable boundary      SMILES + RAG context
```

## Files Created

### Core Pipeline
- `complete_rag_pipeline.py` - Main pipeline script
  - Step 1: Train graph encoder
  - Step 2: Encode all graphs
  - Step 3: Retrieve k=5 descriptions
  - Step 4: Prepare data for LLM

### Model Components
- `qwen_with_graph.py` - Qwen wrapper with graph encoding support
  - `QwenWithGraphEncoding` class
  - Graph projector network
  - Learnable separator embeddings

### Dataset Classes
- `smiles_dataset_rag.py` - Dataset classes for RAG
  - `SmilesDescriptionRAGDataset` - Training with graph encodings
  - `SmilesInferenceRAGDataset` - Inference with RAG context

### Training & Inference
- `finetune_qwen_with_rag.py` - Fine-tune Qwen with RAG
  - Custom trainer for graph encodings
  - LoRA + 4-bit quantization
  - Handles graph projector training

- `inference_qwen_with_rag.py` - Generate predictions
  - Loads model with graph support
  - Generates with RAG context

### Automation
- `run_complete_rag_pipeline.sh` - Master bash script
  - Runs entire pipeline end-to-end
  - Logging and error handling

## Usage

### Quick Start
```bash
# Attach to tmux session
tmux attach -t rag

# Run complete pipeline
./run_complete_rag_pipeline.sh
```

### Step-by-Step Execution

#### Step 1: RAG Pipeline (Encoder + Retrieval)
```bash
python Hamza/RAG+Predict/complete_rag_pipeline.py
```

This will:
- Train graph encoder (if not exists)
- Encode all train/test graphs
- Retrieve 5 closest descriptions for each
- Save prepared data for LLM training

**Output:**
- `Hamza/RAG+Predict/embeddings/encoder_checkpoints/best_encoder.pt`
- `Hamza/RAG+Predict/embeddings/train_embeddings.pkl`
- `Hamza/RAG+Predict/embeddings/test_embeddings.pkl`
- `Hamza/RAG+Predict/embeddings/train_retrievals.pkl`
- `Hamza/RAG+Predict/embeddings/test_retrievals.pkl`
- `Hamza/RAG+Predict/embeddings/train_llm_data.pkl`
- `Hamza/RAG+Predict/embeddings/test_llm_data.pkl`

#### Step 2: Fine-tune LLM with RAG
```bash
python Hamza/train/finetune_qwen_with_rag.py
```

This will:
- Load base Qwen model
- Wrap with graph encoding support
- Apply LoRA for efficient fine-tuning
- Train on data with graph encodings + RAG context

**Output:**
- `Hamza/checkpoints/qwen2.5-rag/final/` (full model)
- `Hamza/checkpoints/qwen2.5-rag/checkpoint-*` (intermediate)

#### Step 3: Generate Predictions
```bash
python Hamza/train/inference_qwen_with_rag.py \
    --checkpoint ./Hamza/checkpoints/qwen2.5-rag/final \
    --data ./Hamza/RAG+Predict/embeddings/test_llm_data.pkl \
    --output predictions_rag.csv \
    --batch_size 8 \
    --max_new_tokens 256
```

**Output:**
- CSV file with columns: ID, SMILES, Description

## Configuration

### Graph Encoder (config.py)
```python
GRAPH_HIDDEN_DIM = 256
GRAPH_LAYERS = 5
K_NEIGHBORS = 5
DISTANCE_METRIC = "cosine"
```

### LLM Training (finetune_config.py)
```python
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1
USE_LORA = True
LORA_R = 16
USE_4BIT = True
```

### Graph Integration (qwen_with_graph.py)
```python
graph_encoding_dim = 256
num_graph_tokens = 4      # Tokens for graph representation
num_separator_tokens = 2  # Learnable boundary tokens
```

## Model Architecture Details

### Graph Projector
```
Input: [batch_size, 256] (graph encoding)
↓
Linear(256 → 1024) + LayerNorm + GELU + Linear(1024 → 1024)
↓
Reshape: [batch_size, 4, 256] (4 tokens of 256 dims)
↓
Output: 4 token embeddings
```

### Learnable Separators
```
2 trainable embedding vectors of size 256
Randomly initialized, learned during fine-tuning
Serve as boundary markers between graph and text
```

### Full Input Sequence
```
[GRAPH_TOK_1] [GRAPH_TOK_2] [GRAPH_TOK_3] [GRAPH_TOK_4] 
[SEP_1] [SEP_2] 
<|im_start|>system... [TEXT TOKENS] ...<|im_end|>
```

## Tmux Session Management

```bash
# Create session (already done)
tmux new-session -s rag

# Attach to session
tmux attach -t rag

# Detach from session (inside tmux)
Ctrl+b, then d

# List sessions
tmux list-sessions

# Kill session
tmux kill-session -t rag
```

## Expected Improvements

1. **Graph Structure**: Direct encoding of molecular structure supplements SMILES
2. **RAG Context**: Similar molecules provide relevant vocabulary and patterns
3. **Multi-Modal Learning**: Joint optimization of graph and text features
4. **Learnable Integration**: Model learns optimal way to combine modalities

## Monitoring Training

### Using Weights & Biases (if enabled)
Set `USE_WANDB = True` in `finetune_config.py` and login:
```bash
wandb login
```

### Check Training Logs
```bash
# View real-time logs
tail -f logs/rag_pipeline_*.log

# Check training output
ls -lh Hamza/checkpoints/qwen2.5-rag/
```

### Evaluate Predictions
```bash
# Check output
head predictions_rag.csv

# Count predictions
wc -l predictions_rag.csv
```

## Troubleshooting

### Out of Memory
- Reduce `BATCH_SIZE` in `finetune_config.py`
- Use smaller model: `Qwen/Qwen2.5-0.5B-Instruct`
- Reduce `num_graph_tokens` from 4 to 2

### Slow Training
- Increase `BATCH_SIZE` and `GRADIENT_ACCUMULATION_STEPS`
- Use `USE_4BIT = True` for quantization
- Reduce `NUM_EPOCHS`

### Data Not Found
Ensure you have:
- `data_baseline/data/train_graphs.pkl`
- `data_baseline/data/test_graphs.pkl`
- `train_smiles.csv`
- `test_smiles.csv`

## Citation

This pipeline implements ideas from:
- **GINE**: Graph Isomorphism Network with Edge features
- **InfoNCE**: Contrastive learning for representation learning
- **LoRA**: Low-Rank Adaptation for efficient fine-tuning
- **RAG**: Retrieval-Augmented Generation

## Contact

For questions or issues, check the main repository README or logs.
