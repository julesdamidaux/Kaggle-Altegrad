# Complete Training Pipeline: LLM with Graph Encodings + 5NN RAG

## Overview
This pipeline trains an LLM (Qwen2.5) to generate molecular descriptions using:
1. **Graph Encodings**: 256-dim embeddings from trained graph encoder
2. **5NN RAG**: Retrieved descriptions from 5 most similar molecules
3. **Fine-tuning**: LoRA + 4-bit quantization for efficient training

## Architecture

```
Input: Molecular Graph
   ↓
1. Graph Encoder (MoleculeGINE)
   ↓ (256-dim embedding)
2. Retrieval System (finds 5 most similar molecules)
   ↓ (5 descriptions)
3. LLM Input Construction:
   - Graph encoding → projected to 4 tokens
   - Learnable separator → 2 tokens
   - System prompt with 5NN descriptions
   - Text tokens
   ↓
4. Qwen2.5-0.5B with LoRA
   ↓
Output: Ontology-style description
```

## Status

✅ **Step 1: Graph Encoder Training** - COMPLETED
   - Trained with InfoNCE contrastive loss
   - Checkpoint: `Hamza/RAG+Predict/embeddings/encoder_checkpoints/best_encoder.pt`

✅ **Step 2: Graph Encoding** - COMPLETED
   - Train graphs: 31,008 molecules
   - Test graphs: 1,001 molecules
   - Embeddings: `train_embeddings.pkl`, `test_embeddings.pkl`

✅ **Step 3: 5NN Retrieval** - COMPLETED
   - Retrieved 5 nearest neighbors for each molecule
   - Training data: `Hamza/RAG+Predict/embeddings/train_llm_data.pkl`
   - Test data: `Hamza/RAG+Predict/embeddings/test_llm_data.pkl`

✅ **Step 4: Test Predictions (1NN)** - COMPLETED
   - Output: `Hamza/RAG+Predict/predictions/test_best_neighbor.csv`
   - Format: ID, description

🔄 **Step 5: LLM Fine-tuning** - READY TO RUN

## How to Run

### Option 1: Run LLM Training Directly
```bash
cd /Data/hamzaazzouzi/Kaggle-Altegrad
python Hamza/RAG+Predict/train_llm_with_graph_rag.py
```

### Option 2: Use tmux (recommended for long training)
```bash
tmux new -s llm_training
cd /Data/hamzaazzouzi/Kaggle-Altegrad
python Hamza/RAG+Predict/train_llm_with_graph_rag.py 2>&1 | tee llm_training.log
# Press Ctrl+B then D to detach
# tmux attach -t llm_training  # to reattach
```

## Training Configuration

**Model**: Qwen2.5-0.5B-Instruct
**Technique**: LoRA + 4-bit quantization
**Key Parameters**:
- Batch size: 4
- Gradient accumulation: 4 (effective batch size: 16)
- Learning rate: 2e-4
- LoRA rank: 16
- Max length: 512 tokens
- Epochs: 1 (can be increased in config)

**Graph Integration**:
- Graph encoding dim: 256
- Projected to: 4 tokens
- Separator tokens: 2
- Total prefix: 6 tokens

## Data Structure

### Training Sample
```python
{
    'ID': 0,
    'graph_encoding': [256-dim array],
    'Description': 'The molecule is a...',
    'retrieved_desc_1': 'Similar molecule 1 description',
    'retrieved_desc_2': 'Similar molecule 2 description',
    'retrieved_desc_3': 'Similar molecule 3 description',
    'retrieved_desc_4': 'Similar molecule 4 description',
    'retrieved_desc_5': 'Similar molecule 5 description',
}
```

### System Prompt Format
```
Here are descriptions of structurally similar molecules:
1. [retrieved_desc_1]
2. [retrieved_desc_2]
3. [retrieved_desc_3]
4. [retrieved_desc_4]
5. [retrieved_desc_5]

Write the ontology-style description for this molecule.
```

## Output

**Model checkpoints**: `Hamza/checkpoints/qwen2.5-0.5b-smiles-rag/`
**Final model**: `Hamza/checkpoints/qwen2.5-0.5b-smiles-rag/final/`

## Inference

After training, run inference:
```bash
python Hamza/train/inference_qwen_with_rag.py
```

## Files Created

1. `train_llm_with_graph_rag.py` - Main training launcher
2. `retrieve_test_only.py` - Generate 1NN predictions for test set
3. `complete_rag_pipeline.py` - Full pipeline (already run)
4. `predictions/test_best_neighbor.csv` - 1NN baseline predictions

## Notes

- The pipeline uses data already computed from previous RAG pipeline run
- Graph encodings and 5NN retrievals are cached and don't need recomputation
- Training will take several hours depending on GPU
- Model uses learnable projection layer to integrate graph encodings
- The 5NN descriptions provide strong context for generation
