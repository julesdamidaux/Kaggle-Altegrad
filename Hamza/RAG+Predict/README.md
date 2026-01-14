# RAG+Predict: Retrieval-Augmented Generation for Molecular Descriptions

This module implements a RAG (Retrieval-Augmented Generation) system for predicting molecular descriptions. Given a molecular graph, it:

1. **Trains** a graph encoder using contrastive learning (InfoNCE loss)
2. **Encodes** all training graphs into embeddings
3. **Retrieves** the k most similar graphs from the training set
4. **Uses** the descriptions of similar molecules to generate/predict the description for the query molecule

## Architecture

```
                      Contrastive Training
                            ↓
Training Graphs → Graph Encoder → Embeddings → k-NN Index
                                                    ↓
Query Graph → Graph Encoder → Embedding → k-NN Search → Retrieved Descriptions → Prediction
```

## Files

- **`config.py`**: Configuration parameters (paths, model settings, RAG parameters)
- **`train_encoder.py`**: Train graph encoder with contrastive learning (InfoNCE/Triplet loss)
- **`encode_graphs.py`**: Encode all training graphs and store embeddings
- **`retrieval.py`**: k-NN retrieval system with multiple distance metrics
- **`predict.py`**: Main prediction script using RAG

## Usage

### 1. Train the Graph Encoder

First, train the graph encoder using contrastive learning:

```bash
cd /Data/hamzaazzouzi/Kaggle-Altegrad/Hamza/RAG+Predict
python train_encoder.py
```

This will:
- Load training graphs with their text descriptions
- Use sentence-transformers to compute text embeddings
- Train the graph encoder using InfoNCE loss to align graph embeddings with text similarity
- Save checkpoints to `embeddings/encoder_checkpoints/`

**Contrastive Learning Approach:**
- Molecules with similar descriptions should have similar graph embeddings
- Uses InfoNCE (NT-Xent) loss: pulls similar pairs together, pushes dissimilar pairs apart
- Alternative: Triplet loss with hard negative mining

### 2. Encode Training Graphs

Once the encoder is trained, encode all training graphs:

```bash
python encode_graphs.py
```

This will:
- Load the trained graph encoder
- Encode all training graphs
- Save embeddings to `embeddings/train_embeddings.pkl`

### 3. Generate Predictions with RAG

Run predictions on test set using RAG:

```bash
python predict.py
```

This will:
- Load precomputed training embeddings
- Initialize the retrieval system
- For each test graph:
  - Encode the graph
  - Retrieve k=5 most similar training graphs
  - Use their descriptions to generate prediction
- Save predictions to `predictions/rag_predictions.pkl`

## Configuration

Edit `config.py` to customize:

```python
# Graph encoder parameters
GRAPH_HIDDEN_DIM = 256
GRAPH_LAYERS = 5

# Retrieval parameters
K_NEIGHBORS = 5              # Number of similar graphs to retrieve
DISTANCE_METRIC = "cosine"   # Options: cosine, euclidean, dot
```

## Contrastive Loss Functions

Two loss functions are implemented in [`train_encoder.py`](train_encoder.py):

### 1. InfoNCE (NT-Xent) Loss (Default)
- Symmetric contrastive loss
- Aligns graph embeddings with text embeddings
- Temperature parameter controls distribution sharpness
- Best for learning semantic similarity

### 2. Triplet Margin Loss
- Anchor-positive-negative triplets
- Hard negative mining based on text similarity
- Margin parameter controls separation
- Good for metric learning

## Distance Metrics

Three distance metrics are supported:

1. **Cosine similarity** (default): Measures angle between embeddings
   - Range: [-1, 1], higher is more similar
   - Good for semantic similarity

2. **Euclidean distance**: L2 distance between embeddings
   - Range: [0, ∞], lower is more similar
   - Good for direct geometric distance

3. **Dot product**: Inner product of embeddings
   - Range: (-∞, ∞), higher is more similar
   - Fast but sensitive to magnitude

## Output Format

Predictions are saved in two formats:

1. **`rag_predictions.pkl`**: Full details including:
   - Prediction text
   - Retrieved descriptions
   - Retrieval scores
   - RAG prompt used

2. **`rag_predictions.csv`**: Simple CSV with:
   - Graph index
   - Predicted description

## Future Improvements

1. **LLM Integration**: Use retrieved descriptions as context in prompt for T5/Qwen
2. **Weighted retrieval**: Weight retrieved descriptions by similarity score
3. **Query expansion**: Use multiple query representations
4. **Hybrid retrieval**: Combine graph structure + embedding similarity
5. **Re-ranking**: Add a re-ranker after initial retrieval
6. **Cache embeddings**: Use FAISS for faster large-scale retrieval

## Requirements

```bash
pip install torch torch-geometric transformers scikit-learn pandas tqdm
```

## Example

```python
from retrieval import GraphRetriever
import pickle

# Load embeddings
with open('embeddings/train_embeddings.pkl', 'rb') as f:
    data = pickle.load(f)

# Create retriever
retriever = GraphRetriever(
    data['embeddings'],
    data['descriptions'],
    data['graph_ids'],
    metric='cosine'
)

# Retrieve similar graphs
query_embedding = model.encode(test_graph)
descriptions, scores, ids = retriever.retrieve(query_embedding, k=5)

print("Top 5 similar molecules:")
for desc, score in zip(descriptions, scores):
    print(f"Score: {score:.3f} - {desc[:100]}...")
```
