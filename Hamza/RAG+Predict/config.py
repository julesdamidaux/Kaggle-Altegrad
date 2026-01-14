"""
Configuration for RAG-based prediction system.
"""

import os

# Data paths
TRAIN_GRAPHS = "/Data/hamzaazzouzi/Kaggle-Altegrad/data_baseline/data/train_graphs.pkl"
TEST_GRAPHS = "/Data/hamzaazzouzi/Kaggle-Altegrad/data_baseline/data/test_graphs.pkl"

# Model paths for graph encoding
GRAPH_ENCODER_CHECKPOINT = "/Data/hamzaazzouzi/Kaggle-Altegrad/Hamza/RAG+Predict/embeddings/encoder_checkpoints/best_encoder.pt"

# Graph encoder parameters (should match trained model)
GRAPH_HIDDEN_DIM = 256
GRAPH_LAYERS = 5
NODE_DIM = None  # Will be inferred from data
EDGE_DIM = None  # Will be inferred from data

# RAG parameters
K_NEIGHBORS = 5  # Number of similar graphs to retrieve
DISTANCE_METRIC = "cosine"  # Options: cosine, euclidean, dot

# Storage paths
EMBEDDINGS_DIR = "/Data/hamzaazzouzi/Kaggle-Altegrad/Hamza/RAG+Predict/embeddings"
PREDICTIONS_DIR = "/Data/hamzaazzouzi/Kaggle-Altegrad/Hamza/RAG+Predict/predictions"

os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

# Device
DEVICE = "cuda"  # or "cpu"
