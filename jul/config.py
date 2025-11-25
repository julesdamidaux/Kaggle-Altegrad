"""Configuration for Graph-to-Text model."""

import torch

# =========================================================
# Data Configuration
# =========================================================
TRAIN_GRAPHS = "../data_baseline/data/train_graphs.pkl"
VAL_GRAPHS = "../data_baseline/data/validation_graphs.pkl"
TEST_GRAPHS = "../data_baseline/data/test_graphs.pkl"

# =========================================================
# Model Configuration
# =========================================================
# Graph Encoder: Pretrained Graphormer (FROZEN)
GRAPHORMER_MODEL_NAME = "clefourrier/graphormer-base-pcqm4mv2"
GRAPH_HIDDEN_DIM = 768  # Graphormer output dimension

# Q-Former (Query Transformer)
NUM_QUERY_TOKENS = 32  # Learnable queries to extract graph features #32
QFORMER_HIDDEN_DIM = 768 
QFORMER_NUM_LAYERS = 6 # 6
QFORMER_NUM_HEADS = 8  # 8

# LLM Decoder
LLM_MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"  # Small model for 24GB GPU
USE_LORA = True  # Use LoRA for efficient fine-tuning
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# =========================================================
# Training Configuration
# =========================================================
BATCH_SIZE = 16  # Small batch size for memory efficiency
GRADIENT_ACCUMULATION_STEPS = 8  # Effective batch size = 32
MAX_EPOCHS = 10
LEARNING_RATE = 2e-5
WARMUP_STEPS = 500
MAX_LENGTH = 256  # Maximum caption length

# Optimization
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# Paths
# =========================================================
CHECKPOINT_DIR = "./checkpoints"
OUTPUT_DIR = "./outputs"
LOG_DIR = "./logs"

# =========================================================
# Inference Configuration
# =========================================================
GENERATION_MAX_LENGTH = 256
GENERATION_NUM_BEAMS = 4
GENERATION_TEMPERATURE = 0.7
GENERATION_TOP_P = 0.9
