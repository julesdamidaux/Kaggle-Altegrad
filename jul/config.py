"""Configuration for Graph-to-Text model."""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
# Graph Encoder: Pretrained Graphormer
# We now allow fine-tuning the graph encoder to align it with chemical captions
FREEZE_GRAPH_ENCODER = False
GRAPHORMER_MODEL_NAME = "clefourrier/graphormer-base-pcqm4mv2"
GRAPH_HIDDEN_DIM = 768  # Graphormer output dimension

# Q-Former (Query Transformer)
NUM_QUERY_TOKENS = 32  # Learnable queries to extract graph features #32
QFORMER_HIDDEN_DIM = 768 
QFORMER_NUM_LAYERS = 6 # 6
QFORMER_NUM_HEADS = 8  # 8

# LLM Decoder
hf_token = "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"

LLM_MODEL_NAME = "meta-llama/Llama-3.2-3B"
USE_LORA = True  # Enabled LoRA for efficient fine-tuning
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# =========================================================
# Training Configuration
# =========================================================
# 3-Stage Training Pipeline (inspired by MolCA)
# Stage 1: Graph Encoder + Q-Former (Contrastive/Matching)
# Stage 2: Graph Encoder + Q-Former + Projector (Alignment to Frozen LLM)
# Stage 3: Graph Encoder + Q-Former + Projector + LLM LoRA (End-to-End)

EPOCHS_STAGE1 = 10  # Increased for retrieval alignment
EPOCHS_STAGE2 = 3   # Quick alignment to LLM token space
EPOCHS_STAGE3 = 3  # Generative fine-tuning

BATCH_SIZE = 4  # Reduced to 4 for 4B model on 24GB GPU
GRADIENT_ACCUMULATION_STEPS = 32  # Increased to maintain effective batch size

# Learning Rates
LEARNING_RATE_STAGE1_HEAD = 2e-4  # Stage 1: Q-Former
LEARNING_RATE_STAGE1_ENCODER = 1e-5 # Stage 1: Graph Encoder (lower LR)

LEARNING_RATE_STAGE2_HEAD = 1e-4  # Stage 2: Q-Former + Projector
LEARNING_RATE_STAGE2_ENCODER = 1e-5 # Stage 2: Graph Encoder

LEARNING_RATE_STAGE3_HEAD = 1e-5  # Stage 3: Q-Former + Projector
LEARNING_RATE_STAGE3_ENCODER = 1e-6 # Stage 3: Graph Encoder (very low)
LEARNING_RATE_STAGE3_LORA = 1e-5  # Stage 3: LLM LoRA

WARMUP_STEPS = 300  # Increased for longer training schedule
MAX_LENGTH = 256  # Maximum caption length

# Optimization
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
LOSS_WEIGHT_ITC = 1.0
LOSS_WEIGHT_ITM = 1.0
LOSS_WEIGHT_ITG = 0.1  # Scaled down to balance with ITC/ITM

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
GENERATION_MAX_NEW_TOKENS = 130  # Increased max length
GENERATION_MIN_NEW_TOKENS = 10   # Forced minimum length to ensure detail
GENERATION_NUM_BEAMS = 4  # Very high beam width for max quality (slower)
GENERATION_TEMPERATURE = 1.0  # Sampling for more natural text
GENERATION_TOP_P = 0.9  # Nucleus sampling
GENERATION_REPETITION_PENALTY = 1.2  # Moderate penalty
GENERATION_NO_REPEAT_NGRAM_SIZE = 3  # Hard block on repeating 3-word phrases (STOP LOOPS)
GENERATION_LENGTH_PENALTY = 1.0  

# Prompt Engineering
USE_FEW_SHOT_PROMPTING = False  # Disabled - causing model to copy examples instead of looking at graph

