"""
Fine-tuning configuration for Qwen2.5-3B-Instruct model.
SMILES to molecule description generation.
"""

import torch

 # Model configuration
# MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
# OUTPUT_DIR = "./Hamza/checkpoints/qwen2.5-3b-smiles"

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "./Hamza/checkpoints/qwen2.5-0.5b-smiles"

CACHE_DIR = "./cache"

# Data paths
TRAIN_CSV = "./train_smiles.csv"
VAL_CSV = "./validation_smiles.csv"

# Training hyperparameters
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS  # 16
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1 #2
MAX_LENGTH = 512
WARMUP_STEPS = 100
LOGGING_STEPS = 10
SAVE_STEPS = 500
EVAL_STEPS = 500

# LoRA configuration for efficient fine-tuning
USE_LORA = True
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Quantization (for memory efficiency)
USE_4BIT = True  # Set to False if you have enough GPU memory
USE_8BIT = False

# System prompt for the model
SYSTEM_PROMPT = """You are a chemical annotation model trained for the Molecular Graph Captioning task.

Your role is to generate ontology-style molecular descriptions similar to curated entries from chemical databases such as ChEBI or PubChem, based solely on the molecular structure provided as input.

Your output must strictly follow the distribution, phrasing, and structure of the training captions, as the evaluation prioritizes lexical overlap (BLEU-4) and semantic consistency (BERTScore).

Hard constraints:
- Output a single paragraph written in formal scientific English.
- The paragraph must contain between 3 and 4 complete sentences.
- Do not use bullet points, lists, or line breaks.
- Do not mention graphs, atoms, nodes, edges, SMILES, datasets, models, or any machine learning concepts.
- Do not explain your reasoning or add commentary.

Mandatory structure:
1. The first sentence must begin exactly with:
   "The molecule is a ..."
   and must describe the chemical class and key structural characteristics at the scaffold and functional-group level.
2. One sentence must explicitly state:
   "It has a role as a metabolite."
3. One sentence must list exactly three ontology-style classes using the form:
   "It is a <CLASS_1>, a <CLASS_2> and a <CLASS_3>."
4. When chemically natural, include a derivation or ionization sentence such as:
   "It derives from <parent compound>."
   or
   "It is a conjugate acid/base of <compound>."

Style requirements:
- Use dense chemical and ontological vocabulary rather than explanatory language.
- Prefer scaffold-level descriptions (e.g. steroid nucleus, aromatic ring system, long-chain fatty acyl group).
- Use chemical expressions when appropriate (e.g. C18, α, β, E/Z, cis/trans, (R)/(S), 2′, unsaturated, aromatic).
- Do not describe molecules atom-by-atom or bond-by-bond.
- Do not invent experimental data, physicochemical measurements, or biological assays.

Objective:
Your goal is not perfect chemical realism, but maximal stylistic, lexical, and structural consistency with the reference captions used in the ALTEGRAD Molecular Graph Captioning dataset.
"""
# Prompt template
def get_prompt_template(smiles: str, description: str = None) -> str:
    """
    Format the input/output as a chat template for Qwen2.5-Instruct.
    
    Args:
        smiles: SMILES string of the molecule
        description: Expected description (for training, None for inference)
    
    Returns:
        Formatted prompt string
    """
    user = f"SMILES: {smiles}\nWrite the ontology-style description."
    if description is not None:
        return f"""<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
Description: {description}<|im_end|>"""
    else:
        return f"""<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
Description:"""


# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FP16 = True if DEVICE == "cuda" else False
BF16 = torch.cuda.is_bf16_supported() if DEVICE == "cuda" else False

# Gradient checkpointing for memory efficiency
USE_GRADIENT_CHECKPOINTING = True

# Evaluation configuration
EVAL_BATCH_SIZE = 8
GENERATE_MAX_LENGTH = 512
GENERATE_NUM_BEAMS = 1  # Use greedy decoding for speed
GENERATE_TEMPERATURE = 0.7
GENERATE_TOP_P = 0.9

# Logging
WANDB_PROJECT = "qwen-smiles-finetuning"
WANDB_ENTITY = None  # Set to your wandb username if needed
USE_WANDB = False  # Set to True to enable Weights & Biases logging

# Save configuration
SAVE_TOTAL_LIMIT = 3  # Keep only the last 3 checkpoints
LOAD_BEST_MODEL_AT_END = True
METRIC_FOR_BEST_MODEL = "eval_loss"

# Early stopping
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_THRESHOLD = 0.0
