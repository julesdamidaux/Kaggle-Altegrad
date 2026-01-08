# Qwen2.5-3B-Instruct Fine-tuning for SMILES to Description

Fine-tune Qwen2.5-3B-Instruct to generate molecular descriptions from SMILES strings.

## Features

- **Model**: Qwen2.5-3B-Instruct (3 billion parameter instruction-tuned model)
- **Task**: SMILES → Molecular Description generation
- **Efficient Training**: 
  - LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
  - 4-bit quantization support for reduced memory usage
  - Gradient checkpointing
- **Prompt Engineering**: 
  - System prompt defining the chemistry expert role
  - Structured user prompts with clear instructions
  - Qwen chat template format

## Setup

### 1. Install Dependencies

```bash
pip install -r Hamza/train/requirements_qwen.txt
```

### 2. Prepare Data

Ensure you have the CSV files with SMILES and descriptions:
- `train_smiles.csv` (ID, SMILES, Description)
- `validation_smiles.csv` (ID, SMILES, Description)

## Configuration

Edit [Hamza/train/finetune_config.py](Hamza/train/finetune_config.py) to customize:

### Key Parameters:
- `MODEL_NAME`: Qwen/Qwen2.5-3B-Instruct
- `BATCH_SIZE`: 4 (adjust based on GPU memory)
- `GRADIENT_ACCUMULATION_STEPS`: 4 (effective batch size = 16)
- `LEARNING_RATE`: 2e-4
- `NUM_EPOCHS`: 3
- `MAX_LENGTH`: 512
- `USE_LORA`: True (parameter-efficient fine-tuning)
- `USE_4BIT`: True (reduces memory usage)

### System Prompt:
```
You are a chemistry expert specialized in molecular analysis. Given a SMILES 
(Simplified Molecular Input Line Entry System) representation of a molecule, 
you provide detailed, accurate chemical descriptions including the molecule's 
structure, functional groups, chemical properties, and biological roles.
```

### User Prompt Template:
```
Analyze the following molecule and provide a detailed chemical description.

SMILES: {smiles}

Please describe:
1. The molecular structure and key functional groups
2. Chemical classification and properties
3. Any biological roles or metabolic functions
4. Related compounds or derivatives
```

## Training

### Start Fine-tuning:

```bash
python Hamza/train/finetune_qwen.py
```

The script will:
1. Load Qwen2.5-3B-Instruct model
2. Apply 4-bit quantization (if enabled)
3. Add LoRA adapters to trainable layers
4. Load training and validation datasets
5. Fine-tune the model
6. Save checkpoints every 500 steps
7. Evaluate on validation set every 500 steps
8. Save the final model to `./Hamza/checkpoints/qwen2.5-3b-smiles/final`

### Training Output:

```
Training: [3/3 epochs, loss=X.XX]
Evaluation: [eval_loss=X.XX]
Model saved to: ./Hamza/checkpoints/qwen2.5-3b-smiles/final
```

### Monitor Training:

Enable Weights & Biases logging in config:
```python
USE_WANDB = True
WANDB_PROJECT = "qwen-smiles-finetuning"
```

## Inference

### Generate Descriptions for Test Set:

```bash
python Hamza/train/inference_qwen.py \
    --checkpoint ./Hamza/checkpoints/qwen2.5-3b-smiles/final \
    --csv test_smiles.csv \
    --output test_predictions.csv \
    --batch_size 8
```

### Arguments:
- `--checkpoint`: Path to fine-tuned model
- `--csv`: Input CSV with ID and SMILES columns
- `--output`: Output CSV with predictions
- `--batch_size`: Inference batch size
- `--device`: cuda or cpu

### Output Format:

CSV with columns: `ID`, `SMILES`, `Description`

## Memory Requirements

### With 4-bit Quantization + LoRA:
- Model: ~2.5 GB
- Training: ~8-12 GB GPU memory (batch_size=4)
- Inference: ~6-8 GB GPU memory

### Without Quantization:
- Model: ~6 GB
- Training: ~24+ GB GPU memory required

## File Structure

```
Hamza/train/
├── finetune_config.py      # Configuration and hyperparameters
├── smiles_dataset.py       # Dataset classes
├── finetune_qwen.py        # Training script
├── inference_qwen.py       # Inference script
└── requirements_qwen.txt   # Python dependencies

Hamza/checkpoints/
└── qwen2.5-3b-smiles/
    ├── checkpoint-500/
    ├── checkpoint-1000/
    └── final/              # Final model
```

## Tips

1. **GPU Memory**: If you run out of memory, reduce `BATCH_SIZE` or enable `USE_4BIT`
2. **Training Speed**: Increase `GRADIENT_ACCUMULATION_STEPS` to maintain effective batch size
3. **Quality**: Adjust `NUM_EPOCHS` and monitor validation loss
4. **Prompt Engineering**: Modify prompts in `finetune_config.py` for better results

## Example Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "./Hamza/checkpoints/qwen2.5-3b-smiles/final",
    device_map="auto",
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(
    "./Hamza/checkpoints/qwen2.5-3b-smiles/final"
)

# Generate description
smiles = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
prompt = f"SMILES: {smiles}\n\nDescription:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=512)
description = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(description)
```
