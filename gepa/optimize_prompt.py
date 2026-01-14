# optimize_prompt.py
import sys
sys.path.insert(0, "gepa/src")

import gepa
import pandas as pd
from dotenv import load_dotenv
from gepa.adapters.default_adapter.default_adapter import DefaultAdapter
from semantic_evaluator import SemanticSimilarityEvaluator
from gemini_client import GeminiChatCallable

load_dotenv()

# Base prompt to optimize
BASE_PROMPT = """You are a chemistry expert specializing in molecular structure analysis. Given a SMILES (Simplified Molecular Input Line Entry System) string representing a molecule, generate a detailed chemical description.

Your description MUST include:
1. The molecule class/type (e.g., amino acid, lipid, carbohydrate, alkaloid)
2. Structural features (functional groups, ring systems, stereochemistry)
3. Biological roles if applicable (metabolite, drug, enzyme cofactor)
4. Chemical relationships (conjugate acid/base, derivatives)

Generate a comprehensive description in 2-4 sentences, similar to ChEBI database entries."""


def load_gepa_dataset(csv_path, has_description=True):
    """Convert CSV to GEPA format."""
    data = pd.read_csv(csv_path)
    dataset = []
    for _, row in data.iterrows():
        item = {
            "input": row["SMILES"],
            "additional_context": {"id": str(row["ID"])}
        }
        if has_description:
            item["answer"] = row["Description"]
        dataset.append(item)
    return dataset


def main():
    import random
    random.seed(42)

    # Load data
    print("Loading datasets...")
    train_data = load_gepa_dataset("smiles/train_smiles.csv")
    val_data = load_gepa_dataset("smiles/validation_smiles.csv")

    print(f"Full Train: {len(train_data)}, Full Val: {len(val_data)}")

    # Randomly sample subsets for efficiency
    # Keep validation small so each eval doesn't exhaust the budget
    train_subset = random.sample(train_data, 150)
    val_subset = random.sample(val_data, 50)

    print(f"Using Train: {len(train_subset)}, Val: {len(val_subset)}")

    # Create Gemini callable
    print("Initializing Gemini client...")
    gemini_callable = GeminiChatCallable(model="gemini-3-flash-preview")

    # Create adapter with custom evaluator
    print("Initializing GEPA adapter...")
    adapter = DefaultAdapter(
        model=gemini_callable,
        evaluator=SemanticSimilarityEvaluator(),
        max_litellm_workers=1,
    )

    # Run optimization
    print("Starting GEPA optimization...")
    print(f"Budget: 750 metric calls")
    print("-" * 50)

    # Create a Gemini callable for reflection LM (use pro for better prompt mutations)
    reflection_gemini = GeminiChatCallable(model="gemini-3-pro-preview")

    result = gepa.optimize(
        seed_candidate={"system_prompt": BASE_PROMPT},
        trainset=train_subset,
        valset=val_subset,
        adapter=adapter,
        reflection_lm=reflection_gemini,
        max_metric_calls=500,
        reflection_minibatch_size=10,
        candidate_selection_strategy="pareto",
        module_selector="round_robin",
        display_progress_bar=True,
        run_dir="./gepa_runs/molecule_prompt",
        seed=42,
    )

    # Save optimized prompt
    best_prompt = result.best_candidate["system_prompt"]
    with open("optimized_prompt.txt", "w") as f:
        f.write(best_prompt)

    print("\n" + "=" * 50)
    print("OPTIMIZATION COMPLETE")
    print("=" * 50)
    print(f"Best validation score: {result.val_aggregate_scores[result.best_idx]:.4f}")
    print(f"Total candidates evaluated: {len(result.candidates)}")
    print(f"Optimized prompt saved to: optimized_prompt.txt")
    print("\nOptimized prompt:")
    print("-" * 50)
    print(best_prompt)


if __name__ == "__main__":
    main()
