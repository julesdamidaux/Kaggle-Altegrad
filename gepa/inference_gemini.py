# inference_gemini.py
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from gemini_client import GeminiClient
import os
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

load_dotenv()

# Thread-safe CSV writer
csv_lock = threading.Lock()

# Validation: description must start with "The molecule is"
def is_valid_description(desc):
    """Check if description is valid (starts with 'The molecule is')."""
    return desc and desc.strip().startswith("The molecule is")


def generate_description(args):
    """Generate description for a single molecule with validation and retries."""
    mol_id, smiles, optimized_prompt, gemini, max_retries = args
    full_prompt = f"{optimized_prompt}\n\nSMILES: {smiles}"

    for attempt in range(max_retries):
        desc = gemini.generate(full_prompt)
        if is_valid_description(desc):
            return mol_id, desc
        # Invalid output (thinking leaked), retry
        if attempt < max_retries - 1:
            print(f"\n[ID {mol_id}] Invalid output (attempt {attempt + 1}), retrying...")

    # All retries failed, return fallback
    return mol_id, "The molecule is a chemical compound."


def main():
    # Load optimized prompt
    print("Loading optimized prompt...")
    try:
        with open("optimized_prompt.txt", "r") as f:
            optimized_prompt = f.read()
        print("Loaded optimized prompt from file.")
    except FileNotFoundError:
        print("Warning: optimized_prompt.txt not found, using base prompt")
        optimized_prompt = """You are a chemistry expert specializing in molecular structure analysis. Given a SMILES (Simplified Molecular Input Line Entry System) string representing a molecule, generate a detailed chemical description.

Your description MUST include:
1. The molecule class/type (e.g., amino acid, lipid, carbohydrate, alkaloid)
2. Structural features (functional groups, ring systems, stereochemistry)
3. Biological roles if applicable (metabolite, drug, enzyme cofactor)
4. Chemical relationships (conjugate acid/base, derivatives)

Generate a comprehensive description in 2-4 sentences, similar to ChEBI database entries."""

    # Load test data
    print("\nLoading test data...")
    test_df = pd.read_csv("smiles/test_smiles.csv")
    print(f"Test molecules: {len(test_df)}")

    # Initialize Gemini client
    print("Initializing Gemini client...")
    gemini = GeminiClient(model="gemini-3-flash-preview")

    output_file = "test_retrieved_descriptions_flash.csv"
    max_retries = 3  # Retries per molecule for invalid outputs

    # Load existing results and identify valid vs failed
    valid_results = {}  # ID -> description
    failed_ids = set()

    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        for _, row in existing_df.iterrows():
            mol_id = row["ID"]
            desc = str(row["description"])
            if is_valid_description(desc):
                valid_results[mol_id] = desc
            else:
                failed_ids.add(mol_id)
        print(f"Loaded {len(valid_results)} valid results, {len(failed_ids)} failed rows to retry")

    # Find all IDs that need processing (missing + failed)
    all_test_ids = set(test_df["ID"].tolist())
    completed_ids = set(valid_results.keys())
    missing_ids = all_test_ids - completed_ids
    pending_ids = missing_ids | failed_ids

    print(f"Missing IDs: {len(missing_ids)}, Failed IDs: {len(failed_ids)}")
    print(f"Total to process: {len(pending_ids)}")

    # Rewrite CSV with only valid results (removes failed rows)
    if failed_ids:
        print(f"Rewriting CSV to remove {len(failed_ids)} failed rows...")
        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "description"])
            for mol_id in sorted(valid_results.keys()):
                writer.writerow([mol_id, valid_results[mol_id]])

    # Filter to only pending molecules
    pending_rows = [(row["ID"], row["SMILES"]) for _, row in test_df.iterrows()
                    if row["ID"] in pending_ids]

    if not pending_rows:
        print("All molecules already processed!")
        return

    print(f"\nGenerating descriptions for {len(pending_rows)} molecules...")
    print("Using 100 parallel workers (within 1K RPM limit)")

    # Prepare args for parallel execution
    args_list = [(mol_id, smiles, optimized_prompt, gemini, max_retries) for mol_id, smiles in pending_rows]

    failed_count = 0
    results = {}

    # Process in parallel with 100 workers (well under 1K RPM)
    with ThreadPoolExecutor(max_workers=100) as executor:
        futures = {executor.submit(generate_description, args): args[0] for args in args_list}

        with tqdm(total=len(futures)) as pbar:
            for future in as_completed(futures):
                mol_id = futures[future]
                try:
                    result_id, desc = future.result()
                    results[result_id] = desc

                    # Write immediately (thread-safe)
                    with csv_lock:
                        with open(output_file, "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([result_id, desc])

                    if desc == "The molecule is a chemical compound.":
                        failed_count += 1
                except Exception as e:
                    print(f"\nError for molecule {mol_id}: {e}")
                    failed_count += 1
                    with csv_lock:
                        with open(output_file, "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([mol_id, "The molecule is a chemical compound."])

                pbar.update(1)

    if failed_count > 0:
        print(f"\nWarning: {failed_count} generations failed and used fallback.")

    print(f"\nSubmission saved to: {output_file}")

    # Show samples from results
    print("\n" + "=" * 50)
    print("SAMPLE PREDICTIONS")
    print("=" * 50)
    sample_ids = list(results.keys())[:5]
    for mol_id in sample_ids:
        smiles = test_df[test_df["ID"] == mol_id]["SMILES"].values[0]
        print(f"\n--- Molecule {mol_id} ---")
        print(f"SMILES: {smiles[:80]}...")
        print(f"Generated: {results[mol_id][:200]}...")

    print("\n" + "=" * 50)
    print("DONE! Submit test_retrieved_descriptions_flash.csv to Kaggle.")
    print("=" * 50)


if __name__ == "__main__":
    main()
