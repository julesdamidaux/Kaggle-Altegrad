import os
import pandas as pd
import re
import config

def clean_predictions():
    # Paths
    predictions_path = os.path.join(config.OUTPUT_DIR, "test_predictions.csv")
    baseline_path = "../data_baseline/test_retrieved_descriptions.csv"
    
    if not os.path.exists(predictions_path):
        print(f"Error: Predictions file not found at {predictions_path}")
        return

    if not os.path.exists(baseline_path):
        # Try relative path from script location if run from project root
        baseline_path = "data_baseline/test_retrieved_descriptions.csv"
        if not os.path.exists(baseline_path):
            print(f"Error: Baseline file not found at {baseline_path}")
            return
            
    print(f"Loading predictions from: {predictions_path}")
    print(f"Loading baseline from: {baseline_path}")
    
    predictions_df = pd.read_csv(predictions_path)
    baseline_df = pd.read_csv(baseline_path)
    
    # Create a baseline map for fast lookup
    baseline_map = dict(zip(baseline_df['ID'], baseline_df['description']))
    
    cleaned_count = 0
    # Specific pattern requested: 1, 2, 3, ..., 12
    catastrophic_pattern = re.compile(r'1\s*,\s*2\s*,\s*3\s*,\s*4\s*,\s*5\s*,\s*6\s*,\s*7\s*,\s*8\s*,\s*9\s*,\s*10\s*,\s*11\s*,\s*12')
    
    for i, row in predictions_df.iterrows():
        cid = row['ID']
        caption = str(row['description'])
        
        # Check for catastrophic failure
        if catastrophic_pattern.search(caption):
            if cid in baseline_map:
                predictions_df.at[i, 'description'] = baseline_map[cid]
                cleaned_count += 1
            else:
                print(f"Warning: No baseline found for catastrophic caption ID {cid}")
                
    print(f"Replaced {cleaned_count} catastrophic captions with baseline.")
    
    # Save processed predictions
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Saved cleaned predictions to: {predictions_path}")

if __name__ == "__main__":
    clean_predictions()
