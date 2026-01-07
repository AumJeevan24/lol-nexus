import os
import time
import requests
import json
from deepeval.synthesizer import Synthesizer
from deepeval.dataset import EvaluationDataset
from rag_core.test_config import groq_judge
from etl.parser import LoLPatchParser

# Configuration
PATCH_URL = "https://www.leagueoflegends.com/en-us/news/game-updates/patch-14-1-notes/"
PATCH_VERSION = "14.1"

def generate_gold_dataset():
    print(f"Fetching Patch {PATCH_VERSION} data...")
    response = requests.get(PATCH_URL)
    
    parser = LoLPatchParser(response.text, PATCH_VERSION)
    chunks = parser.parse()
    
    # Take first 5 chunks
    target_chunks = chunks[:5]
    print(f"Generating synthetic questions for {len(target_chunks)} chunks (Sequential Mode)...")

    synthesizer = Synthesizer(model=groq_judge)
    all_goldens = []

    # Loop manually to prevent Rate Limit (429) errors
    for i, chunk in enumerate(target_chunks):
        print(f"Processing chunk {i+1}/{len(target_chunks)}...")
        
        try:
            # Generate for just ONE chunk
            # Note: We wrap chunk in a list because the function expects a list of contexts
            goldens = synthesizer.generate_goldens_from_contexts(
                contexts=[[chunk.to_text()]], 
                include_expected_output=True,
                max_goldens_per_context=2
            )
            all_goldens.extend(goldens)
            
            # CRITICAL: Sleep to reset the Token Limit (12k tokens/min)
            if i < len(target_chunks) - 1:
                print("Sleeping 10s to cool down API...")
                time.sleep(10)
                
        except Exception as e:
            print(f"Skipped chunk {i} due to error: {e}")

    # Save to File
    if all_goldens:
        print(f"Saving {len(all_goldens)} test cases to 'golden_dataset.json'...")
        dataset = EvaluationDataset(goldens=all_goldens)
        dataset.save_as(file_type="json", directory="./rag_core")
        print("Done. Synthetic test suite created.")
    else:
        print("No data generated. Check errors.")

if __name__ == "__main__":
    generate_gold_dataset()