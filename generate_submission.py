#!/usr/bin/env python3
"""Submission generation using exact same environment as training script"""

import os
os.environ["WANDB_DISABLED"] = "true"

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# Configuration (same as training)
MODEL_PATH = "./results"
TEST_PARQUET_PATH = "test.parquet"
SUBMISSION_OUTPUT_PATH = "submission.csv"
MAX_LENGTH = 512
NUM_LABELS = 4
HUMAN_LABEL_ID = 0
MACHINE_LABEL_ID = 1
HYBRID_LABEL_ID = 2
ADV_LABEL_ID = 3

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

@torch.no_grad()
def predict_multiview(code, model, tokenizer, max_length, device):
    """
    Run multi-view inference on a single code snippet.
    Exact same function as in training script.
    """
    # Tokenize without truncation
    tokens = tokenizer(code, add_special_tokens=False, truncation=False, return_tensors=None)
    input_ids = tokens["input_ids"]
    tok_len = len(input_ids)
    
    if tok_len == 0:
        # Empty code - return uniform logits
        return torch.zeros(NUM_LABELS).to(device)
    
    if tok_len <= max_length:
        # Single view
        encoded = tokenizer(code, truncation=True, max_length=max_length, return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        outputs = model(**encoded)
        logits = outputs.logits[0]  # [NUM_LABELS]
        return logits
    else:
        # Multi-view: start + end
        # Start crop
        start_tokens = input_ids[:max_length]
        start_code = tokenizer.decode(start_tokens, skip_special_tokens=False)
        encoded_start = tokenizer(start_code, truncation=True, max_length=max_length, return_tensors="pt")
        encoded_start = {k: v.to(device) for k, v in encoded_start.items()}
        outputs_start = model(**encoded_start)
        logits_start = outputs_start.logits[0]  # [NUM_LABELS]
        
        # End crop
        end_tokens = input_ids[-max_length:]
        end_code = tokenizer.decode(end_tokens, skip_special_tokens=False)
        encoded_end = tokenizer(end_code, truncation=True, max_length=max_length, return_tensors="pt")
        encoded_end = {k: v.to(device) for k, v in encoded_end.items()}
        outputs_end = model(**encoded_end)
        logits_end = outputs_end.logits[0]  # [NUM_LABELS]
        
        # Aggregate: mean(logits)
        # Optional: adv boost (max of start_adv, end_adv)
        aggregated_logits = (logits_start + logits_end) / 2.0
        
        # Optional adversarial boost
        adv_logit_start = logits_start[ADV_LABEL_ID]
        adv_logit_end = logits_end[ADV_LABEL_ID]
        aggregated_logits[ADV_LABEL_ID] = torch.max(adv_logit_start, adv_logit_end)
        
        return aggregated_logits

def predict_test_and_write_submission(test_parquet_path, output_csv_path, model, tokenizer, device):
    """
    Stream test set, run multi-view inference, write 4-class predictions to submission.
    Exact same function as in training script.
    """
    model.eval()
    
    # Check if test file exists
    if not os.path.exists(test_parquet_path):
        print(f"❌ Test file not found: {test_parquet_path}")
        print("Please ensure the test file is in the current directory.")
        return False
    
    # Load test data
    try:
        test_df = pd.read_parquet(test_parquet_path)
        print(f"✅ Loaded test data: {len(test_df)} samples")
        
        # Add ID column if it doesn't exist
        if 'ID' not in test_df.columns:
            test_df['ID'] = range(len(test_df))
            print(f"✅ Added ID column (0 to {len(test_df)-1})")
    except Exception as e:
        print(f"❌ Error loading test file: {e}")
        return False
    
    print("\n" + "="*70)
    print("PREDICTING ON TEST SET")
    print("="*70)
    
    try:
        with open(output_csv_path, "w") as f:
            f.write("ID,prediction\n")
            
            count = 0
            for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Predicting"):
                code = row["code"]
                ex_id = row["ID"]
                
                # Run multi-view inference
                logits = predict_multiview(code, model, tokenizer, MAX_LENGTH, device)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                pred_4class = int(np.argmax(probs))
                
                # Write to CSV
                f.write(f"{ex_id},{pred_4class}\n")
                
                count += 1
                if count % 1000 == 0:
                    print(f"Processed {count} samples...")
        
        print(f"\nPredictions saved to {output_csv_path}")
        print(f"Total samples: {count}")
        
        # Verify file exists and show info
        if os.path.exists(output_csv_path):
            file_size = os.path.getsize(output_csv_path) / (1024 * 1024)  # Size in MB
            print(f"\n" + "="*70)
            print("SUBMISSION FILE READY")
            print("="*70)
            print(f"File: {output_csv_path}")
            print(f"Size: {file_size:.2f} MB")
            print(f"Total predictions: {count}")
            print("="*70)
            return True
        else:
            print(f"\n⚠️  WARNING: File {output_csv_path} was not created!")
            return False
            
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return False

def main():
    print("="*70)
    print("SUBMISSION GENERATION - COMPATIBLE MODE")
    print("="*70)
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        print("Please ensure you have trained the model and it's saved in ./results/")
        print("\nTo train the model:")
        print("1. Run: python3 local_4to4.py")
        print("2. Wait for training to complete")
        print("3. Then run this script again")
        return
    
    # Check for required model files
    required_files = ["config.json"]
    model_files = ["pytorch_model.bin", "model.safetensors"]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(MODEL_PATH, file)):
            missing_files.append(file)
    
    # Check for at least one model file format
    has_model = any(os.path.exists(os.path.join(MODEL_PATH, f)) for f in model_files)
    if not has_model:
        missing_files.extend(model_files)
    
    if missing_files:
        print(f"❌ Missing model files: {missing_files}")
        print(f"Expected files in {MODEL_PATH}:")
        for file in required_files:
            print(f"  - {file}")
        print("  - At least one of these model files:")
        for file in model_files:
            print(f"    - {file}")
        print("\nPlease retrain the model to generate these files.")
        return
    
    # Load trained model and tokenizer (same way as training)
    print("Loading trained model and tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
        print(f"✅ Loaded model from {MODEL_PATH}")
        print(f"Model device: {device}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("This might be due to NumPy version compatibility issues.")
        print("Try running: pip install 'numpy<2.0'")
        return
    
    # Generate submission
    success = predict_test_and_write_submission(
        TEST_PARQUET_PATH, 
        SUBMISSION_OUTPUT_PATH, 
        model, 
        tokenizer, 
        device
    )
    
    if success:
        print("\n🎉 Submission generation completed successfully!")
    else:
        print("\n❌ Submission generation failed!")

if __name__ == "__main__":
    main()
