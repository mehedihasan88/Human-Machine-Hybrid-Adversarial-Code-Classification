#!/usr/bin/env python3
"""Task C: Hybrid Code Detection with AST Enhancement - Local Version

Core Strategy:
- Train 4-class classifier directly (Human, Machine, Hybrid, Adversarial)
- ENHANCED with AST features for language-agnostic classification
- Handle unknown programming languages in test data using structural analysis
- NO sliding-window inference
- Handle long code via multi-view cropping (start / end / optional middle)
- Improve rare classes using balanced subset sampling, focal loss, multi-view inference aggregation, and hard negative mining

AST ENHANCEMENTS:
- Language-agnostic AST feature extraction
- Structural pattern analysis independent of programming language
- Combined transformer + AST features for robust classification
- Handles "Unknown" language in test data (84,571 samples)

LOCAL VERSION MODIFICATIONS:
- Removed Kaggle-specific paths
- Updated data paths to use local files
- Disabled submission generation by default
- Added local file handling
- Integrated AST feature extraction
"""

import os
os.environ["WANDB_DISABLED"] = "true"

import torch
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback)
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    confusion_matrix, f1_score)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import warnings
import json
import time
from collections import defaultdict
import pickle

# Import AST feature extractor
from ast_feature_extractor import ASTFeatureExtractor

warnings.filterwarnings("ignore")

# ==================================================
# SECTION A — CONFIG (LOCAL VERSION WITH AST)
# ==================================================

# Data / Subset - ENABLED WITHOUT BALANCING
USE_SUBSET = True
SUBSET_TOTAL_N = 80_000         # Total samples for subset
SUBSET_SEED = 42
BALANCE_BY_LANGUAGE = False      # Disable balancing
BALANCE_BY_GENERATOR = False
SAVE_SUBSET_PATH = "train_subset_4class_ast.parquet"

# Validation control
USE_FULL_VALIDATION = False     # False for local testing to speed up
VAL_DRY_RUN_MAX_N = 20_00       # Reduced for local testing
VAL_SUBSET_SEED = 123

# Labels (4-class direct)
HUMAN_LABEL_ID = 0
MACHINE_LABEL_ID = 1
HYBRID_LABEL_ID = 2
ADV_LABEL_ID = 3
NUM_LABELS = 4

# AST Configuration
USE_AST_FEATURES = True
AST_FEATURE_WEIGHT = 0.3        # Weight for AST features in final prediction
COMBINE_METHOD = "ensemble"     # Options: "concatenate", "ensemble", "late_fusion"

# Token / Cropping
MAX_LENGTH = 512
CROP_STRATEGY = "start_end_middle"        # options: start_only | start_end | start_end_middle
MIDDLE_CROP_FOR = ["hybrid", "adversarial"]
RANDOM_CROP_SEED = 123

# Training - OPTIMIZED FOR FULL DATASET
MODEL_NAME = "microsoft/unixcoder-base"
# MODEL_NAME = "microsoft/codebert-base"  # Alternative option
# MODEL_NAME = "Salesforce/codet5-base"   # If you want to experiment

PER_DEVICE_TRAIN_BATCH = 16        # Reduced for full dataset memory efficiency
PER_DEVICE_EVAL_BATCH = 32         # Reduced for memory efficiency
GRAD_ACCUM = 2                    # Increased to maintain effective batch size
LR = 3e-5                         # Slightly higher for full dataset
NUM_EPOCHS = 1                    # Reduced for full dataset training time
WARMUP_RATIO = 0.2
MAX_GRAD_NORM = 1.0
FP16 = True if torch.cuda.is_available() else False

# Loss
USE_FOCAL_LOSS = False
FOCAL_GAMMA = 1.0
USE_CLASS_WEIGHTS = True

# Logging / Eval
LOGGING_STEPS = 100               # Reduced for local testing
SAVE_TOTAL_LIMIT = 2
REPORT_LENGTH_BUCKETS = True
SAVE_MISTAKES = True
MISTAKES_N = 50                   # Reduced for local testing

# Early Stopping Configuration - DISABLED
USE_EARLY_STOPPING = False        # Disable early stopping
EARLY_STOPPING_PATIENCE = 2       # Stop after 2 evaluations without improvement (reduced for epoch-wise)
EARLY_STOPPING_THRESHOLD = 0.001  # Minimum improvement threshold
EVAL_STRATEGY = "epoch"           # Evaluation strategy: "steps" or "epoch"
EVAL_STEPS = 500                  # Evaluate every 500 steps (not used with "epoch" strategy)
METRIC_FOR_BEST_MODEL = "eval_f1" # Metric to monitor for early stopping
GREATER_IS_BETTER = True          # Higher F1 is better
LOAD_BEST_MODEL_AT_END = True     # Load best model at end of training

# Hard Negative Mining - OPTIMIZED FOR FULL DATASET
DO_HARD_MINING = False
HARD_MINING_AFTER_EPOCH = 1
HARD_MINING_MIN_CONF = 0.70       # Lowered for full dataset diversity
HARD_MINING_FOCUS_LABELS = [HYBRID_LABEL_ID, ADV_LABEL_ID]
HARD_MINING_INCLUDE_LOW_CONF = True  # Include low-confidence correct predictions
HARD_MINING_LOW_CONF_THRESHOLD = 0.65  # Lowered for more comprehensive mining

# Dynamic mining size - OPTIMIZED FOR FULL DATASET
HARD_MINING_FRAC = 0.05           # Reduced fraction for full dataset
HARD_MINING_MAX_CAP = 50_000       # Increased cap for full dataset

# Test prediction / submission - LOCAL VERSION
GENERATE_SUBMISSION = True       # Enable for full dataset testing
TEST_PARQUET_PATH = "test.parquet"  # Use full test file
SUBMISSION_OUTPUT_PATH = "submission_ast.csv"  # Local output

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==================================================
# SECTION B — DATA LOADING + AST FEATURE EXTRACTION
# ==================================================

# Initialize AST feature extractor
ast_extractor = ASTFeatureExtractor()

# Check if local data files exist
def check_local_files():
    """Check if required local data files exist."""
    required_files = [
        "task_c_training_set_1.parquet",
        "task_c_validation_set.parquet"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required data files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure the following files are in the current directory:")
        for file in required_files:
            print(f"   - {file}")
        print("\nYou can download these from the competition dataset or use the sample files provided.")
        return False
    
    print("✅ All required data files found!")
    return True

# Check for local files
if not check_local_files():
    print("\n⚠️  Exiting due to missing data files.")
    exit(1)

# Load train and validation - LOCAL PATHS
try:
    train_df = pd.read_parquet('task_c_training_set_1.parquet')
    val_df = pd.read_parquet('task_c_validation_set.parquet')
    print("✅ Successfully loaded local data files")
except Exception as e:
    print(f"❌ Error loading data files: {e}")
    print("\nPlease ensure you have the correct parquet files in the current directory.")
    exit(1)

print("="*70)
print("ORIGINAL 4-CLASS DISTRIBUTIONS")
print("="*70)
print("\nTraining set:")
print(train_df["label"].value_counts().sort_index())
print("\nValidation set:")
print(val_df["label"].value_counts().sort_index())

# Check language distribution
if "language" in train_df.columns:
    print("\nTraining language distribution:")
    print(train_df["language"].value_counts())
else:
    print("\nNo language column in training data")

if "language" in val_df.columns:
    print("\nValidation language distribution:")
    print(val_df["language"].value_counts())
else:
    print("\nNo language column in validation data")

# Initialize tokenizer for length computation
temp_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

def compute_token_lengths(df, tokenizer, batch_size=256):
    """Compute true token length for each code snippet."""
    texts = df["code"].tolist()
    lengths = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Computing token lengths"):
        batch_texts = texts[i:i+batch_size]
        encoded = tokenizer(
            batch_texts,
            add_special_tokens=False,
            truncation=False,
            return_length=True
        )
        lengths.extend(encoded["length"])
    
    df = df.copy()
    df["tok_len"] = lengths
    return df

def extract_ast_features_batch(df, batch_size=100):
    """Extract AST features for a dataframe in batches."""
    print("Extracting AST features...")
    ast_features_list = []
    
    for i in tqdm(range(0, len(df), batch_size), desc="Extracting AST features"):
        batch_df = df.iloc[i:i+batch_size]
        batch_features = []
        
        for _, row in batch_df.iterrows():
            code = row["code"]
            language = row.get("language", None)
            features = ast_extractor.extract_features(code, language)
            batch_features.append(features)
        
        ast_features_list.extend(batch_features)
    
    # Convert to DataFrame
    ast_features_df = pd.DataFrame(ast_features_list)
    
    # Add prefix to feature names
    ast_features_df.columns = [f"ast_{col}" for col in ast_features_df.columns]
    
    return ast_features_df

def make_validation_subset(val_df, max_n, seed):
    """
    Create a stratified validation subset capped at max_n.
    Keeps label balance and shuffles.
    """
    np.random.seed(seed)
    
    if len(val_df) <= max_n:
        return val_df.reset_index(drop=True)
    
    per_label = max_n // val_df["label"].nunique()
    subsets = []
    
    for label_id in sorted(val_df["label"].unique()):
        label_df = val_df[val_df["label"] == label_id]
        n = min(per_label, len(label_df))
        subsets.append(label_df.sample(n=n, random_state=seed))
    
    subset_df = pd.concat(subsets, ignore_index=True)
    return subset_df.sample(frac=1, random_state=seed).reset_index(drop=True)

# Compute token lengths
print("\n" + "="*70)
print("COMPUTING TOKEN LENGTHS")
print("="*70)
train_df = compute_token_lengths(train_df, temp_tokenizer)
val_df = compute_token_lengths(val_df, temp_tokenizer)

# Apply validation subsetting (if needed)
if not USE_FULL_VALIDATION:
    print("\n" + "="*70)
    print("[LOCAL RUN] LIMITING VALIDATION SET")
    print("="*70)
    print(f"Original validation size: {len(val_df)}")
    print(f"Capping at {VAL_DRY_RUN_MAX_N} samples (stratified by label)")
    val_df = make_validation_subset(val_df, VAL_DRY_RUN_MAX_N, VAL_SUBSET_SEED)
    print(f"Validation samples used: {len(val_df)}")
    print("\nValidation subset label distribution:")
    print(val_df["label"].value_counts().sort_index())
else:
    print("\n" + "="*70)
    print("[LOCAL RUN] USING FULL VALIDATION SET")
    print("="*70)
    print(f"Validation samples used: {len(val_df)}")

# Extract AST features if enabled
if USE_AST_FEATURES:
    print("\n" + "="*70)
    print("EXTRACTING AST FEATURES")
    print("="*70)
    
    # Extract AST features for training data
    train_ast_features = extract_ast_features_batch(train_df)
    print(f"Extracted {train_ast_features.shape[1]} AST features for training data")
    
    # Extract AST features for validation data
    val_ast_features = extract_ast_features_batch(val_df)
    print(f"Extracted {val_ast_features.shape[1]} AST features for validation data")
    
    # Save AST feature names for later use
    ast_feature_names = train_ast_features.columns.tolist()
    with open("ast_feature_names.pkl", "wb") as f:
        pickle.dump(ast_feature_names, f)
    
    print(f"AST feature names saved to ast_feature_names.pkl")

# Print statistics
print("\n" + "="*70)
print("TOKEN LENGTH STATISTICS")
print("="*70)

# Label distribution
print("\nLabel distribution (training):")
print(train_df["label"].value_counts().sort_index())

# Language distribution
if "language" in train_df.columns:
    print("\nLanguage distribution (training):")
    print(train_df["language"].value_counts().head(10))

# % of samples with tok_len > MAX_LENGTH
pct_long_train = (train_df["tok_len"] > MAX_LENGTH).sum() / len(train_df) * 100
pct_long_val = (val_df["tok_len"] > MAX_LENGTH).sum() / len(val_df) * 100
print(f"\n% of samples with tok_len > {MAX_LENGTH} (training): {pct_long_train:.2f}%")
print(f"% of samples with tok_len > {MAX_LENGTH} (validation): {pct_long_val:.2f}%")

# Stats specifically for Hybrid and Adversarial
for label_id, label_name in [(HYBRID_LABEL_ID, "Hybrid"), (ADV_LABEL_ID, "Adversarial")]:
    label_df = train_df[train_df["label"] == label_id]
    if len(label_df) > 0:
        pct_long = (label_df["tok_len"] > MAX_LENGTH).sum() / len(label_df) * 100
        median_len = label_df["tok_len"].median()
        print(f"\n{label_name} (training):")
        print(f"  Count: {len(label_df)}")
        print(f"  Median tok_len: {median_len:.0f}")
        print(f"  % with tok_len > {MAX_LENGTH}: {pct_long:.2f}%")

# ==================================================
# SECTION C — BALANCED 4-CLASS SUBSET BUILDER
# ==================================================

def make_balanced_subset_4class(df, total_n, seed, balance_by_language=True, balance_by_generator=False):
    """
    Create subset with optional balancing:
    1. If balancing enabled: STRICT balance by label (25% each for 4 classes)
    2. If balancing disabled: Simple random sampling preserving original distribution
    3. Optional soft balance by language (proportional with minimum floor)
    4. DO NOT hard-balance generators for small subsets
    5. Length-aware bias using tok_len bins (when balancing enabled)
    """
    np.random.seed(seed)
    
    # If no balancing, just do simple random sampling
    if not balance_by_language and not balance_by_generator:
        if len(df) <= total_n:
            return df.sample(frac=1, random_state=seed).reset_index(drop=True)
        else:
            return df.sample(n=total_n, random_state=seed).reset_index(drop=True)
    
    # Original balanced logic
    per_label = total_n // 4
    
    # Token length bins and weights for all 4 classes
    length_bins = [(0, 128), (128, 256), (256, 512), (512, 1024), (1024, float('inf'))]
    label_weights = {
        HUMAN_LABEL_ID: [1.2, 1.1, 1.0, 0.8, 0.6],      # prefer short/medium
        MACHINE_LABEL_ID: [1.0, 1.0, 1.0, 1.1, 1.2],    # neutral/slight long
        HYBRID_LABEL_ID: [0.6, 0.8, 1.0, 1.4, 1.8],     # prefer longer
        ADV_LABEL_ID: [0.7, 0.9, 1.1, 1.3, 1.5]         # prefer longer mildly
    }
    
    subset_rows = []
    
    for label_id in [HUMAN_LABEL_ID, MACHINE_LABEL_ID, HYBRID_LABEL_ID, ADV_LABEL_ID]:
        label_df = df[df["label"] == label_id].copy()
        
        if len(label_df) == 0:
            print(f"Warning: No samples found for label {label_id}")
            continue
        
        # Step 1: Soft balance by language (if enabled)
        sampled_from_groups = []
        remaining_budget = per_label
        
        if balance_by_language and "language" in label_df.columns:
            languages = label_df["language"].unique()
            num_languages = len(languages)
            min_per_lang = max(1, per_label // (num_languages * 2))  # Minimum floor
            
            for lang in languages:
                lang_df = label_df[label_df["language"] == lang]
                if len(lang_df) == 0:
                    continue
                
                # Proportional sampling with floor
                lang_proportion = len(lang_df) / len(label_df)
                lang_quota = max(min_per_lang, int(per_label * lang_proportion))
                lang_quota = min(lang_quota, len(lang_df), remaining_budget)
                
                if lang_quota > 0:
                    sampled = lang_df.sample(n=lang_quota, random_state=seed)
                    sampled_from_groups.append(sampled)
                    remaining_budget -= len(sampled)
        else:
            # No language balancing, sample directly
            pass
        
        # Step 2: Fill remaining with length-aware sampling
        if remaining_budget > 0:
            already_sampled_indices = set()
            for df_part in sampled_from_groups:
                already_sampled_indices.update(df_part.index)
            
            remaining_df = label_df[~label_df.index.isin(already_sampled_indices)]
            
            if len(remaining_df) > 0 and remaining_budget > 0:
                if "tok_len" in remaining_df.columns:
                    # Assign weights based on length bins
                    weights = np.zeros(len(remaining_df))
                    label_w = label_weights[label_id]
                    
                    for idx, row in remaining_df.iterrows():
                        tok_len = row["tok_len"]
                        for bin_idx, (low, high) in enumerate(length_bins):
                            if low <= tok_len < high:
                                weights[remaining_df.index.get_loc(idx)] = label_w[bin_idx]
                                break
                    
                    # Normalize weights
                    if weights.sum() > 0:
                        weights = weights / weights.sum()
                    else:
                        weights = np.ones(len(remaining_df)) / len(remaining_df)
                    
                    # Sample with weights
                    sample_size = min(remaining_budget, len(remaining_df))
                    if sample_size > 0:
                        sampled_indices = np.random.choice(
                            remaining_df.index,
                            size=sample_size,
                            replace=False,
                            p=weights
                        )
                        sampled_from_groups.append(remaining_df.loc[sampled_indices])
                else:
                    # Uniform sampling
                    sample_size = min(remaining_budget, len(remaining_df))
                    if sample_size > 0:
                        sampled_from_groups.append(remaining_df.sample(n=sample_size, random_state=seed))
        
        # Combine all sampled rows for this label
        if sampled_from_groups:
            subset_rows.append(pd.concat(sampled_from_groups, ignore_index=False))
    
    if not subset_rows:
        return df.head(0)
    
    result = pd.concat(subset_rows, ignore_index=True)
    return result.sample(frac=1, random_state=seed).reset_index(drop=True)

# Create subset (if enabled)
if USE_SUBSET:
    print("\n" + "="*70)
    print("CREATING TRAINING SUBSET (NO BALANCING)")
    print("="*70)
    
    balancing_type = "with balancing" if (BALANCE_BY_LANGUAGE or BALANCE_BY_GENERATOR) else "without balancing"
    print(f"Creating subset of {SUBSET_TOTAL_N} samples {balancing_type}...")
    
    train_df_subset = make_balanced_subset_4class(
        train_df, 
        SUBSET_TOTAL_N, 
        SUBSET_SEED,
        balance_by_language=BALANCE_BY_LANGUAGE,
        balance_by_generator=BALANCE_BY_GENERATOR
    )
    
    print(f"\nSubset created: {len(train_df_subset)} samples")
    print("\nSubset label distribution:")
    print(train_df_subset["label"].value_counts().sort_index())
    
    # Print median tok_len per label
    print("\nMedian tok_len per label:")
    for label_id in [HUMAN_LABEL_ID, MACHINE_LABEL_ID, HYBRID_LABEL_ID, ADV_LABEL_ID]:
        label_subset = train_df_subset[train_df_subset["label"] == label_id]
        if len(label_subset) > 0:
            median_len = label_subset["tok_len"].median()
            pct_long = (label_subset["tok_len"] > MAX_LENGTH).sum() / len(label_subset) * 100
            print(f"  Label {label_id}: median={median_len:.0f}, %>{MAX_LENGTH}={pct_long:.2f}%")
    
    # Print % of Hybrid/Adv >512 tokens
    for label_id, label_name in [(HYBRID_LABEL_ID, "Hybrid"), (ADV_LABEL_ID, "Adversarial")]:
        label_subset = train_df_subset[train_df_subset["label"] == label_id]
        if len(label_subset) > 0:
            pct_long = (label_subset["tok_len"] > MAX_LENGTH).sum() / len(label_subset) * 100
            print(f"\n{label_name} samples with tok_len > {MAX_LENGTH}: {pct_long:.2f}%")
    
    # Save subset
    train_df_subset.to_parquet(SAVE_SUBSET_PATH, index=False)
    print(f"\nSubset saved to {SAVE_SUBSET_PATH}")
    
    train_df = train_df_subset

# Extract AST features for subset if needed
if USE_AST_FEATURES and USE_SUBSET:
    print("\n" + "="*70)
    print("EXTRACTING AST FEATURES FOR SUBSET")
    print("="*70)
    train_ast_features = extract_ast_features_batch(train_df)
    print(f"Extracted {train_ast_features.shape[1]} AST features for training subset")

# ==================================================
# SECTION D — MULTI-VIEW CROPPING (NO WINDOWS)
# ==================================================

def tokenize_and_crop(df, tokenizer, max_length, crop_strategy="start_end", middle_crop_for=None):
    """
    Apply multi-view cropping to create training rows.
    
    If tok_len <= MAX_LENGTH: single view
    If tok_len > MAX_LENGTH:
        - start crop (first MAX_LENGTH tokens)
        - end crop (last MAX_LENGTH tokens)
        - optional middle crop ONLY for specified labels if crop_strategy == start_end_middle
    Each crop becomes its own training row with metadata.
    """
    if middle_crop_for is None:
        middle_crop_for = []
    
    expanded_rows = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Applying multi-view cropping"):
        code = row["code"]
        label = row["label"]
        tok_len = row.get("tok_len", 0)
        
        # Tokenize to get actual token IDs
        tokens = tokenizer(code, add_special_tokens=False, truncation=False, return_tensors=None)
        input_ids = tokens["input_ids"]
        actual_tok_len = len(input_ids)
        
        # Determine if we need middle crop
        label_name_map = {
            HUMAN_LABEL_ID: "human",
            MACHINE_LABEL_ID: "machine",
            HYBRID_LABEL_ID: "hybrid",
            ADV_LABEL_ID: "adversarial"
        }
        label_name = label_name_map.get(label, "").lower()
        use_middle = (crop_strategy == "start_end_middle" and
                      label_name in [x.lower() for x in middle_crop_for] and
                      actual_tok_len > max_length)
        
        if actual_tok_len <= max_length:
            # Single view
            expanded_rows.append({
                'code': code,
                'label': label,
                'view': 'single',
                'orig_id': idx
            })
        else:
            # Start crop
            start_tokens = input_ids[:max_length]
            start_code = tokenizer.decode(start_tokens, skip_special_tokens=False)
            expanded_rows.append({
                'code': start_code,
                'label': label,
                'view': 'start',
                'orig_id': idx
            })
            
            # End crop
            end_tokens = input_ids[-max_length:]
            end_code = tokenizer.decode(end_tokens, skip_special_tokens=False)
            expanded_rows.append({
                'code': end_code,
                'label': label,
                'view': 'end',
                'orig_id': idx
            })
            
            # Optional middle crop
            if use_middle:
                middle_start = (actual_tok_len - max_length) // 2
                middle_tokens = input_ids[middle_start:middle_start + max_length]
                middle_code = tokenizer.decode(middle_tokens, skip_special_tokens=False)
                expanded_rows.append({
                    'code': middle_code,
                    'label': label,
                    'view': 'middle',
                    'orig_id': idx
                })
    
    return pd.DataFrame(expanded_rows)

# ==================================================
# SECTION E — MODEL + TRAINER (4 CLASS WITH AST)
# ==================================================

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS  # 4-class: Human, Machine, Hybrid, Adversarial
).to(device)

print(f"Model: {MODEL_NAME}")
print(f"Device: {device}")
print(f"Number of labels: {NUM_LABELS} (Human={HUMAN_LABEL_ID}, Machine={MACHINE_LABEL_ID}, Hybrid={HYBRID_LABEL_ID}, Adversarial={ADV_LABEL_ID})")

# Apply multi-view cropping to training data
print("\n" + "="*70)
print("APPLYING MULTI-VIEW CROPPING TO TRAINING DATA")
print("="*70)
train_df_expanded = tokenize_and_crop(
    train_df,
    tokenizer,
    MAX_LENGTH,
    crop_strategy=CROP_STRATEGY,
    middle_crop_for=MIDDLE_CROP_FOR)

print(f"Original training samples: {len(train_df)}")
print(f"After multi-view expansion: {len(train_df_expanded)}")
print("\nView distribution:")
print(train_df_expanded["view"].value_counts())

# Compute class weights AFTER multi-view expansion
unique_labels = np.array(sorted(train_df_expanded["label"].unique()))
class_weights_array = compute_class_weight(
    'balanced',
    classes=unique_labels,
    y=train_df_expanded["label"].values)
class_weights = torch.tensor(class_weights_array, dtype=torch.float32).to(device)
print(f"\nClass weights (after expansion): {dict(zip(unique_labels, class_weights_array))}")

# Tokenization function (NO padding, NO return_tensors)
def tokenize_function(examples):
    return tokenizer(
        examples['code'],
        truncation=True,
        max_length=MAX_LENGTH
        # NO padding, NO return_tensors
    )

# Prepare datasets
train_dataset = Dataset.from_pandas(train_df_expanded[['code', 'label']])
val_dataset = Dataset.from_pandas(val_df[['code', 'label']])  # Keep 4-class for validation

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=['code'])
val_dataset = val_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=['code'])

train_dataset = train_dataset.rename_column('label', 'labels')
val_dataset = val_dataset.rename_column('label', 'labels')

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

print(f"\nTrain dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

# Train AST classifier if enabled
ast_classifier = None
ast_scaler = None

if USE_AST_FEATURES:
    print("\n" + "="*70)
    print("TRAINING AST CLASSIFIER")
    print("="*70)
    
    # Prepare AST features for training
    if USE_SUBSET:
        # Use subset AST features
        X_train_ast = train_ast_features.values
    else:
        # Use full training AST features
        X_train_ast = train_ast_features.values
    
    y_train_ast = train_df["label"].values
    
    # Prepare AST features for validation
    X_val_ast = val_ast_features.values
    y_val_ast = val_df["label"].values
    
    # Scale AST features
    ast_scaler = StandardScaler()
    X_train_ast_scaled = ast_scaler.fit_transform(X_train_ast)
    X_val_ast_scaled = ast_scaler.transform(X_val_ast)
    
    # Train AST classifier
    print("Training Random Forest on AST features...")
    ast_classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    ast_classifier.fit(X_train_ast_scaled, y_train_ast)
    
    # Evaluate AST classifier
    ast_val_pred = ast_classifier.predict(X_val_ast_scaled)
    ast_val_f1 = f1_score(y_val_ast, ast_val_pred, average='macro')
    print(f"AST Classifier Validation Macro F1: {ast_val_f1:.4f}")
    
    # Save AST components
    with open("ast_classifier.pkl", "wb") as f:
        pickle.dump(ast_classifier, f)
    with open("ast_scaler.pkl", "wb") as f:
        pickle.dump(ast_scaler, f)
    
    print("AST classifier and scaler saved")

# Compute metrics function for early stopping
def compute_metrics(eval_pred):
    """Compute metrics for evaluation during training."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Macro F1 (primary metric for early stopping)
    macro_f1 = f1_score(labels, predictions, average='macro', zero_division=0)
    
    # Additional metrics for logging
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1_per_class, _ = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )
    
    return {
        'eval_f1': macro_f1,
        'eval_accuracy': accuracy,
        'eval_precision_macro': np.mean(precision),
        'eval_recall_macro': np.mean(recall)
    }

# Custom Trainer with focal/weighted loss
class CustomTrainer(Trainer):
    def __init__(self, class_weights=None, use_focal_loss=False, focal_gamma=2.0, use_class_weights=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        self.use_class_weights = use_class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Pop labels before forward pass
        labels = inputs.pop("labels")
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.use_focal_loss:
            # Focal loss: ce = cross_entropy without weight
            ce = torch.nn.functional.cross_entropy(
                logits, labels, reduction="none"
            )
            pt = torch.exp(-ce)
            focal = ((1 - pt) ** self.focal_gamma) * ce
            
            # Apply class weights as alpha
            if self.class_weights is not None and self.use_class_weights:
                alpha = self.class_weights[labels]
                loss = (alpha * focal).mean()
            else:
                loss = focal.mean()
        else:
            # Weighted cross-entropy
            if self.class_weights is not None and self.use_class_weights:
                loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
            else:
                loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return (loss, outputs) if return_outputs else loss

# Training arguments with early stopping support
training_args_epoch1 = TrainingArguments(
    output_dir="./results",
    num_train_epochs=NUM_EPOCHS,  # Train for full epochs with early stopping
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    warmup_ratio=WARMUP_RATIO,
    max_grad_norm=MAX_GRAD_NORM,
    logging_steps=LOGGING_STEPS,
    evaluation_strategy=EVAL_STRATEGY if USE_EARLY_STOPPING else "no",
    eval_steps=EVAL_STEPS if USE_EARLY_STOPPING else None,
    save_strategy="epoch" if USE_EARLY_STOPPING else "steps",  # Match evaluation strategy
    save_total_limit=SAVE_TOTAL_LIMIT,
    load_best_model_at_end=LOAD_BEST_MODEL_AT_END if USE_EARLY_STOPPING else False,
    metric_for_best_model=METRIC_FOR_BEST_MODEL if USE_EARLY_STOPPING else None,
    greater_is_better=GREATER_IS_BETTER if USE_EARLY_STOPPING else None,
    fp16=FP16,
    report_to=[],
    remove_unused_columns=False,
    lr_scheduler_type="cosine"
)

# Create trainer with early stopping support
trainer = CustomTrainer(
    model=model,
    args=training_args_epoch1,
    train_dataset=train_dataset,
    eval_dataset=val_dataset if USE_EARLY_STOPPING else None,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics if USE_EARLY_STOPPING else None,
    class_weights=class_weights if USE_CLASS_WEIGHTS else None,
    use_focal_loss=USE_FOCAL_LOSS,
    focal_gamma=FOCAL_GAMMA,
    use_class_weights=USE_CLASS_WEIGHTS,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)] if USE_EARLY_STOPPING else None
)

print(f"\nTraining configuration:")
print(f"  Focal loss: {USE_FOCAL_LOSS} (gamma={FOCAL_GAMMA})")
print(f"  Class weights: {USE_CLASS_WEIGHTS}")
print(f"  FP16: {FP16}")
print(f"  Effective batch size: {PER_DEVICE_TRAIN_BATCH * GRAD_ACCUM}")
print(f"  Total epochs: {NUM_EPOCHS}")
print(f"  AST features: {USE_AST_FEATURES}")
if USE_AST_FEATURES:
    print(f"  AST feature weight: {AST_FEATURE_WEIGHT}")
    print(f"  Combination method: {COMBINE_METHOD}")

if USE_EARLY_STOPPING:
    print(f"  Early stopping: ENABLED")
    print(f"    Patience: {EARLY_STOPPING_PATIENCE}")
    print(f"    Evaluation strategy: {EVAL_STRATEGY}")
    if EVAL_STRATEGY == "epoch":
        print(f"    Evaluation: Once per epoch on {len(val_df)} validation samples")
    else:
        print(f"    Evaluation steps: {EVAL_STEPS}")
    print(f"    Metric for best model: {METRIC_FOR_BEST_MODEL}")
    print(f"    Load best model at end: {LOAD_BEST_MODEL_AT_END}")
else:
    print(f"  Early stopping: DISABLED")

# Train the model without early stopping
print("\n" + "="*70)
print("TRAINING TRANSFORMER MODEL")
print("="*70)
trainer.train()
trainer.save_model()
tokenizer.save_pretrained("./results")
print("Training completed!")

# ==================================================
# SECTION F — MULTI-VIEW INFERENCE FUNCTIONS WITH AST
# ==================================================

@torch.no_grad()
def predict_multiview_with_ast(code, model, tokenizer, ast_extractor, ast_classifier, ast_scaler, 
                               max_length, device, use_ast=True, ast_weight=0.3):
    """
    Run multi-view inference on a single code snippet with AST features.
    
    For short code (tok_len <= MAX_LENGTH): single forward pass
    For long code (tok_len > MAX_LENGTH): start + end crops
    
    Returns aggregated logits (4-class) combining transformer and AST predictions.
    """
    # Transformer prediction
    transformer_logits = predict_multiview_transformformer(code, model, tokenizer, max_length, device)
    
    if not use_ast or ast_classifier is None or ast_scaler is None:
        return transformer_logits
    
    # AST prediction
    try:
        ast_features = ast_extractor.extract_features(code)
        ast_feature_vector = np.array([ast_features[name] for name in ast_feature_names])
        ast_feature_scaled = ast_scaler.transform([ast_feature_vector])
        ast_probs = ast_classifier.predict_proba(ast_feature_scaled)[0]
        ast_logits = torch.log(torch.tensor(ast_probs + 1e-8)).to(device)
        
        # Combine predictions
        combined_logits = (1 - ast_weight) * transformer_logits + ast_weight * ast_logits
        
        return combined_logits
    except Exception as e:
        print(f"AST prediction failed: {e}")
        return transformer_logits

@torch.no_grad()
def predict_multiview_transformformer(code, model, tokenizer, max_length, device):
    """
    Run multi-view transformer inference on a single code snippet.
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

# ==================================================
# SECTION G — HARD NEGATIVE MINING (DISABLED WITH EARLY STOPPING)
# ==================================================

# Note: Hard negative mining is disabled when using early stopping
# Early stopping provides automatic regularization and prevents overfitting
# If you want to use hard negative mining, set USE_EARLY_STOPPING = False
if not USE_EARLY_STOPPING and DO_HARD_MINING and NUM_EPOCHS > 1:
    print("\n" + "="*70)
    print("HARD NEGATIVE MINING (EARLY STOPPING DISABLED)")
    print("="*70)
    print("Warning: Hard negative mining is not compatible with early stopping.")
    print("Set USE_EARLY_STOPPING = False to enable hard negative mining.")
else:
    if USE_EARLY_STOPPING:
        print("\n" + "="*70)
        print("HARD NEGATIVE MINING SKIPPED (EARLY STOPPING ENABLED)")
        print("="*70)
        print("Early stopping provides automatic regularization.")

# ==================================================
# SECTION H — MANUAL EVALUATION (MULTI-VIEW + AST AWARE)
# ==================================================

def evaluate_4class_multiview_ast(val_df, model, tokenizer, ast_extractor, ast_classifier, ast_scaler, 
                                  device, save_mistakes=True, mistakes_n=200):
    """
    Evaluate on validation set with multi-view inference and AST features.
    """
    model.eval()
    
    predictions_4class = []
    true_labels = []
    all_stats = []  # For error analysis
    
    start_time = time.time()
    
    print("\n" + "="*70)
    print("RUNNING MULTI-VIEW + AST INFERENCE ON VALIDATION SET")
    print("="*70)
    
    for idx, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Evaluating"):
        code = row["code"]
        true_label = row["label"]
        tok_len = row.get("tok_len", 0)
        
        # Run multi-view inference with AST
        logits = predict_multiview_with_ast(
            code, model, tokenizer, ast_extractor, ast_classifier, ast_scaler, 
            MAX_LENGTH, device, USE_AST_FEATURES, AST_FEATURE_WEIGHT
        )
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        pred_4class = int(np.argmax(probs))
        
        predictions_4class.append(pred_4class)
        true_labels.append(true_label)
        
        # Store stats for error analysis
        all_stats.append({
            'idx': idx,
            'true_label': true_label,
            'pred_label': pred_4class,
            'probs': probs.tolist(),
            'max_prob': float(np.max(probs)),
            'tok_len': tok_len,
            'code_preview': code[:200] if code else ""
        })
    
    elapsed = time.time() - start_time
    
    # Compute metrics
    predictions_4class = np.array(predictions_4class)
    true_labels = np.array(true_labels)
    
    # Macro F1 (primary metric)
    macro_f1 = f1_score(true_labels, predictions_4class, average='macro', zero_division=0)
    
    # Per-class F1
    unique_labels = sorted(np.unique(np.concatenate([true_labels, predictions_4class])))
    per_class_f1 = f1_score(true_labels, predictions_4class, labels=unique_labels, average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions_4class, labels=unique_labels)
    
    # Classification report
    label_names = {
        HUMAN_LABEL_ID: "Human",
        MACHINE_LABEL_ID: "Machine",
        HYBRID_LABEL_ID: "Hybrid",
        ADV_LABEL_ID: "Adversarial"
    }
    target_names = [label_names.get(label, f"Class_{label}") for label in unique_labels]
    report = classification_report(true_labels, predictions_4class, labels=unique_labels,
                                   target_names=target_names, zero_division=0)
    
    # Print results
    print("\n" + "="*70)
    print("SNIPPET-LEVEL EVALUATION RESULTS (4-CLASS + AST)")
    print("="*70)
    print(f"\nMacro F1 (Primary): {macro_f1:.4f}")
    print(f"\nPer-class F1 Scores:")
    for i, label in enumerate(unique_labels):
        label_name = label_names.get(label, f"Class_{label}")
        print(f"   {label_name} ({label}): {per_class_f1[i]:.4f}")
    
    print(f"\nConfusion Matrix:")
    print("Actual \\ Predicted ->", end="")
    for label in unique_labels:
        print(f"{label:>8}", end="")
    print()
    for i, label in enumerate(unique_labels):
        label_name = label_names.get(label, f"Class_{label}")
        print(f"{label_name:15s}", end="")
        for j in range(len(unique_labels)):
            print(f"{cm[i][j]:>8}", end="")
        print()
    
    print(f"\nClassification Report:")
    print(report)
    
    # F1 by length buckets (if enabled)
    if REPORT_LENGTH_BUCKETS:
        print(f"\nMacro F1 by Length Buckets:")
        length_buckets = [(0, 128), (129, 256), (257, 512), (513, 1024), (1025, float('inf'))]
        for low, high in length_buckets:
            mask = np.array([s['tok_len'] for s in all_stats])
            if high == float('inf'):
                mask = (mask >= low)
            else:
                mask = (mask >= low) & (mask <= high)
            
            if mask.sum() > 0:
                bucket_true = true_labels[mask]
                bucket_pred = predictions_4class[mask]
                bucket_f1 = f1_score(bucket_true, bucket_pred, average='macro', zero_division=0)
                print(f"   {low}-{high if high != float('inf') else '+'} tokens: {bucket_f1:.4f} (n={mask.sum()})")
    
    # Runtime stats
    print(f"\nRuntime Stats:")
    print(f"   Total snippets: {len(val_df)}")
    print(f"   Time: {elapsed:.2f}s")
    print(f"   Snippets/sec: {len(val_df)/elapsed:.2f}")
    
    # Save hardest mistakes
    if save_mistakes:
        mistakes = []
        for stat in all_stats:
            if stat['true_label'] != stat['pred_label']:
                mistakes.append(stat)
        
        # Sort by confidence (lower confidence = harder mistake)
        mistakes.sort(key=lambda x: x['max_prob'])
        
        if mistakes:
            mistakes_df = pd.DataFrame(mistakes[:mistakes_n])
            mistakes_df.to_csv("validation_mistakes_ast.csv", index=False)
            print(f"\nSaved {len(mistakes_df)} hardest mistakes to validation_mistakes_ast.csv")
    
    return {
        'macro_f1': macro_f1,
        'per_class_f1': dict(zip(unique_labels, per_class_f1)),
        'confusion_matrix': cm,
        'predictions': predictions_4class,
        'true_labels': true_labels,
        'all_stats': all_stats
    }

# Load AST feature names if using AST
if USE_AST_FEATURES:
    try:
        with open("ast_feature_names.pkl", "rb") as f:
            ast_feature_names = pickle.load(f)
        print(f"Loaded {len(ast_feature_names)} AST feature names")
    except FileNotFoundError:
        print("Warning: AST feature names not found, using empty list")
        ast_feature_names = []

# Run evaluation on validation set
print("\n" + "="*70)
print("FINAL VALIDATION EVALUATION WITH AST")
print("="*70)
eval_results = evaluate_4class_multiview_ast(
    val_df, 
    model, 
    tokenizer, 
    ast_extractor, 
    ast_classifier, 
    ast_scaler,
    device,
    save_mistakes=SAVE_MISTAKES,
    mistakes_n=MISTAKES_N
)

# ==================================================
# SECTION I — TEST PREDICTION + SUBMISSION (LOCAL VERSION WITH AST)
# ==================================================

def predict_test_and_write_submission_ast(test_parquet_path, output_csv_path, model, tokenizer, 
                                         ast_extractor, ast_classifier, ast_scaler, device):
    """
    Stream test set, run multi-view inference with AST, write 4-class predictions to submission.
    LOCAL VERSION: Uses local file paths
    """
    model.eval()
    
    # Check if test file exists
    if not os.path.exists(test_parquet_path):
        print(f"❌ Test file not found: {test_parquet_path}")
        print("Please ensure the test file is in the current directory.")
        return
    
    # Load test data
    try:
        test_df = pd.read_parquet(test_parquet_path)
        print(f"✅ Loaded test data: {len(test_df)} samples")
        
        # Check language distribution in test data
        if "language" in test_df.columns:
            print("\nTest language distribution:")
            print(test_df["language"].value_counts())
            
            # Count unknown languages
            unknown_count = (test_df["language"] == "Unknown").sum()
            if unknown_count > 0:
                print(f"\n⚠️  Found {unknown_count} samples with 'Unknown' language")
                print("AST features will help classify these language-agnostic samples")
        
    except Exception as e:
        print(f"❌ Error loading test file: {e}")
        return
    
    print("\n" + "="*70)
    print("PREDICTING ON TEST SET WITH AST FEATURES")
    print("="*70)
    
    with open(output_csv_path, "w") as f:
        f.write("ID,prediction\n")
        
        count = 0
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Predicting"):
            code = row["code"]
            ex_id = row["ID"]
            
            # Run multi-view inference with AST
            logits = predict_multiview_with_ast(
                code, model, tokenizer, ast_extractor, ast_classifier, ast_scaler, 
                MAX_LENGTH, device, USE_AST_FEATURES, AST_FEATURE_WEIGHT
            )
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
        print(f"AST features enabled: {USE_AST_FEATURES}")
        if USE_AST_FEATURES:
            print(f"AST feature weight: {AST_FEATURE_WEIGHT}")
        print("="*70)
    else:
        print(f"\n⚠️  WARNING: File {output_csv_path} was not created!")

# Generate submission file on test set (if enabled)
if GENERATE_SUBMISSION:
    predict_test_and_write_submission_ast(
        TEST_PARQUET_PATH, SUBMISSION_OUTPUT_PATH, model, tokenizer, 
        ast_extractor, ast_classifier, ast_scaler, device
    )
else:
    print("\n" + "="*70)
    print("SUBMISSION GENERATION SKIPPED")
    print("="*70)
    print("To generate submission_ast.csv, set GENERATE_SUBMISSION = True in the config section")
    print(f"Test parquet path: {TEST_PARQUET_PATH}")
    print(f"Output path: {SUBMISSION_OUTPUT_PATH}")

# ==================================================
# SECTION J — FINAL RUN MODES (LOCAL VERSION WITH AST)
# ==================================================

## LOCAL RUN MODE:
# - SUBSET_TOTAL_N = 20_000 (reduced for local testing)
# - NUM_EPOCHS = 2
# - HARD NEGATIVE MINING ENABLED (small scale: ~1000-2000 mined)
# - VALIDATION LIMITED to 2_000 samples
# - AST FEATURES ENABLED for language-agnostic classification
# 
# Goal:
#   - Test the complete pipeline locally with AST enhancement
#   - Verify training and evaluation work with AST features
#   - Handle unknown programming languages in test data
#   - Debug any issues before full-scale run
#
## To run with full data locally:
# - Set SUBSET_TOTAL_N = 200_000 (or available data size)
# - Set USE_FULL_VALIDATION = True
# - Increase batch sizes if you have enough GPU memory
# - Set GENERATE_SUBMISSION = True if you have test data
#
## IMPORTANT LOCAL NOTES:
# - Ensure you have the required parquet files in the current directory
# - The script will check for required files before starting
# - Model and results will be saved to "./results/" directory
# - Validation mistakes will be saved to "validation_mistakes_ast.csv"
# - AST components will be saved as pickle files
# - All paths are relative to the current working directory
#
## AST ENHANCEMENT NOTES:
# - AST features provide language-agnostic structural analysis
# - Helps classify code from unknown programming languages
# - Combined with transformer features for robust classification
# - Particularly useful for the 84,571 "Unknown" language samples in test data

print("\n" + "="*70)
print("LOCAL EXECUTION WITH AST COMPLETED")
print("="*70)
print("Files created:")
print("  - ./results/ (model checkpoints and tokenizer)")
print("  - train_subset_4class_ast.parquet (training subset)")
if SAVE_MISTAKES:
    print("  - validation_mistakes_ast.csv (error analysis)")
if GENERATE_SUBMISSION:
    print("  - submission_ast.csv (test predictions with AST)")
if USE_AST_FEATURES:
    print("  - ast_classifier.pkl (trained AST classifier)")
    print("  - ast_scaler.pkl (AST feature scaler)")
    print("  - ast_feature_names.pkl (AST feature names)")
print("="*70)
