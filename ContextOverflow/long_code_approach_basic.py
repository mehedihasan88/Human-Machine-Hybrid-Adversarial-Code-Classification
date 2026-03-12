#!/usr/bin/env python3
"""Task C: Hybrid Code Detection with Direct 4-Class Classification - Local Version

Core Strategy:
- Train 4-class classifier directly (Human, Machine, Hybrid, Adversarial)
- NO sliding-window inference
- Handle long code via multi-view cropping (start / end / optional middle)
- Improve rare classes using balanced subset sampling, focal loss, multi-view inference aggregation, and hard negative mining

LOCAL VERSION MODIFICATIONS:
- Removed Kaggle-specific paths
- Updated data paths to use local files
- Disabled submission generation by default
- Added local file handling
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
    EarlyStoppingCallback,
    TrainerCallback)
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import warnings
import json
import time
from collections import defaultdict

warnings.filterwarnings("ignore")

# ==================================================
# SECTION A — CONFIG (LOCAL VERSION)
# ==================================================

# ---------------------------------------------------------------------------
# RUN MODE: set RUN_ON_KAGGLE = True for Kaggle, False for local
# ---------------------------------------------------------------------------
RUN_ON_KAGGLE = False
if RUN_ON_KAGGLE:
    DATA_DIR = "/kaggle/input/competitions/sem-eval-2026-task-13-subtask-c/Task_C"
    OUTPUT_DIR = "/kaggle/working"
    TRAIN_FILE = "train.parquet"
    VAL_FILE = "validation.parquet"
    TEST_FILE = "test.parquet"
else:
    DATA_DIR = "."
    OUTPUT_DIR = "."
    TRAIN_FILE = "task_c_training_set_1.parquet"
    VAL_FILE = "task_c_validation_set.parquet"
    TEST_FILE = "test.parquet"
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

# ---------------------------------------------------------------------------
# QUICK TEST RUN: set QUICK_TEST = True for 2k train / 1k val / 1k test
# For full run: QUICK_TEST = False (uses SUBSET_TOTAL_N, VAL_SIZE, TEST_SIZE below)
# ---------------------------------------------------------------------------
QUICK_TEST = False                # True = 2k train, 1k val, 1k test; False = use values below
if QUICK_TEST:
    _train_n = 2_000
    _val_n = 1_000
    _test_n = 1_000
else:
    _train_n = 200_000
    _val_n = 40_000
    _test_n = 40_000

# Data / Subset - LENGTH-BASED BUCKET SAMPLING
USE_SUBSET = True
SUBSET_TOTAL_N = _train_n       # Total samples for subset (2k for quick test, 200k full)
SUBSET_SEED = 42                 # Random seed for reproducibility
# Subset distribution:
#   - 20% short snippets (<=512 tokens)
#   - 80% long snippets (>512 tokens), evenly distributed across 512-token buckets
#   - Buckets: [512, 1024), [1024, 1536), [1536, 2048), [2048, 2560), ...
SAVE_SUBSET_PATH = os.path.join(OUTPUT_DIR, "train_subset_4class.parquet")

# Validation control
USE_FULL_VALIDATION = True     # Keep True so val/test use VAL_SIZE / TEST_SIZE (1k each when QUICK_TEST)
VAL_DRY_RUN_MAX_N = 20_000       # Used only when USE_FULL_VALIDATION = False
VAL_SUBSET_SEED = 123
# Validation data split: val set and test set, same distribution as train (20% short, 80% long)
VAL_SIZE = _val_n               # 1k for quick test, 40k full
TEST_SIZE = _test_n             # 1k for quick test, 40k full
VAL_TEST_SPLIT_SEED = 42

# Labels (4-class direct)
HUMAN_LABEL_ID = 0
MACHINE_LABEL_ID = 1
HYBRID_LABEL_ID = 2
ADV_LABEL_ID = 3
NUM_LABELS = 4

# Token 
MAX_LENGTH = 512
USE_MULTIVIEW = False                      # Enable/disable multi-view cropping
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
LR = 2e-5                         # Slightly higher for full dataset
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
SAVE_INFERENCE = True             # Save all inference results to CSV

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

# Test prediction / submission (paths set from RUN_ON_KAGGLE above)
GENERATE_SUBMISSION = False       # Enable for full dataset testing
TEST_PARQUET_PATH = os.path.join(DATA_DIR, TEST_FILE)
SUBMISSION_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "submission.csv")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"[{'KAGGLE' if RUN_ON_KAGGLE else 'LOCAL'}] Data: {DATA_DIR}  |  Output: {OUTPUT_DIR}")
if QUICK_TEST:
    print(f"[QUICK TEST MODE] Train={_train_n:,}, Val={_val_n:,}, Test={_test_n:,}")
else:
    print(f"[FULL RUN] Train={_train_n:,}, Val={_val_n:,}, Test={_test_n:,}")

# ==================================================
# SECTION B — DATA LOADING + TOKEN LENGTH STATS (LOCAL VERSION)
# ==================================================

# Check if required data files exist (in DATA_DIR)
def check_data_files():
    """Check if required data files exist in DATA_DIR."""
    required_files = [TRAIN_FILE, VAL_FILE]
    required_paths = [os.path.join(DATA_DIR, f) for f in required_files]
    missing = [p for p in required_paths if not os.path.exists(p)]
    if missing:
        print("❌ Missing required data files:")
        for p in missing:
            print(f"   - {p}")
        print(f"\nExpected in: {DATA_DIR}")
        return False
    print("✅ All required data files found!")
    return True

# Check for required files
if not check_data_files():
    print("\n⚠️  Exiting due to missing data files.")
    if RUN_ON_KAGGLE and os.path.exists("/kaggle/input"):
        _parent = os.path.dirname(DATA_DIR)
        if os.path.exists(_parent):
            print(f"Contents of {_parent}: {os.listdir(_parent)}")
    raise SystemExit(1)

# Load train and validation
try:
    train_path = os.path.join(DATA_DIR, TRAIN_FILE)
    val_path = os.path.join(DATA_DIR, VAL_FILE)
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    print(f"✅ Loaded train: {train_path}")
    print(f"✅ Loaded validation: {val_path}")
except Exception as e:
    print(f"❌ Error loading data files: {e}")
    raise SystemExit(1)

print("="*70)
print("ORIGINAL 4-CLASS DISTRIBUTIONS")
print("="*70)
print("\nTraining set:")
print(train_df["label"].value_counts().sort_index())
print("\nValidation set:")
print(val_df["label"].value_counts().sort_index())

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


# Same ratio as train set: 20% short (<=512 tokens), 80% long (>512 tokens)
# Long: same bucket structure as train — evenly across [512,1024), [1024,1536), ...
VAL_TEST_SHORT_RATIO = 0.20
VAL_TEST_LONG_RATIO = 0.80
LONG_BUCKET_START = 512
LONG_BUCKET_SIZE = 512


def sample_long_with_buckets(long_df, long_target, seed, stratify_by_label=True):
    """
    Sample long_target rows from long_df using the same bucket logic as train:
    - Buckets: [512, 1024), [1024, 1536), [1536, 2048), ...
    - Distribute long_target evenly across non-empty buckets
    - Within each bucket: sample up to bucket target (stratified by label if requested)
    """
    if len(long_df) == 0 or long_target <= 0:
        return pd.DataFrame()
    if len(long_df) <= long_target:
        return long_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    bucket_start = LONG_BUCKET_START
    bucket_size = LONG_BUCKET_SIZE
    max_tok_len = long_df["tok_len"].max()
    num_buckets = max(1, int(np.ceil((max_tok_len - bucket_start) / bucket_size)))

    long_buckets = []
    for i in range(num_buckets):
        low = bucket_start + i * bucket_size
        high = bucket_start + (i + 1) * bucket_size
        bucket_df = long_df[(long_df["tok_len"] >= low) & (long_df["tok_len"] < high)]
        if len(bucket_df) > 0:
            long_buckets.append({"range": (low, high), "df": bucket_df})

    if not long_buckets:
        return pd.DataFrame()

    per_bucket_target = long_target // len(long_buckets)
    remainder = long_target % len(long_buckets)
    sampled_list = []
    for i, info in enumerate(long_buckets):
        bucket_df = info["df"]
        bucket_target = per_bucket_target + (1 if i < remainder else 0)
        n_take = min(bucket_target, len(bucket_df))
        if n_take <= 0:
            continue
        if stratify_by_label:
            sub = make_validation_subset(bucket_df, n_take, seed + i)
        else:
            sub = bucket_df.sample(n=n_take, random_state=seed + i)
        sampled_list.append(sub)
    if not sampled_list:
        return pd.DataFrame()
    return pd.concat(sampled_list, ignore_index=True)


def split_validation_into_val_and_test(full_val_df, val_size, test_size, seed):
    """
    Split validation data into validation set (val_size) and test set (test_size).
    Same type of logical ratio as the train set:
    - 20% short (<=512 tokens), 80% long (>512 tokens) in each set
    - Short: stratified by label
    - Long: evenly distributed across 512-token buckets [512,1024), [1024,1536), ... (stratified by label within buckets)
    """
    np.random.seed(seed)
    short_df = full_val_df[full_val_df["tok_len"] <= MAX_LENGTH].copy()
    long_df = full_val_df[full_val_df["tok_len"] > MAX_LENGTH].copy()

    short_per_set = int(val_size * VAL_TEST_SHORT_RATIO)   # 8k per set for 40k
    long_per_set = val_size - short_per_set                 # 32k per set for 40k
    short_total = short_per_set * 2
    long_total = long_per_set * 2

    print(f"Target distribution (same as train): {VAL_TEST_SHORT_RATIO*100:.0f}% short (<=512), {VAL_TEST_LONG_RATIO*100:.0f}% long (>512)")
    print(f"  Long: evenly across 512-token buckets [512,1024), [1024,1536), ...")
    print(f"  Per set: {short_per_set:,} short, {long_per_set:,} long")
    print(f"  Available: {len(short_df):,} short, {len(long_df):,} long")

    # If not enough short or long, use all available and adjust the other bucket so each set stays balanced
    if len(short_df) < short_total:
        print(f"  ⚠️  Using all {len(short_df):,} short (target was {short_total:,})")
        short_total = len(short_df)
        short_per_set = short_total // 2
        long_per_set = val_size - short_per_set
        long_total = long_per_set * 2
    if len(long_df) < long_total:
        print(f"  ⚠️  Using all {len(long_df):,} long (target was {long_total:,})")
        long_total = len(long_df)
        long_per_set = long_total // 2
        short_per_set = min(val_size - long_per_set, short_total // 2) if short_total > 0 else 0
        short_total = short_per_set * 2
        if short_per_set * 2 > len(short_df):
            short_per_set = len(short_df) // 2
            short_total = short_per_set * 2

    # Sample short: stratified by label, then split into val/test
    if short_total > 0 and len(short_df) > 0:
        short_sampled = make_validation_subset(short_df, short_total, seed)
        val_short, test_short = train_test_split(
            short_sampled,
            test_size=short_per_set,
            stratify=short_sampled["label"],
            random_state=seed,
            shuffle=True
        )
    else:
        val_short = pd.DataFrame()
        test_short = pd.DataFrame()

    # Sample long: same bucket logic as train (even across 512-token buckets), stratified by label within buckets, then split into val/test
    if long_total > 0 and len(long_df) > 0:
        long_sampled = sample_long_with_buckets(long_df, long_total, seed + 1, stratify_by_label=True)
        if len(long_sampled) > 0:
            # If we got fewer than 2*long_per_set (e.g. sparse buckets), split in half; else give long_per_set to test
            test_n = min(long_per_set, len(long_sampled) // 2) if len(long_sampled) < 2 * long_per_set else long_per_set
            if len(long_sampled) < long_total:
                print(f"  ⚠️  Long sampled {len(long_sampled):,} (target {long_total:,}); splitting for val/test")
            val_long, test_long = train_test_split(
                long_sampled,
                test_size=test_n,
                stratify=long_sampled["label"],
                random_state=seed,
                shuffle=True
            )
        else:
            val_long = pd.DataFrame()
            test_long = pd.DataFrame()
    else:
        val_long = pd.DataFrame()
        test_long = pd.DataFrame()

    val_df = pd.concat([val_short, val_long], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = pd.concat([test_short, test_long], ignore_index=True).sample(frac=1, random_state=seed + 1).reset_index(drop=True)

    return val_df, test_df

# Compute token lengths
print("\n" + "="*70)
print("COMPUTING TOKEN LENGTHS")
print("="*70)
train_df = compute_token_lengths(train_df, temp_tokenizer)
print(f"Train set: {len(train_df):,} samples (tok_len computed)")
val_full_df = compute_token_lengths(val_df.copy(), temp_tokenizer)
print(f"Full validation (before split): {len(val_full_df):,} samples (tok_len computed)")

# Split validation data into validation set and test set, same distribution as train
print("\n" + "="*70)
print(f"SPLITTING VALIDATION INTO VAL ({VAL_SIZE:,}) AND TEST ({TEST_SIZE:,})")
print("="*70)
val_df, test_df = split_validation_into_val_and_test(
    val_full_df, VAL_SIZE, TEST_SIZE, VAL_TEST_SPLIT_SEED
)
print(f"Validation set: {len(val_df):,} samples")
print(f"Test set: {len(test_df):,} samples")
for name, df in [("Validation", val_df), ("Test", test_df)]:
    if len(df) > 0:
        n_short = (df["tok_len"] <= MAX_LENGTH).sum()
        n_long = (df["tok_len"] > MAX_LENGTH).sum()
        pct_s = n_short / len(df) * 100
        pct_l = n_long / len(df) * 100
        print(f"  {name}: {n_short:,} short ({pct_s:.1f}%), {n_long:,} long ({pct_l:.1f}%) [target 20%/80%]")
print("\nValidation set label distribution (should match train):")
print(val_df["label"].value_counts().sort_index())
print("\nTest set label distribution (should match train):")
print(test_df["label"].value_counts().sort_index())

# Apply validation subsetting (if needed, for local quick runs)
if not USE_FULL_VALIDATION:
    print("\n" + "="*70)
    print("[LOCAL RUN] LIMITING VALIDATION SET")
    print("="*70)
    print(f"Validation size before cap: {len(val_df)}")
    print(f"Capping at {VAL_DRY_RUN_MAX_N} samples (stratified by label)")
    val_df = make_validation_subset(val_df, VAL_DRY_RUN_MAX_N, VAL_SUBSET_SEED)
    print(f"Validation samples used: {len(val_df)}")
    print("\nValidation subset label distribution:")
    print(val_df["label"].value_counts().sort_index())
else:
    print("\n" + "="*70)
    print(f"[LOCAL RUN] USING FULL VALIDATION SET ({len(val_df):,} samples)")
    print("="*70)
    print(f"Validation samples used: {len(val_df):,}")

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

# Short/long ratio (target 20% short, 80% long for all sets)
# Note: train_df here is still full train; subset is created in Section C below
n_short_train = (train_df["tok_len"] <= MAX_LENGTH).sum()
n_long_train = (train_df["tok_len"] > MAX_LENGTH).sum()
pct_short_train = n_short_train / len(train_df) * 100
pct_long_train = n_long_train / len(train_df) * 100
n_short_val = (val_df["tok_len"] <= MAX_LENGTH).sum()
n_long_val = (val_df["tok_len"] > MAX_LENGTH).sum()
pct_short_val = n_short_val / len(val_df) * 100
pct_long_val = n_long_val / len(val_df) * 100
n_short_test = (test_df["tok_len"] <= MAX_LENGTH).sum()
n_long_test = (test_df["tok_len"] > MAX_LENGTH).sum()
pct_short_test = n_short_test / len(test_df) * 100
pct_long_test = n_long_test / len(test_df) * 100

print("\n" + "-"*70)
print("TRAIN / VALIDATION / TEST — LENGTH RATIO CHECK (target 20% short, 80% long)")
print("-"*70)
print(f"  {'Set':<12} {'Total':>10} {'Short':>10} {'Long':>10} {'Short%':>8} {'Long%':>8}")
print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
print(f"  {'Train':<12} {len(train_df):>10,} {n_short_train:>10,} {n_long_train:>10,} {pct_short_train:>7.1f}% {pct_long_train:>7.1f}%  (full train; subset below)")
print(f"  {'Validation':<12} {len(val_df):>10,} {n_short_val:>10,} {n_long_val:>10,} {pct_short_val:>7.1f}% {pct_long_val:>7.1f}%")
print(f"  {'Test':<12} {len(test_df):>10,} {n_short_test:>10,} {n_long_test:>10,} {pct_short_test:>7.1f}% {pct_long_test:>7.1f}%")
print("-"*70)

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
# SECTION C — LENGTH-BASED BUCKET SUBSET BUILDER
# ==================================================

def make_length_bucket_subset(df, total_n, seed):
    """
    Create subset based on token length buckets:
    1. 20% from short snippets (<=512 tokens)
    2. 80% from long snippets (>512 tokens), evenly distributed across 512-token buckets
       Buckets: [512, 1024), [1024, 1536), [1536, 2048), [2048, 2560), ...
    3. Shuffle final dataset for randomness and reproducibility
    """
    np.random.seed(seed)
    
    if "tok_len" not in df.columns:
        raise ValueError("DataFrame must have 'tok_len' column. Run compute_token_lengths first.")
    
    # Calculate target sizes
    short_target = int(total_n * 0.20)  # 20% short
    long_target = total_n - short_target  # 80% long
    
    print(f"Target distribution:")
    print(f"  Short (<=512 tokens): {short_target:,} samples (20%)")
    print(f"  Long (>512 tokens): {long_target:,} samples (80%)")
    
    # Split into short and long (no overlap!)
    short_df = df[df["tok_len"] <= 512].copy()
    long_df = df[df["tok_len"] > 512].copy()
    
    print(f"\nAvailable samples:")
    print(f"  Short (<=512 tokens): {len(short_df):,} samples")
    print(f"  Long (>512 tokens): {len(long_df):,} samples")
    
    # Sample from short bucket
    if len(short_df) == 0:
        print("⚠️  Warning: No short samples (<=512 tokens) found!")
        short_sampled = pd.DataFrame()
    elif len(short_df) <= short_target:
        print(f"  Using all {len(short_df):,} short samples (less than target {short_target:,})")
        short_sampled = short_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    else:
        short_sampled = short_df.sample(n=short_target, random_state=seed).reset_index(drop=True)
        print(f"  Sampled {len(short_sampled):,} from short bucket")
    
    # Create long buckets: [512, 1024), [1024, 1536), [1536, 2048), [2048, 2560), ...
    long_buckets = []
    bucket_start = 512
    bucket_size = 512
    
    # Find maximum token length to determine number of buckets
    if len(long_df) > 0:
        max_tok_len = long_df["tok_len"].max()
        num_buckets = int(np.ceil((max_tok_len - bucket_start) / bucket_size))
        
        print(f"\nLong bucket structure:")
        print(f"  Number of buckets: {num_buckets}")
        print(f"  Bucket size: {bucket_size} tokens")
        
        # Group samples into buckets
        for i in range(num_buckets):
            bucket_low = bucket_start + i * bucket_size
            bucket_high = bucket_start + (i + 1) * bucket_size
            
            bucket_df = long_df[(long_df["tok_len"] >= bucket_low) & (long_df["tok_len"] < bucket_high)]
            
            if len(bucket_df) > 0:
                long_buckets.append({
                    'range': (bucket_low, bucket_high),
                    'df': bucket_df,
                    'count': len(bucket_df)
                })
                print(f"  Bucket [{bucket_low:5d}, {bucket_high:5d}): {len(bucket_df):,} samples")
        
        # Distribute long_target evenly across non-empty buckets
        num_non_empty_buckets = 1.5
        if num_non_empty_buckets == 0:
            print("⚠️  Warning: No long samples (>512 tokens) found in any bucket!")
            long_sampled = pd.DataFrame()
        else:
            per_bucket_target = long_target // num_non_empty_buckets
            remainder = long_target % num_non_empty_buckets
            
            print(f"\nSampling from long buckets:")
            print(f"  Target per bucket: {per_bucket_target:,} samples")
            if remainder > 0:
                print(f"  First {remainder} buckets get 1 extra sample")
            
            long_sampled_list = []
            for i, bucket_info in enumerate(long_buckets):
                bucket_df = bucket_info['df']
                bucket_range = bucket_info['range']
                
                # Add remainder to first buckets
                bucket_target = per_bucket_target + (1 if i < remainder else 0)
                
                if len(bucket_df) <= bucket_target:
                    sampled = bucket_df.sample(frac=1, random_state=seed + i).reset_index(drop=True)
                    print(f"  Bucket {bucket_range}: sampled {len(sampled):,} (all available, target was {bucket_target:,})")
                else:
                    sampled = bucket_df.sample(n=int(bucket_target), random_state=seed + i).reset_index(drop=True)
                    print(f"  Bucket {bucket_range}: sampled {len(sampled):,} (target: {bucket_target:,})")
                
                long_sampled_list.append(sampled)
            
            if long_sampled_list:
                long_sampled = pd.concat(long_sampled_list, ignore_index=True)
            else:
                long_sampled = pd.DataFrame()
    else:
        print("⚠️  Warning: No long samples (>512 tokens) found!")
        long_sampled = pd.DataFrame()
    
    # Combine short and long samples
    if len(short_sampled) > 0 and len(long_sampled) > 0:
        combined = pd.concat([short_sampled, long_sampled], ignore_index=True)
    elif len(short_sampled) > 0:
        combined = short_sampled
        print("⚠️  Warning: Only short samples available, using all short samples")
    elif len(long_sampled) > 0:
        combined = long_sampled
        print("⚠️  Warning: Only long samples available, using all long samples")
    else:
        print("❌ Error: No samples available in either bucket!")
        return df.head(0)
    
    # Shuffle for randomness and reproducibility
    final_subset = combined.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    print(f"\nFinal subset:")
    print(f"  Total samples: {len(final_subset):,}")
    if len(final_subset) > 0:
        short_count = len(final_subset[final_subset['tok_len'] <= 512])
        long_count = len(final_subset[final_subset['tok_len'] > 512])
        print(f"  Short samples: {short_count:,} ({short_count/len(final_subset)*100:.1f}%)")
        print(f"  Long samples: {long_count:,} ({long_count/len(final_subset)*100:.1f}%)")
    
    return final_subset

# Create subset (if enabled)
if USE_SUBSET:
    print("\n" + "="*70)
    print("CREATING TRAINING SUBSET (LENGTH-BASED BUCKET SAMPLING)")
    print("="*70)
    print(f"Creating subset of {SUBSET_TOTAL_N:,} samples with length-based bucket distribution...")
    print(f"  - 20% short snippets (<=512 tokens)")
    print(f"  - 80% long snippets (>512 tokens), evenly distributed across 512-token buckets")
    train_df_subset = make_length_bucket_subset(
        train_df, 
        SUBSET_TOTAL_N, 
        SUBSET_SEED
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

# Final summary: train, validation, test — sizes, ratios, label distributions (always print)
print("\n" + "="*70)
print("FINAL TRAIN / VALIDATION / TEST SUMMARY (for ratio verification)")
print("="*70)
train_label = "TRAIN (subset)" if USE_SUBSET else "TRAIN (full)"
for set_name, df in [(train_label, train_df), ("VALIDATION", val_df), ("TEST", test_df)]:
    n = len(df)
    ns = (df["tok_len"] <= MAX_LENGTH).sum()
    nl = (df["tok_len"] > MAX_LENGTH).sum()
    ps = ns / n * 100 if n else 0
    pl = nl / n * 100 if n else 0
    print(f"\n{set_name}: total={n:,}  |  short={ns:,} ({ps:.1f}%)  |  long={nl:,} ({pl:.1f}%)  [target 20%/80%]")
    print(f"  Label distribution: {dict(df['label'].value_counts().sort_index())}")
print("\n" + "="*70)

# ==================================================
# SECTION D — MULTI-VIEW CROPPING (NO WINDOWS)
# ==================================================

def tokenize_and_crop(df, tokenizer, max_length, use_multiview=True, crop_strategy="start_end", middle_crop_for=None):
    """
    Apply multi-view cropping to create training rows.
    
    If use_multiview is False: no cropping, use original code as-is
    If tok_len <= MAX_LENGTH: single view
    If tok_len > MAX_LENGTH and use_multiview is True:
        - start crop (first MAX_LENGTH tokens)
        - end crop (last MAX_LENGTH tokens)
        - optional middle crop ONLY for specified labels if crop_strategy == start_end_middle
    Each crop becomes its own training row with metadata.
    """
    if middle_crop_for is None:
        middle_crop_for = []
    
    expanded_rows = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Tokenizing and cropping"):
        code = row["code"]
        label = row["label"]
        tok_len = row.get("tok_len", 0)
        
        # If multiview is disabled, use code as-is (no cropping)
        if not use_multiview:
            expanded_rows.append({
                'code': code,
                'label': label,
                'view': 'single',
                'orig_id': idx
            })
            continue
        
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
# SECTION E — MODEL + TRAINER (4 CLASS)
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
if USE_MULTIVIEW:
    print(f"APPLYING MULTI-VIEW CROPPING TO TRAINING DATA (enabled)")
else:
    print(f"APPLYING TOKENIZATION TO TRAINING DATA (multiview disabled)")
print("="*70)
train_df_expanded = tokenize_and_crop(
    train_df,
    tokenizer,
    MAX_LENGTH,
    use_multiview=USE_MULTIVIEW,
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

# Custom callback to track per-epoch training time
class EpochTimeCallback(TrainerCallback):
    def __init__(self):
        self.epoch_times = []
        self.epoch_start_time = None
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if self.epoch_start_time is not None:
            epoch_time = time.time() - self.epoch_start_time
            self.epoch_times.append(epoch_time)
            epoch_num = len(self.epoch_times)
            print(f"\n[Epoch {epoch_num}] Training time: {epoch_time:.2f}s ({epoch_time/60:.2f} minutes)")

# Custom Trainer with focal/weighted loss
class CustomTrainer(Trainer):
    def __init__(self, class_weights=None, use_focal_loss=False, focal_gamma=2.0, use_class_weights=True, *args, **kwargs):
        # Newer transformers may not accept tokenizer; pop and set on self for save_model
        self._custom_tokenizer = kwargs.pop("tokenizer", None)
        super().__init__(*args, **kwargs)
        if self._custom_tokenizer is not None:
            self.tokenizer = self._custom_tokenizer
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
    output_dir=RESULTS_DIR,
    num_train_epochs=NUM_EPOCHS,  # Train for full epochs with early stopping
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    warmup_ratio=WARMUP_RATIO,
    max_grad_norm=MAX_GRAD_NORM,
    logging_steps=LOGGING_STEPS,
    eval_strategy=EVAL_STRATEGY if USE_EARLY_STOPPING else "no",
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

# Create callback for tracking epoch times
epoch_time_callback = EpochTimeCallback()

# Prepare callbacks list
callbacks_list = [epoch_time_callback]
if USE_EARLY_STOPPING:
    callbacks_list.append(EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE))

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
    callbacks=callbacks_list)

print(f"\nTraining configuration:")
print(f"  Focal loss: {USE_FOCAL_LOSS} (gamma={FOCAL_GAMMA})")
print(f"  Class weights: {USE_CLASS_WEIGHTS}")
print(f"  FP16: {FP16}")
print(f"  Effective batch size: {PER_DEVICE_TRAIN_BATCH * GRAD_ACCUM}")
print(f"  Total epochs: {NUM_EPOCHS}")
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

# Train the model
print("\n" + "="*70)
if USE_EARLY_STOPPING:
    print("TRAINING WITH EARLY STOPPING")
else:
    print("TRAINING WITHOUT EARLY STOPPING")
print("="*70)

# Track total training time
training_start_time = time.time()
trainer.train()
training_end_time = time.time()
total_training_time = training_end_time - training_start_time

trainer.save_model()
tokenizer.save_pretrained(RESULTS_DIR)

# Report training time statistics
print("\n" + "="*70)
print("TRAINING TIME STATISTICS")
print("="*70)
print(f"Total training time: {total_training_time:.2f}s ({total_training_time/60:.2f} minutes, {total_training_time/3600:.2f} hours)")

if epoch_time_callback.epoch_times:
    print(f"\nPer-epoch training times:")
    for i, epoch_time in enumerate(epoch_time_callback.epoch_times, 1):
        print(f"  Epoch {i}: {epoch_time:.2f}s ({epoch_time/60:.2f} minutes)")
    
    avg_epoch_time = np.mean(epoch_time_callback.epoch_times)
    print(f"\nAverage time per epoch: {avg_epoch_time:.2f}s ({avg_epoch_time/60:.2f} minutes)")
    print(f"Total epochs completed: {len(epoch_time_callback.epoch_times)}")
    
    if USE_EARLY_STOPPING and len(epoch_time_callback.epoch_times) < NUM_EPOCHS:
        print(f"Training stopped early (converged after {len(epoch_time_callback.epoch_times)} epochs)")
    elif len(epoch_time_callback.epoch_times) == NUM_EPOCHS:
        print(f"Training completed all {NUM_EPOCHS} epochs")

print("Training completed!")

# ==================================================
# SECTION F — MULTI-VIEW INFERENCE FUNCTIONS
# ==================================================

@torch.no_grad()
def predict_multiview(code, model, tokenizer, max_length, device, use_multiview=True):
    """
    Run inference on a single code snippet.
    
    If use_multiview is False: single forward pass (truncate if needed)
    If use_multiview is True:
        - For short code (tok_len <= MAX_LENGTH): single forward pass
        - For long code (tok_len > MAX_LENGTH): start + end crops
    
    Returns aggregated logits (4-class).
    """
    # Tokenize without truncation
    tokens = tokenizer(code, add_special_tokens=False, truncation=False, return_tensors=None)
    input_ids = tokens["input_ids"]
    tok_len = len(input_ids)
    
    if tok_len == 0:
        # Empty code - return uniform logits
        return torch.zeros(NUM_LABELS).to(device)
    
    # If multiview is disabled, use simple truncation
    if not use_multiview:
        encoded = tokenizer(code, truncation=True, max_length=max_length, return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        outputs = model(**encoded)
        logits = outputs.logits[0]  # [NUM_LABELS]
        return logits
    
    # Multiview enabled
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
# SECTION H — MANUAL EVALUATION (MULTI-VIEW AWARE)
# ==================================================

def evaluate_4class_multiview(val_df, model, tokenizer, device, save_mistakes=True, mistakes_n=200, mistakes_csv_path=None, save_inference=False, inference_csv_path=None, use_multiview=True):
    """
    Evaluate on validation set with multi-view inference aggregation.
    mistakes_csv_path: if save_mistakes is True, write mistakes here; default "validation_mistakes.csv".
    save_inference: if True, save all inference results to CSV
    inference_csv_path: if save_inference is True, write all inference results here
    use_multiview: enable/disable multi-view inference
    """
    model.eval()
    
    predictions_4class = []
    true_labels = []
    all_stats = []  # For error analysis
    inference_times = []  # Track per-snippet inference latency
    
    start_time = time.time()
    
    print("\n" + "="*70)
    if use_multiview:
        print("RUNNING MULTI-VIEW INFERENCE ON VALIDATION SET (enabled)")
    else:
        print("RUNNING INFERENCE ON VALIDATION SET (multiview disabled)")
    print("="*70)
    
    for idx, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Evaluating"):
        code = row["code"]
        true_label = row["label"]
        tok_len = row.get("tok_len", 0)
        
        # Track inference latency for this snippet
        inference_start = time.time()
        
        # Run inference with or without multiview
        logits = predict_multiview(code, model, tokenizer, MAX_LENGTH, device, use_multiview=use_multiview)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        pred_4class = int(np.argmax(probs))
        
        inference_time = time.time() - inference_start
        inference_times.append(inference_time)
        
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
    print("SNIPPET-LEVEL EVALUATION RESULTS (4-CLASS)")
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
    
    # F1 by length buckets (if enabled) - matches subset creation buckets
    if REPORT_LENGTH_BUCKETS:
        print(f"\nMacro F1 by Length Buckets (matching subset creation):")
        
        # Get max token length to determine number of buckets
        max_tok_len = max([s['tok_len'] for s in all_stats]) if all_stats else 0
        
        # Create buckets matching subset creation: <=512 (short), then [512, 1024), [1024, 1536), etc.
        bucket_start = 512
        bucket_size = 512
        
        # Short bucket: <=512
        length_buckets = [(-1, 512)]  # Use -1 as lower bound to include 0-512
        
        # Long buckets: [512, 1024), [1024, 1536), [1536, 2048), etc.
        num_buckets = int(np.ceil((max_tok_len - bucket_start) / bucket_size)) if max_tok_len > bucket_start else 0
        for i in range(num_buckets):
            bucket_low = bucket_start + i * bucket_size
            bucket_high = bucket_start + (i + 1) * bucket_size
            length_buckets.append((bucket_low, bucket_high))
        
        # Add final catch-all bucket if needed
        if max_tok_len > bucket_start + num_buckets * bucket_size:
            length_buckets.append((bucket_start + num_buckets * bucket_size, float('inf')))
        
        for low, high in length_buckets:
            mask = np.array([s['tok_len'] for s in all_stats])
            if low == -1:  # Short bucket: <=512
                mask = (mask <= high)
                bucket_label = f"<=512"
            elif high == float('inf'):
                mask = (mask >= low)
                bucket_label = f">={low}+"
            else:
                mask = (mask >= low) & (mask < high)  # Use < for upper bound to match subset creation
                bucket_label = f"[{low}, {high})"
            
            if mask.sum() > 0:
                bucket_true = true_labels[mask]
                bucket_pred = predictions_4class[mask]
                bucket_f1 = f1_score(bucket_true, bucket_pred, average='macro', zero_division=0)
                print(f"   {bucket_label} tokens: {bucket_f1:.4f} (n={mask.sum()})")
    
    # Runtime stats
    print(f"\nRuntime Stats:")
    print(f"   Total snippets: {len(val_df)}")
    print(f"   Total time: {elapsed:.2f}s ({elapsed/60:.2f} minutes)")
    print(f"   Throughput: {len(val_df)/elapsed:.2f} snippets/sec")
    
    # Inference latency statistics
    if inference_times:
        avg_inference_time = np.mean(inference_times)
        median_inference_time = np.median(inference_times)
        min_inference_time = np.min(inference_times)
        max_inference_time = np.max(inference_times)
        std_inference_time = np.std(inference_times)
        
        print(f"\nInference Latency (per snippet):")
        print(f"   Average: {avg_inference_time*1000:.2f}ms ({avg_inference_time:.4f}s)")
        print(f"   Median: {median_inference_time*1000:.2f}ms ({median_inference_time:.4f}s)")
        print(f"   Min: {min_inference_time*1000:.2f}ms ({min_inference_time:.4f}s)")
        print(f"   Max: {max_inference_time*1000:.2f}ms ({max_inference_time:.4f}s)")
        print(f"   Std Dev: {std_inference_time*1000:.2f}ms ({std_inference_time:.4f}s)")
    
    # Save all inference results
    if save_inference:
        inference_df = pd.DataFrame(all_stats)
        out_path = inference_csv_path if inference_csv_path is not None else os.path.join(OUTPUT_DIR, "inference_results.csv")
        inference_df.to_csv(out_path, index=False)
        print(f"\nSaved all inference results to {out_path} ({len(inference_df)} samples)")
    else:
        inference_df = None
    
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
            out_path = mistakes_csv_path if mistakes_csv_path is not None else os.path.join(OUTPUT_DIR, "validation_mistakes.csv")
            mistakes_df.to_csv(out_path, index=False)
            print(f"Saved {len(mistakes_df)} hardest mistakes to {out_path}")
        else:
            mistakes_df = None
    else:
        mistakes_df = None

    return {
        'macro_f1': macro_f1,
        'per_class_f1': dict(zip(unique_labels, per_class_f1)),
        'confusion_matrix': cm,
        'predictions': predictions_4class,
        'true_labels': true_labels,
        'all_stats': all_stats,
        'mistakes_df': mistakes_df if save_mistakes else None
    }

# Run evaluation on validation set
print("\n" + "="*70)
print(f"FINAL VALIDATION EVALUATION ({len(val_df):,} samples)")
print("="*70)
eval_results = evaluate_4class_multiview(
    val_df,
    model,
    tokenizer,
    device,
    save_mistakes=SAVE_MISTAKES,
    mistakes_n=MISTAKES_N,
    mistakes_csv_path=os.path.join(OUTPUT_DIR, "validation_mistakes.csv"),
    save_inference=SAVE_INFERENCE,
    inference_csv_path=os.path.join(OUTPUT_DIR, "validation_inference.csv"),
    use_multiview=USE_MULTIVIEW)

# Run evaluation on held-out test set (same distribution as train)
print("\n" + "="*70)
print(f"FINAL TEST SET EVALUATION ({len(test_df):,} samples)")
print("="*70)
evaluate_4class_multiview(
    test_df,
    model,
    tokenizer,
    device,
    save_mistakes=SAVE_MISTAKES,
    mistakes_n=MISTAKES_N,
    mistakes_csv_path=os.path.join(OUTPUT_DIR, "test_mistakes.csv"),
    save_inference=SAVE_INFERENCE,
    inference_csv_path=os.path.join(OUTPUT_DIR, "test_inference.csv"),
    use_multiview=USE_MULTIVIEW)

# ==================================================
# SECTION I — TEST PREDICTION + SUBMISSION (LOCAL VERSION)
# ==================================================

def predict_test_and_write_submission(test_parquet_path, output_csv_path, model, tokenizer, device, use_multiview=True):
    """
    Stream test set, run inference, write 4-class predictions to submission.
    LOCAL VERSION: Uses local file paths
    use_multiview: enable/disable multi-view inference
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
    except Exception as e:
        print(f"❌ Error loading test file: {e}")
        return
    
    print("\n" + "="*70)
    print("PREDICTING ON TEST SET")
    print("="*70)
    
    with open(output_csv_path, "w") as f:
        f.write("ID,prediction\n")
        
        count = 0
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Predicting"):
            code = row["code"]
            ex_id = row["ID"]
            
            # Run inference with or without multiview
            logits = predict_multiview(code, model, tokenizer, MAX_LENGTH, device, use_multiview=use_multiview)
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
    else:
        print(f"\n⚠️  WARNING: File {output_csv_path} was not created!")

# Generate submission file on test set (if enabled)
if GENERATE_SUBMISSION:
    predict_test_and_write_submission(TEST_PARQUET_PATH, SUBMISSION_OUTPUT_PATH, model, tokenizer, device, use_multiview=USE_MULTIVIEW)
else:
    print("\n" + "="*70)
    print("SUBMISSION GENERATION SKIPPED")
    print("="*70)
    print("To generate submission.csv, set GENERATE_SUBMISSION = True in the config section")
    print(f"Test parquet path: {TEST_PARQUET_PATH}")
    print(f"Output path: {SUBMISSION_OUTPUT_PATH}")

# ==================================================
# SECTION J — FINAL RUN MODES (LOCAL VERSION)
# ==================================================

## LOCAL RUN MODE:
# - SUBSET_TOTAL_N = 20_000 (reduced for local testing)
# - NUM_EPOCHS = 2
# - HARD NEGATIVE MINING ENABLED (small scale: ~1000-2000 mined)
# - VALIDATION LIMITED to 2_000 samples
# Goal:
#   - Test the complete pipeline locally
#   - Verify training and evaluation work
#   - Debug any issues before full-scale run

## To run with full data locally:
# - Set SUBSET_TOTAL_N = 200_000 (or available data size)
# - Set USE_FULL_VALIDATION = True
# - Increase batch sizes if you have enough GPU memory
# - Set GENERATE_SUBMISSION = True if you have test data

## IMPORTANT LOCAL NOTES:
# - Ensure you have the required parquet files in the current directory
# - The script will check for required files before starting
# - Model and results: RESULTS_DIR (./results locally, /kaggle/working/results on Kaggle)
# - Mistakes and submission: OUTPUT_DIR
# - All paths are relative to the current working directory

print("\n" + "="*70)
print("LOCAL EXECUTION COMPLETED")
print("="*70)
print("Files created:")
print(f"  - {RESULTS_DIR}/ (model checkpoints and tokenizer)")
print(f"  - {SAVE_SUBSET_PATH} (training subset)")
if SAVE_INFERENCE:
    print(f"  - {os.path.join(OUTPUT_DIR, 'validation_inference.csv')} (validation inference results)")
    print(f"  - {os.path.join(OUTPUT_DIR, 'test_inference.csv')} (test set inference results)")
if SAVE_MISTAKES:
    print(f"  - {os.path.join(OUTPUT_DIR, 'validation_mistakes.csv')} (validation error analysis)")
    print(f"  - {os.path.join(OUTPUT_DIR, 'test_mistakes.csv')} (test set error analysis)")
if GENERATE_SUBMISSION:
    print(f"  - {SUBMISSION_OUTPUT_PATH} (test predictions)")
print("="*70)
