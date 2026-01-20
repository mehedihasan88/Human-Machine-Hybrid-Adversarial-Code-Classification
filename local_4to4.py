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
    DataCollatorWithPadding)
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    confusion_matrix, f1_score)
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

# Data / Subset
USE_SUBSET = True
SUBSET_TOTAL_N = 80_000         # REDUCED for local testing (was 200_000)
SUBSET_SEED = 42
BALANCE_BY_LANGUAGE = True
BALANCE_BY_GENERATOR = False
SAVE_SUBSET_PATH = "train_subset_4class.parquet"

# Validation control
USE_FULL_VALIDATION = False     # False for local testing to speed up
VAL_DRY_RUN_MAX_N = 2_000       # Reduced for local testing
VAL_SUBSET_SEED = 123

# Labels (4-class direct)
HUMAN_LABEL_ID = 0
MACHINE_LABEL_ID = 1
HYBRID_LABEL_ID = 2
ADV_LABEL_ID = 3
NUM_LABELS = 4

# Token / Cropping
MAX_LENGTH = 512
CROP_STRATEGY = "start_end"        # options: start_only | start_end | start_end_middle
MIDDLE_CROP_FOR = ["hybrid", "adversarial"]
RANDOM_CROP_SEED = 123

# Training
MODEL_NAME = "microsoft/unixcoder-base"
PER_DEVICE_TRAIN_BATCH = 4        # Reduced for local testing
PER_DEVICE_EVAL_BATCH = 8         # Reduced for local testing
GRAD_ACCUM = 2
LR = 2e-5
NUM_EPOCHS = 2                    # Keep 2 epochs for testing
WARMUP_RATIO = 0.05
MAX_GRAD_NORM = 1.0
FP16 = True if torch.cuda.is_available() else False

# Loss
USE_FOCAL_LOSS = True
FOCAL_GAMMA = 2.0
USE_CLASS_WEIGHTS = True

# Logging / Eval
LOGGING_STEPS = 100               # Reduced for local testing
SAVE_TOTAL_LIMIT = 2
REPORT_LENGTH_BUCKETS = True
SAVE_MISTAKES = True
MISTAKES_N = 50                   # Reduced for local testing

# Hard Negative Mining (ENABLED EVEN IN DRY RUN)
DO_HARD_MINING = True
HARD_MINING_AFTER_EPOCH = 1
HARD_MINING_MIN_CONF = 0.60
HARD_MINING_FOCUS_LABELS = [HYBRID_LABEL_ID, ADV_LABEL_ID]

# Dynamic mining size (important) - LOCAL VERSION
HARD_MINING_FRAC = 0.20           # fraction of subset to mine
HARD_MINING_MAX_CAP = 2_000       # Reduced for local testing

# Test prediction / submission - LOCAL VERSION
GENERATE_SUBMISSION = False       # Disabled by default for local testing
TEST_PARQUET_PATH = "task_c_test_set_sample.parquet"  # Local test file
SUBMISSION_OUTPUT_PATH = "submission.csv"  # Local output

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==================================================
# SECTION B — DATA LOADING + TOKEN LENGTH STATS (LOCAL VERSION)
# ==================================================

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
    Create balanced subset with:
    1. STRICT balance by label (25% each for 4 classes)
    2. Soft balance by language (proportional with minimum floor)
    3. DO NOT hard-balance generators for small subsets
    4. Length-aware bias using tok_len bins
    """
    np.random.seed(seed)
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

# Create balanced subset (if enabled)
if USE_SUBSET:
    print("\n" + "="*70)
    print("CREATING BALANCED 4-CLASS SUBSET")
    print("="*70)
    
    print(f"Creating balanced subset of {SUBSET_TOTAL_N} samples...")
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

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    warmup_ratio=WARMUP_RATIO,
    max_grad_norm=MAX_GRAD_NORM,
    logging_steps=LOGGING_STEPS,
    eval_strategy="no",  # We'll do snippet-level eval manually
    save_total_limit=SAVE_TOTAL_LIMIT,
    load_best_model_at_end=False,  # Manual evaluation
    fp16=FP16,
    report_to=[],
    remove_unused_columns=False,
)

# Create trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=None,  # No built-in eval (we do snippet-level)
    tokenizer=tokenizer,
    data_collator=data_collator,
    class_weights=class_weights if USE_CLASS_WEIGHTS else None,
    use_focal_loss=USE_FOCAL_LOSS,
    focal_gamma=FOCAL_GAMMA,
    use_class_weights=USE_CLASS_WEIGHTS)

print(f"\nTraining configuration:")
print(f"  Focal loss: {USE_FOCAL_LOSS} (gamma={FOCAL_GAMMA})")
print(f"  Class weights: {USE_CLASS_WEIGHTS}")
print(f"  FP16: {FP16}")
print(f"  Effective batch size: {PER_DEVICE_TRAIN_BATCH * GRAD_ACCUM}")

# Train the model (Epoch 1)
print("\n" + "="*70)
print("TRAINING EPOCH 1")
print("="*70)
trainer.train()
trainer.save_model()
tokenizer.save_pretrained("./results")
print("Epoch 1 completed!")

# ==================================================
# SECTION F — MULTI-VIEW INFERENCE FUNCTIONS
# ==================================================

@torch.no_grad()
def predict_multiview(code, model, tokenizer, max_length, device):
    """
    Run multi-view inference on a single code snippet.
    
    For short code (tok_len <= MAX_LENGTH): single forward pass
    For long code (tok_len > MAX_LENGTH): start + end crops
    
    Returns aggregated logits (4-class).
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
# SECTION G — HARD NEGATIVE MINING (TESTED IN DRY RUN)
# ==================================================

if DO_HARD_MINING and NUM_EPOCHS > 1:
    print("\n" + "="*70)
    print("HARD NEGATIVE MINING")
    print("="*70)
    
    # Calculate mined_count - LOCAL VERSION
    if SUBSET_TOTAL_N <= 10_000:
        # DRY RUN: smaller scale
        mined_count = min(1000, int(SUBSET_TOTAL_N * HARD_MINING_FRAC))
    else:
        # REAL RUN: full scale with cap
        mined_count = min(HARD_MINING_MAX_CAP, int(SUBSET_TOTAL_N * HARD_MINING_FRAC))
    
    print(f"Mining target: {mined_count} hard negatives")
    print(f"Focus labels: {HARD_MINING_FOCUS_LABELS} (Hybrid={HYBRID_LABEL_ID}, Adversarial={ADV_LABEL_ID})")
    
    # Run inference on training subset to find hard negatives
    model.eval()
    hard_negatives = []
    
    print("\nRunning inference on training subset to identify hard negatives...")
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Mining"):
        code = row["code"]
        true_label = row["label"]
        tok_len = row.get("tok_len", 0)
        
        # Run multi-view inference
        logits = predict_multiview(code, model, tokenizer, MAX_LENGTH, device)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        pred_label = int(np.argmax(probs))
        max_prob = float(np.max(probs))
        
        # Check if it's a hard negative
        is_focus_label = true_label in HARD_MINING_FOCUS_LABELS
        is_misclassified = pred_label != true_label
        is_confident = max_prob >= HARD_MINING_MIN_CONF
        
        if is_focus_label and is_misclassified and is_confident:
            # Compute loss for ranking
            loss = -np.log(probs[true_label] + 1e-8)
            hard_negatives.append({
                'idx': idx,
                'code': code,
                'label': true_label,
                'pred_label': pred_label,
                'max_prob': max_prob,
                'loss': loss,
                'tok_len': tok_len
            })
    
    print(f"\nFound {len(hard_negatives)} potential hard negatives")
    
    # Sort by loss (highest loss = hardest)
    hard_negatives.sort(key=lambda x: x['loss'], reverse=True)
    
    # Select top mined_count
    selected_hard = hard_negatives[:mined_count]
    
    # Count by label
    hybrid_count = sum(1 for h in selected_hard if h['label'] == HYBRID_LABEL_ID)
    adv_count = sum(1 for h in selected_hard if h['label'] == ADV_LABEL_ID)
    print(f"\nSelected {len(selected_hard)} hard negatives:")
    print(f"  Hybrid: {hybrid_count}")
    print(f"  Adversarial: {adv_count}")
    
    # Count confusion types
    confusion_types = defaultdict(int)
    for h in selected_hard:
        key = f"True_{h['label']}_Pred_{h['pred_label']}"
        confusion_types[key] += 1
    
    print(f"\nTop confusion types:")
    for conf_type, count in sorted(confusion_types.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {conf_type}: {count}")
    
    # Build new training dataset for Epoch 2
    print("\n" + "="*70)
    print("REBUILDING TRAINING DATASET FOR EPOCH 2")
    print("="*70)
    
    # Get indices of selected hard negatives
    hard_indices = set(h['idx'] for h in selected_hard)
    
    # Remove easiest Human/Machine samples (correct & high confidence)
    # Keep all Hybrid and Adversarial, and hard negatives
    easy_to_remove = []
    for idx, row in train_df.iterrows():
        if idx in hard_indices:
            continue  # Keep hard negatives
        if row["label"] in [HYBRID_LABEL_ID, ADV_LABEL_ID]:
            continue  # Keep all Hybrid/Adversarial
        
        # Check if this was correctly classified with high confidence
        code = row["code"]
        logits = predict_multiview(code, model, tokenizer, MAX_LENGTH, device)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        pred_label = int(np.argmax(probs))
        max_prob = float(np.max(probs))
        
        if pred_label == row["label"] and max_prob >= 0.85:  # Easy correct
            easy_to_remove.append(idx)
    
    # Remove up to len(selected_hard) easy samples to maintain size
    num_to_remove = min(len(easy_to_remove), len(selected_hard))
    remove_indices = set(easy_to_remove[:num_to_remove])
    
    print(f"Removing {len(remove_indices)} easy samples")
    print(f"Adding {len(selected_hard)} hard negatives")
    
    # Build new training dataframe
    # Keep all samples except those we're removing
    train_df_epoch2 = train_df[~train_df.index.isin(remove_indices)].copy()
    
    # Add hard negatives (they might already be in, but we'll ensure they're there)
    hard_df = pd.DataFrame([{
        'code': h['code'],
        'label': h['label'],
        'tok_len': h['tok_len']
    } for h in selected_hard])
    
    # Merge: add hard negatives that aren't already present
    train_df_epoch2 = pd.concat([train_df_epoch2, hard_df], ignore_index=True)
    train_df_epoch2 = train_df_epoch2.drop_duplicates(subset=['code'], keep='last')
    
    print(f"Epoch 2 training set size: {len(train_df_epoch2)}")
    print("\nEpoch 2 label distribution:")
    print(train_df_epoch2["label"].value_counts().sort_index())
    
    # Apply multi-view cropping to Epoch 2 dataset
    print("\nApplying multi-view cropping to Epoch 2 dataset...")
    train_df_epoch2_expanded = tokenize_and_crop(
        train_df_epoch2,
        tokenizer,
        MAX_LENGTH,
        crop_strategy=CROP_STRATEGY,
        middle_crop_for=MIDDLE_CROP_FOR)
    
    print(f"Epoch 2 expanded dataset size: {len(train_df_epoch2_expanded)}")
    
    # Recompute class weights
    unique_labels_epoch2 = np.array(sorted(train_df_epoch2_expanded["label"].unique()))
    class_weights_array_epoch2 = compute_class_weight(
        'balanced',
        classes=unique_labels_epoch2,
        y=train_df_epoch2_expanded["label"].values
    )
    class_weights_epoch2 = torch.tensor(class_weights_array_epoch2, dtype=torch.float32).to(device)
    print(f"\nEpoch 2 class weights: {dict(zip(unique_labels_epoch2, class_weights_array_epoch2))}")
    
    # Create new dataset
    train_dataset_epoch2 = Dataset.from_pandas(train_df_epoch2_expanded[['code', 'label']])
    train_dataset_epoch2 = train_dataset_epoch2.map(
        tokenize_function,
        batched=True,
        remove_columns=['code']
    )
    train_dataset_epoch2 = train_dataset_epoch2.rename_column('label', 'labels')
    
    # Update trainer with new dataset
    trainer.train_dataset = train_dataset_epoch2
    trainer.class_weights = class_weights_epoch2 if USE_CLASS_WEIGHTS else None
    
    # Train Epoch 2
    print("\n" + "="*70)
    print("TRAINING EPOCH 2")
    print("="*70)
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained("./results")
    print("Epoch 2 completed!")
else:
    if NUM_EPOCHS > 1:
        print("\n" + "="*70)
        print("TRAINING EPOCH 2 (NO HARD MINING)")
        print("="*70)
        trainer.train()
        trainer.save_model()
        tokenizer.save_pretrained("./results")
        print("Epoch 2 completed!")

# ==================================================
# SECTION H — MANUAL EVALUATION (MULTI-VIEW AWARE)
# ==================================================

def evaluate_4class_multiview(val_df, model, tokenizer, device, save_mistakes=True, mistakes_n=200):
    """
    Evaluate on validation set with multi-view inference aggregation.
    """
    model.eval()
    
    predictions_4class = []
    true_labels = []
    all_stats = []  # For error analysis
    
    start_time = time.time()
    
    print("\n" + "="*70)
    print("RUNNING MULTI-VIEW INFERENCE ON VALIDATION SET")
    print("="*70)
    
    for idx, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Evaluating"):
        code = row["code"]
        true_label = row["label"]
        tok_len = row.get("tok_len", 0)
        
        # Run multi-view inference
        logits = predict_multiview(code, model, tokenizer, MAX_LENGTH, device)
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
            mistakes_df.to_csv("validation_mistakes.csv", index=False)
            print(f"\nSaved {len(mistakes_df)} hardest mistakes to validation_mistakes.csv")
    
    return {
        'macro_f1': macro_f1,
        'per_class_f1': dict(zip(unique_labels, per_class_f1)),
        'confusion_matrix': cm,
        'predictions': predictions_4class,
        'true_labels': true_labels,
        'all_stats': all_stats
    }

# Run evaluation on validation set
print("\n" + "="*70)
print("FINAL VALIDATION EVALUATION")
print("="*70)
eval_results = evaluate_4class_multiview(
    val_df, 
    model, 
    tokenizer, 
    device,
    save_mistakes=SAVE_MISTAKES,
    mistakes_n=MISTAKES_N)

# ==================================================
# SECTION I — TEST PREDICTION + SUBMISSION (LOCAL VERSION)
# ==================================================

def predict_test_and_write_submission(test_parquet_path, output_csv_path, model, tokenizer, device):
    """
    Stream test set, run multi-view inference, write 4-class predictions to submission.
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
    else:
        print(f"\n⚠️  WARNING: File {output_csv_path} was not created!")

# Generate submission file on test set (if enabled)
if GENERATE_SUBMISSION:
    predict_test_and_write_submission(TEST_PARQUET_PATH, SUBMISSION_OUTPUT_PATH, model, tokenizer, device)
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
# - Model and results will be saved to "./results/" directory
# - Validation mistakes will be saved to "validation_mistakes.csv"
# - All paths are relative to the current working directory

print("\n" + "="*70)
print("LOCAL EXECUTION COMPLETED")
print("="*70)
print("Files created:")
print("  - ./results/ (model checkpoints and tokenizer)")
print("  - train_subset_4class.parquet (training subset)")
if SAVE_MISTAKES:
    print("  - validation_mistakes.csv (error analysis)")
if GENERATE_SUBMISSION:
    print("  - submission.csv (test predictions)")
print("="*70)
