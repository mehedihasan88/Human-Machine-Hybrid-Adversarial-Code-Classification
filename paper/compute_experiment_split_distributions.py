#!/usr/bin/env python3
"""
Rebuild length-bin stats for the *actual* experiment splits:
  - Training: same procedure as ContextOverflow/long_code_approach_basic.py
    (UniXcoder tok_len on full train parquet, then SUBSET_TOTAL_N=200k, SUBSET_SEED=42).
  - Validation / Test: from results/validation_inference.csv and
    results/test_inference_basic.csv (exact rows used at inference).

Writes paper/experiment_split_distributions.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_REPO, "_pydeps"))

from transformers import AutoTokenizer  # noqa: E402

MODEL = "microsoft/unixcoder-base"
TRAIN_PARQUET = os.path.join(_REPO, "dataset", "task_c_training_set_1.parquet")
VAL_CSV = os.path.join(_REPO, "results", "validation_inference.csv")
TEST_CSV = os.path.join(_REPO, "results", "test_inference_basic.csv")
OUT_JSON = os.path.join(_REPO, "paper", "experiment_split_distributions.json")

SUBSET_TOTAL_N = 200_000
SUBSET_SEED = 42
BATCH = 256
MAX_LENGTH = 512

# Same coarse bins as in the paper (for comparison across splits)
BINS = [
    (0, 128, "0--128"),
    (129, 256, "129--256"),
    (257, 512, "257--512"),
    (513, 1024, "513--1024"),
    (1025, 10**15, "1025+"),
]


def make_length_bucket_subset(df: pd.DataFrame, total_n: int, seed: int) -> pd.DataFrame:
    """Mirror of long_code_approach_basic.make_length_bucket_subset (incl. num_non_empty_buckets=1.5)."""
    np.random.seed(seed)
    short_target = int(total_n * 0.20)
    long_target = total_n - short_target
    short_df = df[df["tok_len"] <= 512].copy()
    long_df = df[df["tok_len"] > 512].copy()

    if len(short_df) == 0:
        short_sampled = pd.DataFrame()
    elif len(short_df) <= short_target:
        short_sampled = short_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    else:
        short_sampled = short_df.sample(n=short_target, random_state=seed).reset_index(drop=True)

    long_buckets = []
    if len(long_df) > 0:
        max_tok_len = long_df["tok_len"].max()
        num_buckets = int(np.ceil((max_tok_len - 512) / 512))
        for i in range(num_buckets):
            bucket_low = 512 + i * 512
            bucket_high = 512 + (i + 1) * 512
            bucket_df = long_df[(long_df["tok_len"] >= bucket_low) & (long_df["tok_len"] < bucket_high)]
            if len(bucket_df) > 0:
                long_buckets.append({"range": (bucket_low, bucket_high), "df": bucket_df})

        num_non_empty_buckets = 1.5
        if num_non_empty_buckets == 0:
            long_sampled = pd.DataFrame()
        else:
            per_bucket_target = long_target // num_non_empty_buckets
            remainder = long_target % num_non_empty_buckets
            long_sampled_list = []
            for i, bucket_info in enumerate(long_buckets):
                bucket_df = bucket_info["df"]
                bucket_range = bucket_info["range"]
                bucket_target = per_bucket_target + (1 if i < remainder else 0)
                if len(bucket_df) <= bucket_target:
                    sampled = bucket_df.sample(frac=1, random_state=seed + i).reset_index(drop=True)
                else:
                    sampled = bucket_df.sample(n=int(bucket_target), random_state=seed + i).reset_index(drop=True)
                long_sampled_list.append(sampled)
            long_sampled = pd.concat(long_sampled_list, ignore_index=True) if long_sampled_list else pd.DataFrame()
    else:
        long_sampled = pd.DataFrame()

    if len(short_sampled) > 0 and len(long_sampled) > 0:
        combined = pd.concat([short_sampled, long_sampled], ignore_index=True)
    elif len(short_sampled) > 0:
        combined = short_sampled
    elif len(long_sampled) > 0:
        combined = long_sampled
    else:
        return df.head(0)

    return combined.sample(frac=1, random_state=seed).reset_index(drop=True)


def tokenize_all(codes: list[str], tokenizer) -> list[int]:
    lengths: list[int] = []
    for i in range(0, len(codes), BATCH):
        batch = codes[i : i + BATCH]
        enc = tokenizer(
            batch,
            add_special_tokens=False,
            truncation=False,
            return_length=True,
        )
        lengths.extend(enc["length"])
    return lengths


def bin_counts(tok: np.ndarray) -> dict:
    out = {}
    total = len(tok)
    for lo, hi, lab in BINS:
        if hi >= 10**14:
            m = tok >= lo
        else:
            m = (tok >= lo) & (tok <= hi)
        c = int(m.sum())
        out[lab] = {"count": c, "pct": round(100.0 * c / total, 2) if total else 0.0}
    return out


def label_summary(series: pd.Series) -> dict:
    vc = series.value_counts().sort_index()
    n = int(vc.sum())
    return {str(int(k)): {"count": int(v), "pct": round(100.0 * v / n, 2)} for k, v in vc.items()}


def main() -> None:
    if not os.path.isfile(TRAIN_PARQUET):
        print(f"Missing {TRAIN_PARQUET}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(VAL_CSV) or not os.path.isfile(TEST_CSV):
        print("Missing validation or test inference CSV under results/", file=sys.stderr)
        sys.exit(1)

    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(MODEL)

    print("Reading training parquet...")
    train_df = pd.read_parquet(TRAIN_PARQUET)
    codes = train_df["code"].fillna("").astype(str).tolist()
    print(f"Tokenizing {len(codes):,} training rows...")
    lengths = tokenize_all(codes, tok)
    train_df = train_df.copy()
    train_df["tok_len"] = lengths

    print("Building 200k stratified training subset (same logic as training script)...")
    train_subset = make_length_bucket_subset(train_df, SUBSET_TOTAL_N, SUBSET_SEED)
    train_tok = train_subset["tok_len"].values

    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)
    val_tok = val_df["tok_len"].values
    test_tok = test_df["tok_len"].values

    payload = {
        "tokenizer": MODEL,
        "max_length_encoder": MAX_LENGTH,
        "training_subset": {
            "n": int(len(train_subset)),
            "target_n": SUBSET_TOTAL_N,
            "seed": SUBSET_SEED,
            "policy": "20% tok_len<=512, 80% long with bucket sampling (see methodology)",
            "label_distribution": label_summary(train_subset["label"]),
            "length_bins": bin_counts(train_tok),
            "short_le512": int((train_tok <= MAX_LENGTH).sum()),
            "long_gt512": int((train_tok > MAX_LENGTH).sum()),
        },
        "validation_split": {
            "source_csv": os.path.relpath(VAL_CSV, _REPO),
            "n": int(len(val_df)),
            "label_distribution": label_summary(val_df["true_label"]),
            "length_bins": bin_counts(val_tok),
            "short_le512": int((val_tok <= MAX_LENGTH).sum()),
            "long_gt512": int((val_tok > MAX_LENGTH).sum()),
        },
        "test_split": {
            "source_csv": os.path.relpath(TEST_CSV, _REPO),
            "n": int(len(test_df)),
            "label_distribution": label_summary(test_df["true_label"]),
            "length_bins": bin_counts(test_tok),
            "short_le512": int((test_tok <= MAX_LENGTH).sum()),
            "long_gt512": int((test_tok > MAX_LENGTH).sum()),
        },
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
