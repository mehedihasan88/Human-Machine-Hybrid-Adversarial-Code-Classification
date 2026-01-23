#!/usr/bin/env python3
"""
Analyze test.parquet dataset to generate token length statistics
Format: 0-128 tokens: 0.8064 (n=44613)
"""

import pandas as pd
from tqdm import tqdm
import numpy as np
import re

def estimate_token_length(text):
    """
    Estimate token length for code using a simple heuristic.
    This approximates how code tokenizers work by counting:
    - Words, numbers, operators, punctuation as separate tokens
    - Handles common programming language patterns
    """
    if not text or pd.isna(text):
        return 0
    
    # Remove comments and strings to avoid counting them as individual tokens
    # Simple regex patterns for common comment styles
    text = re.sub(r'//.*', '', text)  # Single line comments
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)  # Multi-line comments
    text = re.sub(r'#.*', '', text)  # Python comments
    
    # Split on common programming language token boundaries
    # This includes: whitespace, operators, punctuation, brackets
    tokens = re.findall(r'\w+|[^\w\s]', text)
    
    # Filter out empty tokens
    tokens = [t for t in tokens if t.strip()]
    
    return len(tokens)

def compute_token_lengths(df, batch_size=256):
    """Compute estimated token length for each code snippet."""
    texts = df["code"].tolist()
    lengths = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Computing token lengths"):
        batch_texts = texts[i:i+batch_size]
        batch_lengths = [estimate_token_length(text) for text in batch_texts]
        lengths.extend(batch_lengths)
    
    df = df.copy()
    df["tok_len"] = lengths
    return df

def generate_token_statistics(df):
    """Generate statistics in the requested format."""
    # Define token buckets
    buckets = [
        (0, 128, "0-128"),
        (129, 256, "129-256"), 
        (257, 512, "257-512"),
        (513, 1024, "513-1024"),
        (1025, float('inf'), "1025-+")
    ]
    
    total_samples = len(df)
    stats = []
    
    for low, high, label in buckets:
        if high == float('inf'):
            mask = df["tok_len"] >= low
        else:
            mask = (df["tok_len"] >= low) & (df["tok_len"] <= high)
        
        count = mask.sum()
        percentage = count / total_samples
        
        stats.append({
            'range': label,
            'percentage': percentage,
            'count': count
        })
    
    return stats

def main():
    print("="*70)
    print("TEST DATASET TOKEN ANALYSIS")
    print("="*70)
    
    # Load test data
    try:
        test_df = pd.read_parquet('test.parquet')
        print(f"✅ Loaded test data: {len(test_df)} samples")
    except Exception as e:
        print(f"❌ Error loading test.parquet: {e}")
        return
    
    print("Using custom token estimation for code analysis")
    
    # Compute token lengths
    print("\nComputing token lengths...")
    test_df = compute_token_lengths(test_df)
    
    # Generate statistics
    print("\nGenerating statistics...")
    stats = generate_token_statistics(test_df)
    
    # Print results
    print("\n" + "="*70)
    print("TOKEN LENGTH DISTRIBUTION")
    print("="*70)
    
    for stat in stats:
        print(f"   {stat['range']} tokens: {stat['percentage']:.4f} (n={stat['count']})")
    
    # Save to file
    output_file = "test_dataset_stats.txt"
    with open(output_file, 'w') as f:
        f.write("TEST DATASET TOKEN LENGTH STATISTICS\n")
        f.write("="*50 + "\n\n")
        for stat in stats:
            f.write(f"   {stat['range']} tokens: {stat['percentage']:.4f} (n={stat['count']})\n")
    
    print(f"\n✅ Statistics saved to {output_file}")
    
    # Additional summary statistics
    print(f"\nAdditional Statistics:")
    print(f"   Total samples: {len(test_df)}")
    print(f"   Mean token length: {test_df['tok_len'].mean():.1f}")
    print(f"   Median token length: {test_df['tok_len'].median():.1f}")
    print(f"   Min token length: {test_df['tok_len'].min()}")
    print(f"   Max token length: {test_df['tok_len'].max()}")
    print(f"   Std token length: {test_df['tok_len'].std():.1f}")

if __name__ == "__main__":
    main()
