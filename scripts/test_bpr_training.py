"""
Test BPR Training Pipeline End-to-End
"""
import sys
sys.path.insert(0, 'D:/app/IAI/viecomrec')

import numpy as np
import pandas as pd
from recsys.cf.data import DataProcessor
from recsys.cf.model.bpr import (
    BPRDataLoader,
    TripletSampler,
    BPRModelInitializer,
    BPRTrainer,
    train_bpr_model
)

print("="*60)
print("BPR Training Pipeline Test")
print("="*60)

# Step 1: Load data using DataProcessor
print("\n--- Step 1: Load and Prepare Data ---")
processor = DataProcessor(base_path='data/published_data')
interactions_df = pd.read_parquet('data/processed/interactions.parquet')

# Filter to train split only
train_df = interactions_df[interactions_df['split'] == 'train'].copy()
print(f"Train interactions: {len(train_df):,}")

# Build positive sets
user_pos_sets = processor.build_bpr_positive_sets(train_df)
print(f"Users with positives: {len(user_pos_sets):,}")

# Build positive pairs
positive_pairs, pair_stats = processor.build_bpr_positive_pairs(train_df)
print(f"Positive pairs: {len(positive_pairs):,}")

# Get num_users and num_items
num_users = train_df['u_idx'].max() + 1
num_items = train_df['i_idx'].max() + 1
print(f"Num users: {num_users}, Num items: {num_items}")

# Step 2: Test Triplet Sampling
print("\n--- Step 2: Test Triplet Sampling ---")
sampler = TripletSampler(
    positive_pairs=positive_pairs,
    user_pos_sets=user_pos_sets,
    num_items=num_items,
    hard_neg_sets={},  # No hard negatives for this test
    hard_ratio=0.0,
    samples_per_positive=3,
    random_seed=42
)

triplets = sampler.sample_epoch()
print(f"Sampled triplets: {triplets.shape}")
print(f"First 5 triplets:\n{triplets[:5]}")

# Validate triplets
print("\nValidating triplets...")
invalid = 0
for u, i_pos, i_neg in triplets[:1000]:
    if i_neg in user_pos_sets.get(int(u), set()):
        invalid += 1
print(f"Invalid triplets (neg in pos set): {invalid}/1000")

# Step 3: Test Model Initialization
print("\n--- Step 3: Test Model Initialization ---")
initializer = BPRModelInitializer(
    num_users=num_users,
    num_items=num_items,
    factors=32,
    preset='fast'
)
U, V = initializer.initialize_embeddings()
print(f"Initialized U: {U.shape}, V: {V.shape}")
print(f"U stats: mean={U.mean():.6f}, std={U.std():.6f}")
print(f"V stats: mean={V.mean():.6f}, std={V.std():.6f}")

# Step 4: Test Training (quick run)
print("\n--- Step 4: Test Training (3 epochs) ---")
trainer = BPRTrainer(
    U=U.copy(),
    V=V.copy(),
    learning_rate=0.1,
    regularization=0.001
)

summary = trainer.fit(
    positive_pairs=positive_pairs,
    user_pos_sets=user_pos_sets,
    num_items=num_items,
    hard_neg_sets={},
    epochs=3,
    samples_per_positive=2,
    batch_size=2048,
    show_progress=True
)

print(f"\nTraining complete!")
print(f"Duration: {summary['total_duration_seconds']:.1f}s")
print(f"Final loss: {summary['final_loss']:.4f}")
print(f"U shape: {summary['U_shape']}")
print(f"V shape: {summary['V_shape']}")

# Step 5: Test Recommendations
print("\n--- Step 5: Test Recommendations ---")
U_trained, V_trained = trainer.get_embeddings()

# Sample a test user
test_user = list(user_pos_sets.keys())[0]
user_scores = U_trained[test_user] @ V_trained.T

# Mask seen items
for item in user_pos_sets[test_user]:
    user_scores[item] = -np.inf

top_10 = np.argsort(user_scores)[-10:][::-1]
print(f"User {test_user} training items: {sorted(list(user_pos_sets[test_user]))[:5]}")
print(f"Top 10 recommendations: {top_10}")

print("\n" + "="*60)
print("[SUCCESS] BPR Pipeline Test Complete!")
print("="*60)
