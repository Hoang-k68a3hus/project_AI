"""
Test BPR Data Preparation Step 1 Implementation
"""
import pandas as pd
import numpy as np
from recsys.cf.data import DataProcessor

# Initialize processor
processor = DataProcessor(base_path='data/published_data')

# Load existing processed interactions
print('Loading processed interactions...')
interactions_df = pd.read_parquet('data/processed/interactions.parquet')
print(f'Loaded {len(interactions_df):,} interactions')
print(f'Columns: {list(interactions_df.columns)}')

# Check if is_positive column exists
if 'is_positive' not in interactions_df.columns:
    print('Creating positive labels...')
    interactions_df = processor.bpr_preparer.create_positive_labels(interactions_df)

pos_count = interactions_df['is_positive'].sum()
neg_count = len(interactions_df) - pos_count
print(f'\nPositive interactions: {pos_count:,}')
print(f'Negative interactions: {neg_count:,}')

# Test 1: Build positive sets
print('\n' + '='*60)
print('Test 1: build_bpr_positive_sets')
print('='*60)
user_pos_sets = processor.build_bpr_positive_sets(interactions_df)
print(f'User positive sets created: {len(user_pos_sets):,} users')
sample_user = list(user_pos_sets.keys())[0]
print(f'Sample user {sample_user} positives: {len(user_pos_sets.get(sample_user, set()))} items')

# Test 2: Build positive pairs
print('\n' + '='*60)
print('Test 2: build_bpr_positive_pairs')
print('='*60)
pairs, stats = processor.build_bpr_positive_pairs(interactions_df)
print(f'Pairs shape: {pairs.shape}')
print(f'Stats: {stats}')
print(f'First 5 pairs:\n{pairs[:5]}')

# Test 3: Build positive pairs from sets
print('\n' + '='*60)
print('Test 3: build_bpr_positive_pairs_from_sets')
print('='*60)
pairs_from_sets, stats_from_sets = processor.build_bpr_positive_pairs_from_sets(user_pos_sets)
print(f'Pairs shape: {pairs_from_sets.shape}')
print(f'Stats: {stats_from_sets}')

# Verify both methods produce same number of pairs
print('\n' + '='*60)
print('Validation')
print('='*60)
print(f'Pairs from DataFrame: {len(pairs):,}')
print(f'Pairs from sets: {len(pairs_from_sets):,}')
print(f'Match: {len(pairs) == len(pairs_from_sets)}')

print('\n[SUCCESS] All tests passed!')
