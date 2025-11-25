# Product ID Mapping Bug - Root Cause & Solution

## 🔴 Critical Bug Identified

**Issue**: Mapping file stores **contiguous indices** instead of **real product IDs**.

### Current (Buggy) Mapping:
```json
{
  "item_to_idx": {"0": 0, "1": 1, "2": 2, ...},  // contiguous_idx → contiguous_idx (WRONG!)
  "idx_to_item": {"0": 0, "1": 1, "2": 2, ...}   // contiguous_idx → contiguous_idx (WRONG!)
}
```

### Should Be:
```json
{
  "item_to_idx": {"740187": 0, "8587034871": 1, ...},  // real_product_id → contiguous_idx
  "idx_to_item": {"0": 740187, "1": 8587034871, ...}    // contiguous_idx → real_product_id
}
```

## 🔍 Root Cause

**Problem**: `interactions_df['product_id']` was already converted to contiguous indices (0, 1, 2...) **BEFORE** `IDMapper.create_mappings()` was called.

**Result**: 
- IDMapper receives `product_id = [0, 1, 2, ...]` instead of real product IDs
- Creates mapping: `0→0, 1→1, 2→2...` (useless!)
- Cannot map between training matrix and enriched products (which use real product IDs)

## 📊 Impact

1. **BERT Coverage Issue**: Cannot match training items with enriched products
   - Training uses contiguous indices: [0, 1, 2...]
   - Enriched products use real IDs: [740187, 8587034871...]
   - **No overlap** → 0% match when comparing by real product IDs

2. **Cannot Fix BERT Coverage**: 
   - Scripts cannot identify which products are missing
   - Cannot filter training matrix correctly
   - Cannot regenerate BERT embeddings for correct products

3. **Model Serving Issues**:
   - Cannot map predictions back to real product IDs
   - Cannot serve recommendations correctly

## ✅ Solutions

### Option 1: Fix Mapping File (Recommended if you have raw interactions)

**Requirements**: Raw interactions CSV file with real product IDs

**Steps**:
```bash
# 1. Find raw interactions file
#    Usually at: data/published_data/interactions.csv
#    Or: data/raw/interactions.csv

# 2. Fix mapping
python scripts/fix_product_id_mapping.py \
    --interactions data/published_data/interactions.csv \
    --backup

# 3. Re-run Task 01 data pipeline
python -m recsys.cf.data.data --output data/processed

# 4. Re-generate BERT embeddings (if needed)
python scripts/generate_bert_embeddings.py

# 5. Re-train models
```

**What it does**:
- Loads raw interactions to get real product IDs
- Creates correct mapping: real_product_id → contiguous_index
- Updates mapping file
- Creates backup of original

### Option 2: Re-run Task 01 Pipeline (Recommended if Option 1 fails)

**Steps**:
```bash
# 1. Ensure raw interactions file exists with real product IDs
#    Check: data/published_data/interactions.csv

# 2. Delete processed data (optional, to start fresh)
rm -rf data/processed/*

# 3. Re-run complete Task 01 pipeline
python -m recsys.cf.data.data --output data/processed

# 4. Verify mapping is correct
python scripts/check_product_id_mapping.py

# 5. Re-generate BERT embeddings
python scripts/generate_bert_embeddings.py

# 6. Re-train models
```

**What it does**:
- Re-processes raw data from scratch
- Creates correct mapping from real product IDs
- Ensures all steps use correct IDs

### Option 3: Manual Fix (If you know the mapping)

If you have a way to map contiguous indices to real product IDs:

```python
import json

# Load current mapping
with open('data/processed/user_item_mappings.json', 'r') as f:
    mapping = json.load(f)

# Create correct mapping (you need to provide real_product_ids)
real_product_ids = [...]  # List of real product IDs in order

new_item_to_idx = {str(pid): idx for idx, pid in enumerate(real_product_ids)}
new_idx_to_item = {str(idx): pid for idx, pid in enumerate(real_product_ids)}

mapping['item_to_idx'] = new_item_to_idx
mapping['idx_to_item'] = new_idx_to_item

# Save
with open('data/processed/user_item_mappings.json', 'w') as f:
    json.dump(mapping, f, indent=2)
```

## 🔧 Scripts Available

1. **`scripts/check_product_id_mapping.py`**
   - Diagnose mapping bug
   - Compare mapping with enriched products
   - Identify if mapping is buggy

2. **`scripts/fix_product_id_mapping.py`**
   - Fix mapping from raw interactions
   - Creates backup
   - Updates mapping file

3. **`scripts/analyze_bert_coverage.py`**
   - Analyze BERT coverage (after mapping is fixed)
   - Identify missing products

## 📋 Verification Steps

After fixing mapping, verify:

```bash
# 1. Check mapping is correct
python scripts/check_product_id_mapping.py

# Expected output:
#   ✅ Mapping looks correct: real_product_id → contiguous_idx
#   ✅ Some mapping keys match enriched product_ids

# 2. Check BERT coverage
python scripts/analyze_bert_coverage.py

# Expected output:
#   ✅ Training in Enriched: > 0 (should match)
#   ✅ BERT in Enriched: > 0 (should match)
```

## 🚨 Important Notes

1. **After fixing mapping, you MUST**:
   - Re-run Task 01 data pipeline (to update matrices with correct IDs)
   - Re-generate BERT embeddings (if product list changed)
   - Re-train all models (matrices have different structure)

2. **Backup everything** before fixing:
   ```bash
   cp data/processed/user_item_mappings.json data/processed/user_item_mappings_BACKUP.json
   cp data/processed/X_train_confidence.npz data/processed/X_train_confidence_BACKUP.npz
   ```

3. **Root cause prevention**:
   - Ensure `interactions_df['product_id']` contains **real product IDs** before calling `IDMapper.create_mappings()`
   - Check data pipeline to find where product_id is converted to contiguous indices

## 🔗 Related Files

- `recsys/cf/data/processing/id_mapping.py` - IDMapper class
- `recsys/cf/data/data.py` - DataProcessor pipeline
- `data/processed/user_item_mappings.json` - Mapping file (needs fix)
- `data/processed/enriched_products.parquet` - Source of truth for real product IDs

## 📝 Next Steps

1. ✅ Identify bug (DONE)
2. ⏳ Fix mapping file (use script or re-run pipeline)
3. ⏳ Verify mapping is correct
4. ⏳ Re-run Task 01 pipeline
5. ⏳ Re-generate BERT embeddings
6. ⏳ Fix BERT coverage issue
7. ⏳ Re-train models

