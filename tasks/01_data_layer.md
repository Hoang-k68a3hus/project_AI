# Task 01: Tầng Dữ Liệu (Data Layer)

## Mục Tiêu

Xây dựng pipeline xử lý dữ liệu ổn định, có khả năng tái tạo (reproducible) và hiệu năng cao cho hệ thống CF. Pipeline này sẽ chuyển đổi raw CSV thành các định dạng tối ưu cho training và serving.

## Input Data Sources

### Raw CSV Files
Tất cả nằm trong `data/published_data/`:

1. **data_reviews_purchase.csv**
   - Columns: `user_id`, `product_id`, `rating`, `comment`, `cmt_date`
   - Rows: ~369K interactions
   - Issues: Trùng lặp user-item, missing timestamps, inconsistent types

2. **data_product.csv**
   - Columns: `product_id`, `product_name`, `brand`, `type`, `price`, `avg_star`, `num_sold_time`, `processed_description`
   - Rows: 2,244 products
   - Usage: Popularity baseline, metadata enrichment

3. **data_product_attribute.csv**
   - Columns: `product_id`, `ingredient`, `feature`, `skin_type`, `capacity`, `design`, `brand`, `expiry`, `origin`
   - Rows: 2,244 products
   - Usage: Attribute-based filtering, reranking signals

4. **data_shop.csv**
   - Columns: Shop metadata
   - Usage: Optional shop-level features (future)

## Preprocessing Steps

### Step 1: Data Validation & Cleaning

#### 1.1 Load và Audit
- **Encoding**: Đọc CSV với `encoding='utf-8'` (Vietnamese characters)
- **Type enforcement**:
  - `user_id`, `product_id`: int
  - `rating`: float (validate range 1.0-5.0)
  - `cmt_date`: parse thành datetime (format: DD/MM/YYYY hoặc auto-detect)
- **Missing values**:
  - Drop rows với missing `user_id`, `product_id`, `rating`
  - Fill missing `cmt_date` với placeholder timestamp (cuối dataset)
  - Log số lượng rows bị drop

#### 1.2 Deduplication
- **Rule**: Mỗi (user_id, product_id) chỉ giữ 1 interaction
- **Strategy**: Giữ interaction có `cmt_date` mới nhất
- **Fallback**: Nếu `cmt_date` trùng → giữ rating cao nhất
- **Log**: Số lượng duplicates removed

#### 1.3 Outlier Detection
- **User activity**: Identify users với >500 interactions (potential bots/scrapers)
- **Item popularity**: Flag items với <3 interactions (very cold items)
- **Rating distribution**: Check for rating bias (e.g., >90% ratings = 5)
- **Action**: Log outliers, quyết định filter sau

**✅ IMPLEMENTED**: See `recsys/cf/data.py` - `DataValidator` class with complete validation pipeline

### Step 2: Positive Signal Definition

#### 2.1 Implicit Feedback Conversion
- **Primary threshold**: `rating >= 4` → implicit positive
- **Alternative**: Test với `rating >= 3` như sensitivity analysis
- **Binary matrix**: Create `is_positive` column (0/1)

#### 2.2 User/Item Filtering
- **Min user interactions**: Giữ users có ≥2 positives (để có train + test)
- **Min item interactions**: Giữ items có ≥5 positives (đủ signal, giảm noise)
- **Iterative filtering**: Lặp lại cho đến khi stable (filtering users → items → users...)
- **Log stats**:
  - Users before/after: X → Y (Z% kept)
  - Items before/after: A → B (C% kept)
  - Interactions before/after: M → N (P% kept)

**✅ IMPLEMENTED**: See `recsys/cf/data.py` - Functions: `create_positive_labels()`, `filter_users_items()`, `apply_positive_filtering_pipeline()`

### Step 3: ID Mapping (Contiguous Indexing)

#### 3.1 User Mapping
- **Original**: `user_id` (sparse integers, gaps, range 1-304708)
- **Mapped**: `u_idx` (contiguous 0 to num_users-1)
- **Dict structure**: 
  ```
  {
    "user_to_idx": {original_id: idx, ...},
    "idx_to_user": {idx: original_id, ...}
  }
  ```

#### 3.2 Item Mapping
- **Original**: `product_id` (range 0-2243)
- **Mapped**: `i_idx` (contiguous 0 to num_items-1)
- **Dict structure**: Tương tự user mapping

#### 3.3 Apply Mapping
- Add columns `u_idx`, `i_idx` vào interactions DataFrame
- Validate: Không có missing mappings

#### 3.4 Save Mappings
- **Format**: JSON (human-readable, dễ debug)
- **Location**: `data/processed/user_item_mappings.json`
- **Include metadata**:
  - Timestamp tạo mappings
  - Số lượng users/items
  - Hash của raw data

### Step 4: Temporal Split (Leave-One-Out)

#### 4.1 Sort Per User
- Group interactions theo `u_idx`
- Sort mỗi group theo `cmt_date` (ascending)
- Handle ties: Nếu `cmt_date` trùng → sort theo `rating` desc

#### 4.2 Train/Test Split
- **Train**: Tất cả interactions trừ latest positive
- **Test**: Latest positive interaction per user
- **Validation**: Optional - lấy 2nd latest làm val, 3rd+ làm train

#### 4.3 Edge Cases
- Users với chỉ 1 positive → không có test data → skip user
- Users với 0 positives sau filtering → đã removed ở Step 2.2

#### 4.4 Create Datasets
- `train_interactions`: DataFrame với columns [u_idx, i_idx, rating, is_positive, timestamp]
- `test_interactions`: Tương tự train
- `val_interactions`: Optional

### Step 5: Matrix Construction

#### 5.1 CSR Sparse Matrix (X_train)
- **Shape**: (num_users, num_items)
- **Values**: 
  - Binary version: 1 for positive interactions
  - Rating version: Actual rating values (for experiments)
- **Library**: `scipy.sparse.csr_matrix`
- **Memory**: CSR efficient cho sparse data (~369K nonzeros / 2244*304708 cells)

#### 5.2 User Positive Sets
- **Structure**: Dict `user_pos_train[u_idx] = set(i_idx, ...)`
- **Usage**: 
  - Negative sampling trong BPR (exclude positives)
  - Filtering seen items khi generate recommendations
  - Fast lookup O(1) thay vì DataFrame query

#### 5.3 Item Popularity
- **Count**: Số lần mỗi item xuất hiện trong train
- **Normalization**: Convert sang probability distribution (sum=1)
- **Usage**: Popularity-based negative sampling, baseline ranking

### Step 6: Save Processed Data

#### 6.1 Parquet Format
- **File**: `data/processed/interactions.parquet`
- **Content**: Full DataFrame với columns:
  - `user_id`, `product_id`, `u_idx`, `i_idx`
  - `rating`, `is_positive`, `timestamp`
  - `split` (train/val/test)
- **Advantages**: 
  - 10x faster read/write vs CSV
  - Type preservation
  - Compression (~50% size reduction)

#### 6.2 Mappings JSON
- **File**: `data/processed/user_item_mappings.json`
- **Structure**:
  ```json
  {
    "metadata": {
      "created_at": "2025-01-15T10:30:00",
      "num_users": 12000,
      "num_items": 2200,
      "data_hash": "abc123..."
    },
    "user_to_idx": {...},
    "idx_to_user": {...},
    "item_to_idx": {...},
    "idx_to_user": {...}
  }
  ```

#### 6.3 Matrix Files
- **X_train.npz**: CSR matrix trong scipy sparse format
- **user_pos_train.pkl**: Pickle dict với sets (fast serialization)
- **item_popularity.npy**: NumPy array với popularity scores

#### 6.4 Statistics Summary
- **File**: `data/processed/data_stats.json`
- **Content**:
  - Train/val/test sizes
  - Sparsity: nonzeros / (users * items)
  - Rating distribution
  - User/item interaction histograms (quantiles)
  - Filtered counts (users, items, interactions)

### Step 7: Data Versioning

#### 7.1 Hash Calculation
- **Method**: MD5 hash của raw CSV files (sorted concatenation)
- **Purpose**: Track data changes, invalidate stale models
- **Storage**: In mappings JSON and model artifacts

#### 7.2 Timestamp Tracking
- **Creation time**: Khi nào data được processed
- **Usage**: Detect stale data, schedule retraining

#### 7.3 Version Registry
- **File**: `data/processed/versions.json`
- **Structure**:
  ```json
  {
    "v1": {
      "hash": "abc123",
      "timestamp": "2025-01-15T10:30:00",
      "filters": {"min_user_pos": 2, "min_item_pos": 5},
      "files": ["interactions.parquet", "mappings.json", ...]
    }
  }
  ```

## Output Artifacts

### Primary Files
1. `data/processed/interactions.parquet` - Full interaction data
2. `data/processed/user_item_mappings.json` - ID mappings với metadata
3. `data/processed/X_train.npz` - Sparse CSR matrix
4. `data/processed/user_pos_train.pkl` - User positive sets
5. `data/processed/item_popularity.npy` - Popularity distribution

### Metadata Files
6. `data/processed/data_stats.json` - Statistics summary
7. `data/processed/versions.json` - Version tracking
8. `data/processed/preprocessing.log` - Processing logs

## Quality Checks

### Validation Tests
- [ ] No missing values trong key columns (u_idx, i_idx)
- [ ] u_idx range = [0, num_users-1], i_idx range = [0, num_items-1]
- [ ] CSR matrix shape matches (num_users, num_items)
- [ ] user_pos_train keys = all u_idx in train
- [ ] Test set: 1 interaction per user (hoặc 0 nếu user filtered)
- [ ] No data leakage: Test timestamps > Train timestamps per user
- [ ] Mappings reversible: user_to_idx → idx_to_user round-trip OK

### Performance Benchmarks
- [ ] Parquet load time: <5 seconds cho 369K rows
- [ ] CSR matrix construction: <2 seconds
- [ ] Mapping lookup: O(1) constant time

## Configuration Example

```yaml
# data_config.yaml
raw_data:
  base_path: "data/published_data"
  interactions: "data_reviews_purchase.csv"
  products: "data_product.csv"
  attributes: "data_product_attribute.csv"

preprocessing:
  rating_threshold: 4  # For positive labels
  min_user_positives: 2
  min_item_positives: 5
  dedup_strategy: "keep_latest"  # or "keep_highest_rating"
  
temporal_split:
  method: "leave_one_out"  # or "leave_k_out", "timestamp"
  validation: false  # Enable val set?

output:
  processed_path: "data/processed"
  format: "parquet"  # or "csv"
  save_matrix: true
  save_stats: true
  
versioning:
  enable: true
  hash_method: "md5"
```

## Module Interface

### Module: `recsys/cf/data.py`

#### Function 1: `load_raw_data(config)`
- **Input**: Config dict/object
- **Output**: Dict với keys: interactions_df, products_df, attributes_df
- **Purpose**: Load và validate raw CSVs

#### Function 2: `preprocess_interactions(df, config)`
- **Input**: Raw interactions DataFrame, config
- **Output**: Cleaned DataFrame (dedup, typed, filtered)
- **Steps**: Validation, deduplication, filtering

#### Function 3: `create_mappings(df, user_col, item_col)`
- **Input**: DataFrame, column names
- **Output**: Dict với user/item mappings và metadata
- **Purpose**: Contiguous ID mapping

#### Function 4: `temporal_split(df, method='leave_one_out')`
- **Input**: Interactions DataFrame với timestamps
- **Output**: train_df, test_df, (val_df optional)
- **Purpose**: Split data theo temporal logic

#### Function 5: `build_csr_matrix(df, num_users, num_items)`
- **Input**: Interactions với u_idx/i_idx, dimensions
- **Output**: scipy.sparse.csr_matrix
- **Purpose**: Efficient sparse matrix construction

#### Function 6: `build_user_pos_sets(df)`
- **Input**: Train interactions
- **Output**: Dict[u_idx, Set[i_idx]]
- **Purpose**: Fast positive item lookup per user

#### Function 7: `save_processed_data(artifacts, output_path)`
- **Input**: Dict với all artifacts (df, matrix, mappings, stats)
- **Output**: None (save to disk)
- **Purpose**: Persist processed data với versioning

#### Function 8: `load_processed_data(version=None)`
- **Input**: Optional version identifier
- **Output**: Dict với loaded artifacts
- **Purpose**: Load processed data cho training/serving

## Component 8: BERT/PhoBERT Embeddings Pipeline

### Purpose
Tích hợp BERT embeddings vào data layer để hỗ trợ hybrid reranking và content-based fallback.

### Step 1: Extract Product Descriptions

#### Load Product Text Data
```python
  products_df = pd.read_csv('data/published_data/data_product.csv', encoding='utf-8')

# Combine fields for rich text representation
products_df['combined_text'] = (
    products_df['product_name'] + ' [SEP] ' +
    products_df['brand'].fillna('') + ' [SEP] ' +
    products_df['processed_description'].fillna('')
)
```

### Step 2: Generate BERT Embeddings

#### Module: `recsys/bert/embedding_generator.py`

```python
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm

class BERTEmbeddingGenerator:
    """
    Generate BERT/PhoBERT embeddings cho product descriptions.
    """
    
    def __init__(self, model_name='vinai/phobert-base', device='cpu'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
    
    def encode_texts(self, texts, batch_size=32, max_length=256):
        """
        Encode texts thành embeddings.
        
        Args:
            texts: List of text strings
            batch_size: Batch size cho encoding
            max_length: Max token length
        
        Returns:
            np.array: (len(texts), hidden_dim) embeddings
        """
        embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size)):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors='pt'
                ).to(self.device)
                
                # Forward pass
                outputs = self.model(**encoded)
                
                # Mean pooling over sequence
                batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def save_embeddings(self, embeddings, product_ids, output_path):
        """
        Save embeddings với metadata.
        
        Args:
            embeddings: np.array (N, D)
            product_ids: List of product IDs
            output_path: Path to save .pt file
        """
        torch.save({
            'embeddings': torch.from_numpy(embeddings),
            'product_ids': product_ids,
            'model_name': self.tokenizer.name_or_path,
            'embedding_dim': embeddings.shape[1],
            'num_products': len(product_ids),
            'created_at': datetime.now().isoformat()
        }, output_path)
```

### Step 3: Embedding Generation Workflow

#### Script: `scripts/generate_bert_embeddings.py`
```python
"""
Generate BERT embeddings cho all products.

Usage:
    python scripts/generate_bert_embeddings.py \
        --model vinai/phobert-base \
        --output data/processed/content_based_embeddings/
"""

import argparse
import pandas as pd
from recsys.bert.embedding_generator import BERTEmbeddingGenerator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='vinai/phobert-base')
    parser.add_argument('--output', default='data/processed/content_based_embeddings/')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    
    # Load products
    products = pd.read_csv('data/published_data/data_product.csv', encoding='utf-8')
    products['combined_text'] = (
        products['product_name'] + ' [SEP] ' +
        products['brand'].fillna('') + ' [SEP] ' +
        products['processed_description'].fillna('')
    )
    
    # Generate embeddings
    generator = BERTEmbeddingGenerator(model_name=args.model)
    embeddings = generator.encode_texts(
        products['combined_text'].tolist(),
        batch_size=args.batch_size
    )
    
    # Save
    os.makedirs(args.output, exist_ok=True)
    output_file = os.path.join(args.output, 'product_embeddings.pt')
    generator.save_embeddings(
        embeddings,
        products['product_id'].tolist(),
        output_file
    )
    
    print(f"Saved {len(embeddings)} embeddings to {output_file}")
    print(f"Embedding dimension: {embeddings.shape[1]}")

if __name__ == '__main__':
    main()
```

### Step 4: User Profile Embeddings

#### Strategy 1: Interaction-Weighted Average
```python
def compute_user_profile_embedding(user_history_items, item_embeddings, item_to_idx):
    """
    Compute user profile bằng weighted average của item embeddings.
    
    Args:
        user_history_items: List[(product_id, weight)]
        item_embeddings: np.array (num_items, dim)
        item_to_idx: Dict mapping product_id -> idx
    
    Returns:
        np.array: (dim,) user profile embedding
    """
    history_embeddings = []
    weights = []
    
    for product_id, weight in user_history_items:
        if product_id in item_to_idx:
            idx = item_to_idx[product_id]
            history_embeddings.append(item_embeddings[idx])
            weights.append(weight)
    
    if not history_embeddings:
        # Fallback: zero embedding hoặc mean embedding
        return np.zeros(item_embeddings.shape[1])
    
    # Weighted average
    history_embeddings = np.array(history_embeddings)
    weights = np.array(weights).reshape(-1, 1)
    weights = weights / weights.sum()  # Normalize
    
    user_profile = (history_embeddings * weights).sum(axis=0)
    return user_profile
```

#### Strategy 2: TF-IDF Weighted
```python
from sklearn.feature_extraction.text import TfidfVectorizer

def compute_user_tfidf_profile(user_history_texts, item_embeddings, product_ids):
    """
    Compute user profile bằng TF-IDF weighted item embeddings.
    """
    # Compute TF-IDF weights
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(user_history_texts)
    
    # Weight embeddings by TF-IDF importance
    # ... (implementation)
```

### Step 5: Embedding Versioning

#### Metadata File: `data/processed/content_based_embeddings/embedding_metadata.json`
```json
{
  "version": "v1_20250115_103000",
  "model_name": "vinai/phobert-base",
  "embedding_dim": 768,
  "num_products": 2244,
  "created_at": "2025-01-15T10:30:00",
  "data_hash": "abc123...",
  "git_commit": "def456...",
  "files": {
    "product_embeddings": "product_embeddings.pt",
    "user_profiles": "user_profile_embeddings.pt"
  }
}
```

### Step 6: Sync with CF Data

#### Validation: Check Alignment
```python
def validate_embedding_alignment(mappings_path, embeddings_path):
    """
    Validate BERT embeddings align với CF item mappings.
    """
    # Load mappings
    with open(mappings_path) as f:
        mappings = json.load(f)
    
    # Load embeddings
    embeddings_data = torch.load(embeddings_path)
    
    cf_product_ids = set(mappings['item_to_idx'].keys())
    bert_product_ids = set(str(pid) for pid in embeddings_data['product_ids'])
    
    # Check coverage
    missing_in_bert = cf_product_ids - bert_product_ids
    extra_in_bert = bert_product_ids - cf_product_ids
    
    print(f"CF products: {len(cf_product_ids)}")
    print(f"BERT products: {len(bert_product_ids)}")
    print(f"Missing in BERT: {len(missing_in_bert)}")
    print(f"Extra in BERT: {len(extra_in_bert)}")
    
    if missing_in_bert:
        warnings.warn(f"Warning: {len(missing_in_bert)} products have CF embeddings but no BERT embeddings")
```

## Dependencies

```python
# requirements_data.txt
pandas>=1.5.0
numpy>=1.23.0
scipy>=1.9.0
pyarrow>=10.0.0  # For parquet

# BERT dependencies
torch>=1.13.0
transformers>=4.25.0
sentencepiece>=0.1.96  # For PhoBERT tokenizer
```

## Error Handling

### Common Errors
1. **CSV encoding error** → Enforce UTF-8, log problematic rows
2. **Missing columns** → Raise clear error với expected schema
3. **Type conversion failure** → Log rows, fill with defaults hoặc drop
4. **Mapping collision** → Should not happen với unique IDs, validate
5. **Empty DataFrame after filtering** → Log warning, adjust thresholds

### Logging Strategy
- **Level INFO**: Summary stats (rows processed, filtered)
- **Level WARNING**: Outliers, missing data
- **Level ERROR**: Critical failures (corrupt file, schema mismatch)
- **Output**: `logs/data_processing.log` với rotation

## Monitoring Metrics

### Data Quality Metrics
- **Sparsity**: % of nonzero cells trong matrix
- **Coverage**: % users/items còn lại sau filtering
- **Rating distribution**: Mean, std, quantiles
- **Temporal spread**: Min/max timestamps, gaps

### Drift Detection (For Retraining)
- **Distribution shift**: KL divergence của rating distribution
- **Popularity shift**: Spearman correlation của item ranks
- **User growth**: % new users vs existing
- **Trigger**: Retrain nếu shift > threshold

## Timeline Estimate

- **Implementation**: 2-3 days
- **Testing & Validation**: 1 day
- **Documentation**: 0.5 day
- **Total**: ~4 days

## Success Criteria

- [ ] Pipeline chạy end-to-end không errors
- [ ] Output artifacts pass all quality checks
- [ ] Processing time <1 minute cho 369K interactions
- [ ] Reproducible: Same input → same output (với fixed random seed)
- [ ] Documented: Clear README và inline comments
