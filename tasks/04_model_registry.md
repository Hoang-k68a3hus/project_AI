# Task 04: Model Registry & Versioning

## Mục Tiêu

Xây dựng hệ thống quản lý phiên bản models, theo dõi performance, và chọn "best model" cho production serving. Registry hỗ trợ rollback, A/B testing, và audit trail cho reproducibility.

## Registry Architecture

```
artifacts/cf/
├── als/
│   ├── v1_20250115_103000/
│   │   ├── als_U.npy
│   │   ├── als_V.npy
│   │   ├── als_params.json
│   │   └── als_metadata.json
│   ├── v2_20250116_141500/
│   └── ...
├── bpr/
│   ├── v1_20250115_120000/
│   └── ...
├── registry.json          # CF models registry
└── bert_registry.json     # BERT embeddings registry (separate)
```

### Required Model Files

The registry validates that all required files exist before registration:

```python
from recsys.cf.registry import REQUIRED_MODEL_FILES

# For ALS models:
REQUIRED_MODEL_FILES['als'] = [
    'als_U.npy',           # User embeddings
    'als_V.npy',           # Item embeddings
    'als_params.json',     # Hyperparameters
    'als_metadata.json'    # Training metadata
]

# For BPR models:
REQUIRED_MODEL_FILES['bpr'] = [
    'bpr_U.npy',
    'bpr_V.npy',
    'bpr_params.json',
    'bpr_metadata.json'
]

# For BERT-enhanced ALS:
REQUIRED_MODEL_FILES['bert_als'] = [
    'bert_als_U.npy',
    'bert_als_V.npy',
    'bert_als_params.json',
    'bert_als_metadata.json'
]
```

### Model Status Constants

```python
from recsys.cf.registry import MODEL_STATUS

MODEL_STATUS = {
    'active': 'active',      # Currently available for selection
    'archived': 'archived',  # Archived, not available for selection
    'failed': 'failed'       # Training/validation failed
}
```

## Registry Schema

### File: `artifacts/cf/registry.json`

#### Structure
```json
{
  "current_best": {
    "model_id": "als_v2_20250116_141500",
    "model_type": "als",
    "version": "v2_20250116_141500",
    "path": "artifacts/cf/als/v2_20250116_141500",
    "selection_metric": "ndcg@10",
    "selection_value": 0.195,
    "selected_at": "2025-01-16T14:30:00",
    "selected_by": "auto"  // or username
  },
  
  "models": {
    "als_v1_20250115_103000": {
      "model_type": "als",
      "version": "v1_20250115_103000",
      "path": "artifacts/cf/als/v1_20250115_103000",
      "created_at": "2025-01-15T10:30:00",
      "data_version": "abc123...",  // hash từ Task 01
      "git_commit": "def456...",
      
      "hyperparameters": {
        "factors": 64,
        "regularization": 0.01,
        "iterations": 15,
        "alpha": 40
      },
      
      "metrics": {
        "recall@10": 0.234,
        "recall@20": 0.312,
        "ndcg@10": 0.189,
        "ndcg@20": 0.221,
        "coverage": 0.287
      },
      
      "baseline_comparison": {
        "baseline_type": "popularity",
        "improvement_recall@10": 0.614,  // 61.4%
        "improvement_ndcg@10": 0.853
      },
      
      "training_info": {
        "training_time_seconds": 45.2,
        "num_users": 12000,
        "num_items": 2200,
        "num_train_interactions": 320000,
        "num_test_users": 11500
      },
      
      "status": "archived"  // active, archived, failed
    },
    
    "als_v2_20250116_141500": {
      "model_type": "als",
      "version": "v2_20250116_141500",
      "path": "artifacts/cf/als/v2_20250116_141500",
      "created_at": "2025-01-16T14:15:00",
      "data_version": "abc123...",
      "git_commit": "ghi789...",
      
      "hyperparameters": {
        "factors": 128,
        "regularization": 0.01,
        "iterations": 20,
        "alpha": 60
      },
      
      "metrics": {
        "recall@10": 0.245,
        "recall@20": 0.325,
        "ndcg@10": 0.195,
        "ndcg@20": 0.229,
        "coverage": 0.310
      },
      
      "baseline_comparison": {
        "baseline_type": "popularity",
        "improvement_recall@10": 0.690,
        "improvement_ndcg@10": 0.912
      },
      
      "training_info": {
        "training_time_seconds": 102.8,
        "num_users": 12000,
        "num_items": 2200,
        "num_train_interactions": 320000,
        "num_test_users": 11500
      },
      
      "status": "active"
    },
    
    "bpr_v1_20250115_120000": {
      "model_type": "bpr",
      "version": "v1_20250115_120000",
      "path": "artifacts/cf/bpr/v1_20250115_120000",
      "created_at": "2025-01-15T12:00:00",
      "data_version": "abc123...",
      "git_commit": "def456...",
      
      "hyperparameters": {
        "factors": 64,
        "learning_rate": 0.05,
        "regularization": 0.0001,
        "epochs": 50,
        "samples_per_epoch": 5
      },
      
      "metrics": {
        "recall@10": 0.242,
        "recall@20": 0.321,
        "ndcg@10": 0.192,
        "ndcg@20": 0.228,
        "coverage": 0.301
      },
      
      "baseline_comparison": {
        "baseline_type": "popularity",
        "improvement_recall@10": 0.669,
        "improvement_ndcg@10": 0.882
      },
      
      "training_info": {
        "training_time_seconds": 1824.5,
        "num_users": 12000,
        "num_items": 2200,
        "num_train_interactions": 320000,
        "num_test_users": 11500,
        "num_samples_trained": 80000000  // 50 epochs * 1.6M samples
      },
      
      "status": "active"
    }
  },
  
  "metadata": {
    "registry_version": "1.0",
    "last_updated": "2025-01-16T14:30:00",
    "num_models": 3,
    "selection_criteria": "ndcg@10"
  }
}
```

## Module Architecture

### Package Structure

```
recsys/cf/registry/
├── __init__.py          # Package exports
├── registry.py          # ModelRegistry class
├── model_loader.py      # ModelLoader class
├── bert_registry.py     # BERTEmbeddingsRegistry class
└── utils.py             # Utility functions
```

### Core Classes

1. **`ModelRegistry`**: Main registry for CF models (ALS/BPR)
2. **`ModelLoader`**: Load models for serving with caching and hot-reload
3. **`BERTEmbeddingsRegistry`**: Separate registry for BERT embeddings
4. **Utility Functions**: Versioning, git, hashing, backup, metadata helpers

### Import Usage

```python
from recsys.cf.registry import (
    ModelRegistry, ModelLoader, BERTEmbeddingsRegistry,
    get_registry, get_loader, get_bert_registry,
    register_model, select_best_model,
    generate_version_id, get_git_commit
)
```

## Registry Operations

### 1. Register New Model

#### Class: `ModelRegistry`

##### Method: `register_model()`

```python
from recsys.cf.registry import ModelRegistry

registry = ModelRegistry(registry_path='artifacts/cf/registry.json')

model_id = registry.register_model(
    artifacts_path='artifacts/cf/als/v2_20250116_141500',
    model_type='als',
    hyperparameters={
        'factors': 128,
        'regularization': 0.01,
        'iterations': 20,
        'alpha': 60
    },
    metrics={
        'recall@10': 0.245,
        'ndcg@10': 0.195,
        'coverage': 0.310
    },
    training_info={
        'training_time_seconds': 102.8,
        'num_users': 12000,
        'num_items': 2200
    },
    data_version='abc123...',
    git_commit='ghi789...',
    baseline_comparison={
        'baseline_type': 'popularity',
        'improvement_ndcg@10': 0.912
    },
    bert_integration={  # Optional, for BERT-enhanced models
        'embeddings_version': 'bert_v1_20250115',
        'projection_method': 'svd',
        'explained_variance': 0.873
    },
    version=None,  # Auto-generated if None
    overwrite=False
)
```

##### Workflow
1. **Validate artifacts**: Uses `validate_artifacts()` to check required files
   - For ALS: `als_U.npy`, `als_V.npy`, `als_params.json`, `als_metadata.json`
   - For BPR: `bpr_U.npy`, `bpr_V.npy`, `bpr_params.json`, `bpr_metadata.json`
   - For BERT-ALS: `bert_als_*` files
2. **Generate version**: Uses `generate_version_id()` → `v_{YYYYMMDD}_{HHMMSS}`
3. **Generate model_id**: `{model_type}_{version}` (e.g., `als_v2_20250116_141500`)
4. **Load registry**: Auto-creates if doesn't exist (if `auto_create=True`)
5. **Check duplicates**: Skips if exists (unless `overwrite=True`)
6. **Get git commit**: Auto-detects if `git_commit=None`
7. **Create entry**: Adds to `models` dict with all metadata
8. **Update metadata**: `num_models`, `last_updated`
9. **Save registry**: Pretty-printed JSON with `indent=2`
10. **Audit log**: Writes to `logs/registry_audit.log`

##### Error Handling
- **Missing files** → `ValueError` with list of missing files
- **Duplicate model_id** → Warning log, returns existing ID (unless `overwrite=True`)
- **Invalid model_type** → `ValueError` for unknown types

### 2. Select Best Model

#### Method: `select_best_model()`

```python
best = registry.select_best_model(
    metric='ndcg@10',
    min_improvement=0.1,  # 10% minimum improvement over baseline
    model_type=None  # Filter by type (None = all types)
)

# Returns:
# {
#     'model_id': 'als_v2_20250116_141500',
#     'model_info': {...},  # Full model metadata
#     'metric': 'ndcg@10',
#     'value': 0.195
# }
```

##### Workflow
1. **Filter candidates**: 
   - Status = "active" (excludes archived/failed)
   - Optional: Filter by `model_type`
   - Optional: Check baseline improvement ≥ `min_improvement`
2. **Extract metrics**: Get metric value from each model's `metrics` dict
3. **Sort**: Descending by metric value
4. **Select top**: Model with highest metric
5. **Compare with previous**: Calculate improvement percentage
6. **Update `current_best`**: Set new best model info
7. **Update metadata**: `selection_criteria` = metric name
8. **Save registry**: Persist changes
9. **Audit log**: Log selection with improvement details

##### Logging
```
[INFO] Selected best model: als_v2_20250116_141500 (ndcg@10=0.1950)
[INFO] 2025-01-16 14:30:00 | SELECT_BEST | als_v2_20250116_141500 | ndcg@10=0.1950 improvement=+3.2%
```

### 3. Load Model for Serving

#### Class: `ModelLoader`

##### Purpose
Load models from registry with caching, hot-reload, and thread-safe operations.

##### Initialization

```python
from recsys.cf.registry import ModelLoader, get_loader

# Option 1: Direct instantiation
loader = ModelLoader(
    registry_path='artifacts/cf/registry.json',
    cache_enabled=True,  # Enable embedding cache
    auto_load=True  # Auto-load current best on init
)

# Option 2: Singleton pattern (recommended for serving)
loader = get_loader()  # Returns singleton instance
```

##### Method: `load_current_best()`

```python
U, V, metadata = loader.load_current_best()

# Returns:
# - U: np.ndarray (num_users, factors) - User embeddings
# - V: np.ndarray (num_items, factors) - Item embeddings
# - metadata: dict - Model metadata
```

##### Method: `load_model(model_id)`

```python
U, V, metadata = loader.load_model('als_v2_20250116_141500')
```

##### Method: `reload_model()` - Hot Reload

```python
model_changed = loader.reload_model()  # Returns True if model changed

# Workflow:
# 1. Clear cache
# 2. Reload registry from disk
# 3. Load new current_best
# 4. Compare with previous model_id
# 5. Return True if different
```

##### Method: `get_current_model()` - Get State

```python
state = loader.get_current_model()  # Returns ModelState dataclass

# ModelState contains:
# - model_id, model_type, version
# - U, V (embeddings)
# - params, metadata
# - loaded_at, path
```

##### Method: `get_embeddings()` - Quick Access

```python
U, V = loader.get_embeddings()  # Get current embeddings
```

##### Method: `get_stats()` - Loader Statistics

```python
stats = loader.get_stats()

# Returns:
# {
#     'total_loads': 10,
#     'cache_hits': 8,
#     'cache_misses': 2,
#     'cache_hit_rate': 0.8,
#     'reload_count': 3,
#     'last_load_time_ms': 45.2,
#     'last_reload_at': '2025-01-16T14:30:00',
#     'cached_models': ['als_v2_...']
# }
```

##### Convenience Function

```python
from recsys.cf.registry import load_model_from_registry

# Load current best
U, V, metadata = load_model_from_registry()

# Load specific model
U, V, metadata = load_model_from_registry(model_id='als_v1_...')
```

##### Error Handling
- **Model not found** → `KeyError` with model_id
- **Missing files** → `FileNotFoundError` with file path
- **No current best** → `ValueError` if registry empty
- **Cache miss** → Loads from disk and updates cache

### 4. List Models

#### Method: `list_models()`

```python
import pandas as pd

# List all models
df = registry.list_models()

# Filter and sort
als_models = registry.list_models(
    model_type='als',
    status='active',
    sort_by='ndcg@10',
    ascending=False
)

# DataFrame columns:
# - model_id, model_type, version, status, created_at
# - recall@10, recall@20, ndcg@10, ndcg@20, coverage (from metrics)
# - training_time (from training_info)
```

##### Usage Examples

```python
# Find best BPR model
bpr_models = registry.list_models(model_type='bpr', status='active')
best_bpr = bpr_models.iloc[0]  # Already sorted by sort_by

# Compare all active models
active = registry.list_models(status='active', sort_by='ndcg@10')
print(active[['model_id', 'ndcg@10', 'recall@10']])
```

### 5. Archive Model

#### Method: `archive_model()`

```python
success = registry.archive_model('als_v1_20250115_103000')

# Returns: True if successful, False if current_best (cannot archive)
```

##### Workflow
1. **Validate**: Check model_id exists in registry
2. **Check current_best**: Cannot archive active best model → returns `False` with warning
3. **Update status**: Set `status = "archived"`
4. **Save registry**: Persist changes
5. **Audit log**: Log archive action

##### Purpose
- **Cleanup**: Mark old experiments without deleting
- **Safety**: Keep metadata for audit trail
- **Filtering**: Archived models excluded from best model selection

### 6. Delete Model

#### Method: `delete_model()`

```python
success = registry.delete_model(
    model_id='als_v1_20250115_103000',
    delete_files=False  # If True, also deletes artifact files
)
```

##### Workflow
1. **Validate**: Check model_id exists
2. **Check current_best**: Cannot delete active best → raises `ValueError`
3. **Remove from registry**: Delete entry from `models` dict
4. **Update metadata**: Decrement `num_models`
5. **Delete files** (if `delete_files=True`): Remove model folder with `shutil.rmtree()`
6. **Save registry**: Persist changes
7. **Audit log**: Log deletion with `delete_files` flag

##### Safety
- **Current best protection**: Cannot delete active best model
- **File deletion**: Optional, requires explicit `delete_files=True`
- **Audit trail**: All deletions logged

## Versioning Strategy

### Version Identifier Format

#### Pattern: `{prefix}_{YYYYMMDD}_{HHMMSS}`
- **prefix**: Model type prefix (e.g., `v`, `als`, `bpr`, `bert`)
- **Timestamp**: Training timestamp

#### Generation

```python
from recsys.cf.registry.utils import generate_version_id, parse_version_id

# Generate version
version = generate_version_id(prefix='v')  # → 'v_20250116_141500'
version = generate_version_id(prefix='als')  # → 'als_20250116_141500'

# Parse version
parsed = parse_version_id('v_20250116_141500')
# Returns: {
#     'raw': 'v_20250116_141500',
#     'prefix': 'v',
#     'date': '20250116',
#     'time': '141500',
#     'datetime': '2025-01-16T14:15:00'
# }

# Compare versions
from recsys.cf.registry.utils import compare_versions
cmp = compare_versions('v_20250115_103000', 'v_20250116_141500')  # → -1 (v1 < v2)
```

#### Examples
- `v_20250115_103000` → Version, Jan 15 2025 10:30:00
- `als_20250116_141500` → ALS model, Jan 16 2025 14:15:00

### Data Versioning

#### Data Hash Tracking

```python
from recsys.cf.registry.utils import (
    compute_data_version,
    compute_file_hash,
    compute_directory_hash
)

# Compute data version from multiple files
data_version = compute_data_version([
    'data/processed/interactions.parquet',
    'data/processed/user_item_mappings.json'
])

# Compute file hash
file_hash = compute_file_hash('data/processed/interactions.parquet', algorithm='md5')

# Compute directory hash
dir_hash = compute_directory_hash('data/processed', extensions=['.parquet', '.json'])
```

- **Source**: Hash from `data/processed/versions.json` (Task 01) or computed
- **Purpose**: Link model to exact data version
- **Validation**: Check data_version when loading mappings

#### Retrain Triggers
- **Data drift detected** (Task 06 monitoring)
- **New data available** (scheduled refresh)
- **Manual retrain** (hyperparameter tuning)

### Code Versioning

#### Git Commit Hash

```python
from recsys.cf.registry.utils import (
    get_git_commit,
    get_git_commit_short,
    get_git_branch,
    is_git_clean
)

# Get full commit hash
commit = get_git_commit()  # → 'def4567890abcdef...'

# Get short hash (7 chars)
commit_short = get_git_commit_short()  # → 'def4567'

# Get branch name
branch = get_git_branch()  # → 'main'

# Check if working tree is clean
clean = is_git_clean()  # → True/False
```

- **Auto-capture**: `ModelRegistry.register_model()` auto-detects if `git_commit=None`
- **Purpose**: Reproducibility, rollback code
- **Usage**: Checkout exact commit to re-run training

## Model Comparison

### Method: `compare_models()`

```python
comparison_df = registry.compare_models(
    model_ids=['als_v1_20250115_103000', 'als_v2_20250116_141500', 'bpr_v1_20250115_120000'],
    metrics=['recall@10', 'ndcg@10', 'coverage']
)

# Output DataFrame:
# model_id              | model_type | factors | recall@10 | ndcg@10 | coverage | training_time
# als_v1_20250115...   | als        | 64      | 0.234     | 0.189   | 0.287    | 45.2
# als_v2_20250116...   | als        | 128     | 0.245     | 0.195   | 0.310    | 102.8
# bpr_v1_20250115...   | bpr        | 64      | 0.242     | 0.192   | 0.301    | 1824.5
```

### Method: `get_registry_stats()`

```python
stats = registry.get_registry_stats()

# Returns:
# {
#     'total_models': 3,
#     'active_models': 2,
#     'archived_models': 1,
#     'by_type': {'als': 2, 'bpr': 1},
#     'current_best': 'als_v2_20250116_141500',
#     'last_updated': '2025-01-16T14:30:00'
# }
```

### Visualization: Side-by-Side Bar Chart
- **X-axis**: Model IDs
- **Y-axis**: Metrics (recall@10, ndcg@10)
- **Grouped bars**: Metrics side-by-side per model

## A/B Testing Support

### Canary Deployment

#### Scenario
- **Current production**: als_v1
- **New candidate**: als_v2
- **Strategy**: Serve v2 tới 10% traffic, monitor metrics

#### Registry Extension
```json
{
  "ab_test": {
    "test_id": "ab_als_v2_2025_01_16",
    "start_time": "2025-01-16T15:00:00",
    "control_model": "als_v1_20250115_103000",
    "treatment_model": "als_v2_20250116_141500",
    "traffic_split": 0.1,  // 10% treatment
    "status": "running"
  }
}
```

#### Service Integration
- **Random assignment**: Hash user_id → group (control/treatment)
- **Logging**: Track which model served each request
- **Analysis**: Compare online metrics (CTR, conversion) per group

### Rollback Mechanism

#### Trigger
- **Online metrics degraded** (e.g., CTR drop >5%)
- **Errors/latency spike**

#### Process
1. **Stop A/B test**: Set traffic_split = 0 (100% control)
2. **Update current_best** → previous model
3. **Investigate**: Check logs, offline metrics
4. **Fix or archive** treatment model

## Automation Scripts

### Script: `scripts/update_registry.py`

#### Usage
```bash
python scripts/update_registry.py \
  --model-path artifacts/cf/als/v2_20250116_141500 \
  --auto-select  # Automatically select as best if metrics improve
```

#### Workflow
1. Read model artifacts (params, metrics, metadata)
2. Register model in registry.json
3. If `--auto-select`:
   - Compare với current_best
   - Select nếu ndcg@10 improves
4. Log results

### Script: `scripts/cleanup_old_models.py`

#### Usage
```bash
python scripts/cleanup_old_models.py \
  --keep-last 5 \  # Keep 5 most recent versions per type
  --archive-old    # Archive instead of delete
```

#### Workflow
1. List all models per type (als/bpr)
2. Sort by created_at (descending)
3. Keep top N, archive/delete rest
4. Preserve current_best (never delete)

## Audit Trail

### Log File: `logs/registry_audit.log`

#### Entries
```
2025-01-15 10:30:00 | REGISTER | als_v1_20250115_103000 | ndcg@10=0.189
2025-01-16 14:15:00 | REGISTER | als_v2_20250116_141500 | ndcg@10=0.195
2025-01-16 14:30:00 | SELECT_BEST | als_v2_20250116_141500 | improvement=+3.2%
2025-01-17 09:00:00 | ARCHIVE | als_v1_20250115_103000 | reason=superseded
```

#### Purpose
- **Traceability**: Who/when selected models
- **Debugging**: Investigate production issues
- **Compliance**: Audit trail cho model changes

## Utility Functions

### Backup & Restore

```python
from recsys.cf.registry.utils import (
    backup_registry,
    restore_registry,
    list_backups
)

# Create backup
backup_path = backup_registry('artifacts/cf/registry.json')

# List backups
backups = list_backups('artifacts/cf/registry.json')
# Returns: [{'path': ..., 'name': ..., 'size_bytes': ..., 'created_at': ...}, ...]

# Restore from backup
restore_registry(
    backup_path='artifacts/cf/registry_backup_20250116_120000.json',
    registry_path='artifacts/cf/registry.json',
    create_current_backup=True  # Backup current before restoring
)
```

### Artifact Management

```python
from recsys.cf.registry.utils import (
    copy_model_artifacts,
    cleanup_old_artifacts
)

# Copy model artifacts
copy_model_artifacts(
    src_path='artifacts/cf/als/v1_20250115',
    dest_path='artifacts/cf/als/v1_backup',
    model_type='als'
)

# Cleanup old artifacts (keep last 5 per type)
deleted = cleanup_old_artifacts(
    artifacts_dir='artifacts/cf',
    keep_count=5,
    keep_current_best=True,
    dry_run=True  # Set False to actually delete
)
```

### Metadata Utilities

```python
from recsys.cf.registry.utils import (
    create_model_metadata,
    load_model_metadata,
    save_model_metadata
)

# Create metadata
metadata = create_model_metadata(
    num_users=12000,
    num_items=2200,
    factors=128,
    model_type='als',
    data_version='abc123...',
    git_commit='def456...',
    score_range=(0.0, 1.5),
    extra={'custom_field': 'value'}
)

# Save metadata
save_model_metadata('artifacts/cf/als/v1', 'als', metadata)

# Load metadata
metadata = load_model_metadata('artifacts/cf/als/v1', 'als')
```

## Integration with Training Pipeline

### Modified Training Script

#### `scripts/train_cf.py` (Updated)
```python
from recsys.cf.registry import ModelRegistry, select_best_model
from recsys.cf.registry.utils import compute_data_version, get_git_commit

def main():
    # ... training code ...
    
    # Save artifacts
    save_artifacts(U, V, params, metrics, metadata, output_path)
    
    # Compute data version
    data_version = compute_data_version([
        'data/processed/interactions.parquet',
        'data/processed/user_item_mappings.json'
    ])
    
    # Register model
    registry = ModelRegistry()
    model_id = registry.register_model(
        artifacts_path=output_path,
        model_type=args.model,  # 'als' or 'bpr'
        hyperparameters=params,
        metrics=metrics,
        training_info={
            'training_time_seconds': elapsed_time,
            'num_users': num_users,
            'num_items': num_items,
            'num_train_interactions': num_train
        },
        data_version=data_version,
        git_commit=get_git_commit(),  # Auto-detected
        baseline_comparison={
            'baseline_type': 'popularity',
            'improvement_ndcg@10': improvement_pct
        },
        bert_integration=bert_info if args.bert_init else None
    )
    
    # Auto-select if improvement
    if args.auto_select:
        best = registry.select_best_model(metric='ndcg@10', min_improvement=0.05)
        if best:
            print(f"Selected best model: {best['model_id']}")
```

## BERT Embeddings Registry

### Separate Registry for BERT Embeddings

BERT embeddings are managed in a **separate registry** (`artifacts/cf/bert_registry.json`) to decouple embedding versioning from CF model versioning.

### Class: `BERTEmbeddingsRegistry`

#### Initialization

```python
from recsys.cf.registry import BERTEmbeddingsRegistry, get_bert_registry

# Option 1: Direct instantiation
bert_registry = BERTEmbeddingsRegistry(
    registry_path='artifacts/cf/bert_registry.json',
    auto_create=True
)

# Option 2: Singleton pattern
bert_registry = get_bert_registry()
```

#### Method: `register_embeddings()`

```python
version = bert_registry.register_embeddings(
    embedding_path='data/processed/content_based_embeddings',
    model_name='vinai/phobert-base',
    num_items=2244,
    embedding_dim=768,
    generation_config={
        'batch_size': 32,
        'max_length': 256
    },
    text_fields_used=['product_name', 'description', 'ingredients'],
    data_version='abc123...',
    git_commit='def456...',
    version=None  # Auto-generated if None
)
```

#### Method: `get_current_best()`

```python
best = bert_registry.get_current_best()

# Returns:
# {
#     'version': 'bert_20250115_103000',
#     'info': {
#         'version': 'bert_20250115_103000',
#         'path': 'data/processed/content_based_embeddings',
#         'model_name': 'vinai/phobert-base',
#         'num_items': 2244,
#         'embedding_dim': 768,
#         ...
#     }
# }
```

#### Method: `link_to_model()`

```python
# Link embeddings to a CF model that uses them
bert_registry.link_to_model(
    embedding_version='bert_20250115_103000',
    model_id='als_v2_20250116_141500'
)
```

#### Convenience Function: `load_bert_embeddings()`

```python
from recsys.cf.registry import load_bert_embeddings

# Load current best embeddings
embeddings, metadata = load_bert_embeddings()

# Load specific version
embeddings, metadata = load_bert_embeddings(version='bert_20250115_103000')

# Returns:
# - embeddings: np.ndarray (num_items, embedding_dim)
# - metadata: dict with embedding info
```

### BERT Registry Schema

#### File: `artifacts/cf/bert_registry.json`

```json
{
  "current_best": {
    "model_id": "als_v2_20250116_141500",
    "model_type": "als",
    "version": "v2_20250116_141500",
    "path": "artifacts/cf/als/v2_20250116_141500",
    "selection_metric": "ndcg@10",
    "selection_value": 0.195,
    "selected_at": "2025-01-16T14:30:00",
    "selected_by": "auto",
    
    "bert_embeddings": {
      "version": "v1_20250115_103000",
      "path": "data/processed/content_based_embeddings/product_embeddings.pt",
      "model_name": "vinai/phobert-base",
      "embedding_dim": 768
    }
  },
  
  "models": {
    "als_v2_20250116_141500": {
      "model_type": "als",
      "version": "v2_20250116_141500",
      "path": "artifacts/cf/als/v2_20250116_141500",
      "created_at": "2025-01-16T14:15:00",
      "data_version": "abc123...",
      "git_commit": "ghi789...",
      
      "hyperparameters": {
        "factors": 128,
        "regularization": 0.01,
        "iterations": 20,
        "alpha": 60,
        "bert_init_enabled": true
      },
      
      "metrics": {
        "recall@10": 0.245,
        "ndcg@10": 0.195,
        "coverage": 0.310,
        "diversity@10": 0.418,
        "semantic_alignment@10": 0.531,
        "cold_start_coverage": 0.142
      },
      
      "bert_integration": {
        "embeddings_version": "v1_20250115_103000",
        "embeddings_path": "data/processed/content_based_embeddings/product_embeddings.pt",
        "model_name": "vinai/phobert-base",
        "embedding_dim": 768,
        "projection_method": "svd",
        "projected_dim": 128,
        "explained_variance": 0.873,
        "initialization_strategy": "item_factors_only",
        "alignment_validated": true
      },
      
      "status": "active"
    }
  },
  
  "bert_embeddings": {
    "v1_20250115_103000": {
      "version": "v1_20250115_103000",
      "created_at": "2025-01-15T10:30:00",
      "model_name": "vinai/phobert-base",
      "embedding_dim": 768,
      "num_products": 2244,
      "files": {
        "product_embeddings": "data/processed/content_based_embeddings/product_embeddings.pt",
        "metadata": "data/processed/content_based_embeddings/embedding_metadata.json"
      },
      "data_hash": "abc123...",
      "git_commit": "def456...",
      "status": "active"
    }
  },
  
  "metadata": {
    "registry_version": "1.1",
    "last_updated": "2025-01-16T14:30:00",
    "num_models": 3,
    "num_embeddings": 1,
    "selection_criteria": "ndcg@10"
  }
}
```

### BERT Registry Operations

#### List Embeddings

```python
embeddings_list = bert_registry.list_embeddings(include_linked_models=True)

# Returns list of dicts:
# [
#     {
#         'version': 'bert_20250115_103000',
#         'model_name': 'vinai/phobert-base',
#         'num_items': 2244,
#         'embedding_dim': 768,
#         'created_at': '2025-01-15T10:30:00',
#         'is_current': True,
#         'linked_models': ['als_v2_20250116_141500']
#     },
#     ...
# ]
```

#### Delete Embeddings

```python
success = bert_registry.delete_embeddings(
    version='bert_20250115_103000',
    delete_files=False  # If True, also deletes embedding files
)

# Safety checks:
# - Cannot delete current best
# - Cannot delete if linked to any CF models
```

#### Get Statistics

```python
stats = bert_registry.get_stats()

# Returns:
# {
#     'total_embeddings': 2,
#     'current_best': 'bert_20250115_103000',
#     'total_linked_models': 1,
#     'last_updated': '2025-01-16T14:30:00'
# }
```

## Audit Trail

### Audit Log File: `logs/registry_audit.log`

The `ModelRegistry` class automatically logs all operations to an audit log file.

#### Log Format
```
{timestamp} | {action} | {model_id} | {details}
```

#### Example Entries
```
2025-01-15 10:30:00 | REGISTER | als_v1_20250115_103000 | ndcg@10=0.189
2025-01-16 14:15:00 | REGISTER | als_v2_20250116_141500 | ndcg@10=0.195
2025-01-16 14:30:00 | SELECT_BEST | als_v2_20250116_141500 | ndcg@10=0.1950 improvement=+3.2%
2025-01-17 09:00:00 | ARCHIVE | als_v1_20250115_103000 | reason=manual
2025-01-17 10:00:00 | DELETE | als_v1_20250115_103000 | delete_files=False
```

#### Actions Logged
- `REGISTER`: Model registration
- `SELECT_BEST`: Best model selection
- `ARCHIVE`: Model archiving
- `DELETE`: Model deletion
- `UPDATE_STATUS`: Status updates

## Dependencies

```python
# requirements_registry.txt
numpy>=1.23.0
pandas>=1.5.0

# BERT artifacts (for loading embeddings)
torch>=1.13.0

# Note: Git integration uses subprocess (no external dependency)
```

## Timeline Estimate

- **Implementation**: 1.5 days
- **Testing**: 0.5 day
- **Integration**: 0.5 day
- **Documentation**: 0.5 day
- **Total**: ~3 days

## Advanced Features

### ModelLoader Caching

The `ModelLoader` class implements in-memory caching for efficient serving:

```python
loader = ModelLoader(cache_enabled=True)

# First load: cache miss, loads from disk
U1, V1, _ = loader.load_model('als_v1_20250115_103000')  # ~50ms

# Second load: cache hit, returns cached embeddings
U2, V2, _ = loader.load_model('als_v1_20250115_103000')  # ~0.1ms

# Check cache stats
stats = loader.get_stats()
print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
```

### Thread-Safe Operations

`ModelLoader` uses `threading.RLock()` for thread-safe concurrent access:

```python
import threading

def worker():
    loader = get_loader()  # Singleton
    U, V, _ = loader.load_current_best()
    # Safe to use in multiple threads

# Multiple threads can safely access the same loader
threads = [threading.Thread(target=worker) for _ in range(10)]
for t in threads:
    t.start()
```

### Preloading Models

Preload multiple models into cache for faster switching:

```python
loader.preload_models([
    'als_v1_20250115_103000',
    'als_v2_20250116_141500',
    'bpr_v1_20250115_120000'
])
# Returns: 3 (number successfully loaded)
```

### Data Classes

#### `ModelState`
```python
@dataclass
class ModelState:
    model_id: str
    model_type: str
    version: str
    U: np.ndarray  # User embeddings
    V: np.ndarray  # Item embeddings
    params: Dict[str, Any]
    metadata: Dict[str, Any]
    loaded_at: str  # ISO timestamp
    path: str
```

#### `LoaderStats`
```python
@dataclass
class LoaderStats:
    total_loads: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    reload_count: int = 0
    last_load_time_ms: float = 0
    last_reload_at: Optional[str] = None
```

## Success Criteria

- [x] Registry tracks all trained models với metadata
- [x] Best model selection automated (ndcg@10)
- [x] Load model from registry works cho serving
- [x] Versioning tracks data + code hashes
- [x] Audit log records all registry operations
- [x] ModelLoader with caching and hot-reload
- [x] BERT embeddings separate registry
- [x] Utility functions for backup, restore, cleanup
- [x] Thread-safe operations
- [x] Documentation complete với examples
