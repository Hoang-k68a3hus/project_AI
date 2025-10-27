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
│   │   ├── als_metrics.json
│   │   └── als_metadata.json
│   ├── v2_20250116_141500/
│   └── ...
├── bpr/
│   ├── v1_20250115_120000/
│   └── ...
└── registry.json  # Central registry
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

## Registry Operations

### 1. Register New Model

#### Function: `register_model(artifacts_path, metadata)`

#### Inputs
- **artifacts_path**: Path tới model folder (e.g., `artifacts/cf/als/v2_...`)
- **metadata**: Dict với:
  - Model type (als/bpr)
  - Hyperparameters
  - Metrics
  - Training info
  - Data version
  - Git commit

#### Workflow
1. **Validate artifacts**: Check tất cả required files tồn tại
   - `*_U.npy`, `*_V.npy`
   - `*_params.json`, `*_metrics.json`, `*_metadata.json`
2. **Generate model_id**: `{type}_v{version}` (e.g., `als_v2_20250116_141500`)
3. **Load registry.json** (or create nếu không tồn tại)
4. **Add entry** vào `models` dict
5. **Update metadata**: `num_models`, `last_updated`
6. **Save registry.json** với pretty print (indent=2)

#### Error Handling
- **Missing files** → Raise error với list of missing files
- **Duplicate model_id** → Warning, skip registration (hoặc overwrite với flag)
- **Invalid metrics** → Validate ranges (e.g., Recall ∈ [0,1])

### 2. Select Best Model

#### Function: `select_best_model(metric='ndcg@10', min_improvement=0.1)`

#### Inputs
- **metric**: Metric để compare (e.g., `ndcg@10`, `recall@10`)
- **min_improvement**: Minimum improvement vs baseline (default 10%)

#### Workflow
1. **Load registry.json**
2. **Filter models**: 
   - Status = "active" (exclude archived/failed)
   - Baseline improvement ≥ min_improvement
3. **Sort** models theo metric (descending)
4. **Select top**: Model với highest metric
5. **Update `current_best`** section
6. **Set status** của old best model → "archived" (optional)
7. **Save registry.json**

#### Output
- **model_id**: Best model identifier
- **model_info**: Full metadata của best model

#### Logging
```
[INFO] Selected new best model: als_v2_20250116_141500
[INFO] Metric: ndcg@10 = 0.195 (prev: 0.189)
[INFO] Improvement: +3.2% over previous best
```

### 3. Load Model for Serving

#### Function: `load_model_from_registry(model_id=None)`

#### Inputs
- **model_id**: Optional, nếu None → load current_best

#### Workflow
1. **Load registry.json**
2. **Get model_id**: 
   - If provided → validate exists
   - Else → use `current_best.model_id`
3. **Get model path** từ registry
4. **Load artifacts**:
   - `U = np.load(path + '/*_U.npy')`
   - `V = np.load(path + '/*_V.npy')`
   - `params = json.load(path + '/*_params.json')`
5. **Load mappings** (từ data processing, không lưu trong model folder)
   - Path: `data/processed/user_item_mappings.json`
   - Validate data_version matches model's `data_version`
6. **Return** dict:
   ```python
   {
     'model_id': 'als_v2_...',
     'model_type': 'als',
     'U': np.array,
     'V': np.array,
     'params': dict,
     'mappings': dict,
     'metadata': dict
   }
   ```

#### Error Handling
- **Model not found** → Raise KeyError
- **Data version mismatch** → Warning (mappings stale, may cause errors)
- **Missing files** → Raise FileNotFoundError

### 4. List Models

#### Function: `list_models(model_type=None, status=None, sort_by='created_at')`

#### Inputs
- **model_type**: Filter by 'als' hoặc 'bpr' (None = all)
- **status**: Filter by status (active/archived/failed)
- **sort_by**: Sort column (created_at, ndcg@10, recall@10)

#### Output
- **DataFrame**: Với columns:
  - model_id, model_type, version, status
  - recall@10, ndcg@10, coverage
  - created_at, training_time_seconds

#### Usage
```python
# List all active ALS models
als_models = list_models(model_type='als', status='active')
print(als_models.sort_values('ndcg@10', ascending=False))

# Find best BPR model
best_bpr = list_models(model_type='bpr', status='active').iloc[0]
```

### 5. Archive Model

#### Function: `archive_model(model_id)`

#### Workflow
1. Load registry
2. Validate model_id exists
3. Check nếu model_id == current_best → warning (cannot archive active best)
4. Update `status` → "archived"
5. Save registry

#### Purpose
- **Cleanup**: Mark old experiments
- **Storage**: Optionally move files tới archive location
- **Safety**: Keep metadata even if artifacts deleted

### 6. Delete Model

#### Function: `delete_model(model_id, delete_files=False)`

#### Workflow
1. Validate model_id != current_best (prevent deleting active model)
2. Remove entry từ `models` dict
3. If `delete_files=True`:
   - Delete model folder (U, V, params, etc.)
   - Warning + confirmation prompt
4. Update metadata (num_models)
5. Save registry

#### Safety
- **Confirmation**: Prompt user nếu delete_files=True
- **Backup**: Suggest backup trước khi delete

## Versioning Strategy

### Version Identifier Format

#### Pattern: `v{N}_{YYYYMMDD}_{HHMMSS}`
- **N**: Sequential number (v1, v2, v3, ...)
- **Timestamp**: Khi nào model được trained

#### Examples
- `v1_20250115_103000` → Version 1, Jan 15 2025 10:30:00
- `v2_20250116_141500` → Version 2, Jan 16 2025 14:15:00

### Data Versioning

#### Data Hash Tracking
- **Source**: Hash từ `data/processed/versions.json` (Task 01)
- **Purpose**: Link model tới exact data version
- **Validation**: Warn nếu serving với different data version

#### Retrain Triggers
- **Data drift detected** (Task 06 monitoring)
- **New data available** (scheduled refresh)
- **Manual retrain** (hyperparameter tuning)

### Code Versioning

#### Git Commit Hash
- **Capture**: `git rev-parse HEAD` khi training
- **Purpose**: Reproducibility, rollback code
- **Usage**: Checkout exact commit để re-run training

## Model Comparison

### Function: `compare_models(model_ids, metrics=['recall@10', 'ndcg@10'])`

#### Output: DataFrame
```
model_id              | recall@10 | ndcg@10 | training_time | factors
als_v1_20250115...    | 0.234     | 0.189   | 45.2          | 64
als_v2_20250116...    | 0.245     | 0.195   | 102.8         | 128
bpr_v1_20250115...    | 0.242     | 0.192   | 1824.5        | 64
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

## Integration with Training Pipeline

### Modified Training Script

#### `scripts/train_cf.py` (Updated)
```python
def main():
    # ... training code ...
    
    # Save artifacts
    save_artifacts(U, V, params, metrics, metadata, output_path)
    
    # Register model
    from recsys.cf.registry import register_model, select_best_model
    
    model_id = register_model(
        artifacts_path=output_path,
        metadata={
            'model_type': args.model,
            'hyperparameters': params,
            'metrics': metrics,
            'training_info': {...},
            'data_version': data_hash,
            'git_commit': get_git_commit()
        }
    )
    
    # Auto-select if improvement
    if args.auto_select:
        best = select_best_model(metric='ndcg@10')
        print(f"Selected best model: {best['model_id']}")
```

## Dependencies

```python
# requirements_registry.txt
numpy>=1.23.0
pandas>=1.5.0
pyyaml>=6.0  # For config
gitpython>=3.1.0  # For git commit hash
```

## Timeline Estimate

- **Implementation**: 1.5 days
- **Testing**: 0.5 day
- **Integration**: 0.5 day
- **Documentation**: 0.5 day
- **Total**: ~3 days

## Success Criteria

- [ ] Registry tracks all trained models với metadata
- [ ] Best model selection automated (ndcg@10)
- [ ] Load model from registry works cho serving
- [ ] Versioning tracks data + code hashes
- [ ] Audit log records all registry operations
- [ ] Rollback mechanism tested (restore old best)
- [ ] Scripts integrated với training pipeline
- [ ] Documentation complete với examples
