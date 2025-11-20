# Data Layer Module - Refactored Structure

## Overview
Code đã được tổ chức lại thành cấu trúc class-based với 3 class chính:

## Structure

```
recsys/cf/data/
├── __init__.py          # Package exports
├── read_data.py         # DataReader class - Đọc và load CSV files
├── audit_data.py        # DataAuditor class - Validate và clean data
└── data.py             # DataProcessor class - Unified interface
```

## Classes

### 1. DataReader (read_data.py)
**Chức năng**: Load raw CSV files với UTF-8 encoding và schema validation

**Methods**:
- `load_all_data()` - Load tất cả files (interactions, products, attributes, shops)
- `load_interactions()` - Load chỉ interactions
- `load_products()` - Load chỉ products
- `load_attributes()` - Load chỉ attributes
- `load_shops()` - Load chỉ shops

**Example**:
```python
from recsys.cf.data import DataReader

reader = DataReader(base_path="data/published_data")
interactions = reader.load_interactions()
all_data = reader.load_all_data()
```

### 2. DataAuditor (audit_data.py)
**Chức năng**: Validate, clean và audit data quality

**Methods**:
- `validate_and_clean(df)` - Validate và clean interactions DataFrame
- `generate_quality_report(df, name)` - Tạo báo cáo chất lượng data
- `compute_data_hash(df)` - Tính MD5 hash cho versioning

**Validation Steps**:
1. Type enforcement (user_id, product_id → int; rating → float)
2. Missing value handling (drop missing critical fields)
3. Temporal validation (drop NaT timestamps - prevents data leakage)
4. Rating range validation ([1.0, 5.0])

**Example**:
```python
from recsys.cf.data import DataAuditor

auditor = DataAuditor(
    rating_min=1.0,
    rating_max=5.0,
    drop_missing_timestamps=True
)
df_clean, stats = auditor.validate_and_clean(df_raw)
auditor.generate_quality_report(df_clean, "Report Name")
```

### 3. DataProcessor (data.py)
**Chức năng**: Unified interface kết hợp DataReader và DataAuditor

**Methods**:
- `load_and_validate_interactions()` - Load và validate interactions
- `load_and_validate_all()` - Load và validate tất cả data
- `generate_quality_report(df, name)` - Tạo quality report
- `compute_data_hash(df)` - Tính data hash

**Example** (Recommended):
```python
from recsys.cf.data import DataProcessor

# Initialize processor
processor = DataProcessor(
    base_path="data/published_data",
    rating_min=1.0,
    rating_max=5.0,
    drop_missing_timestamps=True
)

# Load and validate interactions
df_clean, stats = processor.load_and_validate_interactions()

# Generate quality report
processor.generate_quality_report(df_clean, "Cleaned Interactions")

# Compute hash for versioning
data_hash = processor.compute_data_hash(df_clean)
```

## Backward Compatibility

Module vẫn support các convenience functions cho backward compatibility:

```python
from recsys.cf.data import (
    load_raw_data,
    validate_and_clean_interactions,
    compute_data_hash,
    log_data_quality_report
)

# Old style still works
data = load_raw_data("data/published_data")
df_clean, stats = validate_and_clean_interactions(data['interactions'])
```

## Test Results

**Test Date**: November 20, 2025

### Data Statistics:
- **Total interactions**: 369,099 rows
- **Products**: 2,244 rows
- **Attributes**: 2,244 rows
- **Shops**: 1,291 rows

### Validation Results:
- **Retention rate**: 100.00%
- **Missing user_id**: 0
- **Missing product_id**: 0
- **Missing rating**: 0
- **Missing timestamp**: 0
- **Invalid ratings**: 0

### Rating Distribution:
- Rating 1: 2,271 (0.62%)
- Rating 2: 1,308 (0.35%)
- Rating 3: 3,207 (0.87%)
- Rating 4: 9,147 (2.48%)
- Rating 5: 353,166 (95.68%)

## Benefits of Refactoring

✅ **Better Code Organization**: Tách biệt concerns (reading vs validation)  
✅ **Easier to Test**: Có thể test từng class riêng biệt  
✅ **More Maintainable**: Dễ dàng thêm features mới  
✅ **Cleaner API**: Class-based interface rõ ràng hơn  
✅ **Reusability**: Có thể reuse DataReader/DataAuditor ở các context khác  

## Next Steps (Task 01 Continuation)

Các bước tiếp theo theo Task 01:
- [ ] Step 1.2: Deduplication
- [ ] Step 1.3: Outlier Detection
- [ ] Step 2: Explicit Feedback Feature Engineering
- [ ] Step 3: ID Mapping (Contiguous Indexing)
- [ ] Step 4: Temporal Split (Leave-One-Out)
- [ ] Step 5: Matrix Construction
