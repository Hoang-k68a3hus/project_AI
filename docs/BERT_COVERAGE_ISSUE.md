# BERT Embedding Coverage Issue - Root Cause Analysis

## 📋 Tóm tắt vấn đề

**Vấn đề**: BERT embeddings không bao phủ 100% items trong training matrix.

**Hiện trạng**:
- BERT embeddings: 2,244 products
- Training matrix: 1,423 items
- Coverage: 94.0% (1,338/1,423 matched)
- **85 items trong training matrix không có BERT embeddings** (6.0%)

## 🔍 Root Cause Analysis

### Nguyên nhân chính

1. **BERT embeddings được tạo từ `enriched_products.parquet`**
   - File này chứa **TẤT CẢ products** trong database (2,244 products)
   - Được tạo từ merge giữa `products.csv` và `attributes.csv`
   - Không filter theo interactions

2. **Training matrix chỉ chứa products có interactions**
   - Được tạo từ `interactions.csv` sau khi filter users và items
   - Chỉ giữ lại products có ≥3 interactions (cold-start filtering)
   - Kết quả: 1,423 items

3. **Mismatch giữa 2 nguồn dữ liệu**
   - 85 products có interactions nhưng **KHÔNG có trong `enriched_products.parquet`**
   - Hoặc có trong enriched nhưng bị filter ra khi generate BERT embeddings
   - → Không có BERT embeddings cho 85 items này

### Tại sao products có interactions nhưng không có trong enriched_products.parquet?

Có thể do:

1. **Merge failure trong content enrichment**
   - Products không match với attributes (product_id mismatch)
   - Products bị drop do missing data
   - Products không có trong `products.csv` hoặc `attributes.csv`

2. **Data version mismatch**
   - `enriched_products.parquet` được tạo từ version cũ của data
   - Interactions data được update sau khi enrichment chạy
   - → New products có interactions nhưng chưa có embeddings

3. **Filtering trong BERT generation**
   - Script `generate_bert_embeddings.py` có thể filter một số products
   - Products với missing `bert_input_text` bị skip

## 💡 Giải pháp

### Option 1: Filter Training Matrix (Recommended) ✅

**Cách làm**: Loại bỏ 85 items không có BERT embeddings khỏi training matrix.

**Ưu điểm**:
- ✅ Đơn giản, nhanh
- ✅ Đảm bảo 100% coverage
- ✅ Không cần regenerate BERT embeddings
- ✅ Consistent data (tất cả items đều có BERT)

**Nhược điểm**:
- ❌ Mất 85 items (6% của training items)
- ❌ Cần re-run Task 01 data pipeline

**Implementation**:
```bash
# 1. Analyze coverage issue
python scripts/analyze_bert_coverage.py

# 2. Fix by filtering training matrix
python scripts/fix_bert_coverage.py --filter-training

# 3. Re-run Task 01 data pipeline
python -m recsys.cf.data.data --output data/processed
```

### Option 2: Regenerate BERT Embeddings

**Cách làm**: Tạo lại BERT embeddings chỉ cho products có trong training matrix.

**Ưu điểm**:
- ✅ Giữ lại tất cả training items
- ✅ Smaller embedding file

**Nhược điểm**:
- ❌ Cần regenerate embeddings (tốn thời gian)
- ❌ Nếu có products mới, phải regenerate lại

**Implementation**:
```python
# Filter enriched_products.parquet to only include training items
import pandas as pd
import torch

# Load training item IDs
with open('data/processed/user_item_mappings.json', 'r') as f:
    mappings = json.load(f)
training_item_ids = set(int(k) for k in mappings['item_to_idx'].keys())

# Filter enriched products
enriched_df = pd.read_parquet('data/processed/enriched_products.parquet')
enriched_df = enriched_df[enriched_df['product_id'].isin(training_item_ids)]

# Save filtered enriched products
enriched_df.to_parquet('data/processed/enriched_products_filtered.parquet')

# Regenerate BERT embeddings
from recsys.cf.data.processing.embedding_generator import EmbeddingGenerator

generator = EmbeddingGenerator()
generator.process_and_save(
    input_path='data/processed/enriched_products_filtered.parquet',
    output_path='data/processed/content_based_embeddings/product_embeddings.pt'
)
```

### Option 3: Use Zero Vectors (Current Workaround)

**Cách làm**: Giữ nguyên, dùng random initialization cho items không có BERT.

**Ưu điểm**:
- ✅ Không cần thay đổi gì
- ✅ Works immediately

**Nhược điểm**:
- ❌ 85 items không benefit từ BERT initialization
- ❌ Có thể gây NaN trong training (như đã thấy)

**Note**: Đây là cách hiện tại, nhưng gây ra NaN errors.

### Option 4: Fix Content Enrichment Pipeline

**Cách làm**: Đảm bảo tất cả products có interactions đều có trong `enriched_products.parquet`.

**Ưu điểm**:
- ✅ Fix root cause
- ✅ Prevent future issues

**Nhược điểm**:
- ❌ Cần investigate và fix merge logic
- ❌ Có thể phức tạp nếu có data quality issues

**Implementation**: Check `recsys/cf/data/processing/content_enrichment.py`

## 📊 Impact Analysis

### Nếu filter training matrix (Option 1):

- **Items removed**: 85 (6.0%)
- **Interactions removed**: ~X interactions (cần check)
- **Coverage after**: 100%
- **Training impact**: Minimal (6% items, nhưng có thể là cold items)

### Nếu regenerate BERT embeddings (Option 2):

- **Items kept**: 1,423 (100%)
- **BERT embeddings**: 1,423 (down from 2,244)
- **Coverage after**: 100%
- **Training impact**: None

## 🎯 Recommendation

**Recommended**: **Option 1 - Filter Training Matrix**

**Lý do**:
1. Đơn giản và nhanh nhất
2. Đảm bảo 100% coverage
3. 85 items (6%) có thể là cold items (ít interactions)
4. Loss nhỏ so với benefit của 100% coverage

**Steps**:
1. Run `scripts/analyze_bert_coverage.py` để confirm
2. Run `scripts/fix_bert_coverage.py --filter-training`
3. Re-run Task 01 data pipeline
4. Re-train BERT-Enhanced ALS
5. Verify 100% coverage

## 🔧 Scripts Available

1. **`scripts/analyze_bert_coverage.py`**
   - Analyze coverage issue
   - Identify root cause
   - Show unmatched items

2. **`scripts/fix_bert_coverage.py`**
   - Fix coverage by filtering training matrix
   - Create backups
   - Update mappings

## 📝 Notes

- **Current workaround** (zero vectors) gây NaN errors
- **Best practice**: Ensure 100% coverage trước khi train
- **Future**: Sync BERT generation với training data pipeline

## 🔗 Related Files

- `recsys/cf/data/processing/embedding_generator.py` - BERT embedding generation
- `recsys/cf/data/processing/content_enrichment.py` - Content enrichment
- `recsys/cf/model/bert_enhanced_als.py` - BERT-Enhanced ALS model
- `notebooks/Colab_BERT_ALS_Training_Complete.ipynb` - Training notebook

