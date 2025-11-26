# Báo Cáo Debug Module Search

## Tổng Quan
Module `service/search` cung cấp các tính năng tìm kiếm semantic cho sản phẩm mỹ phẩm sử dụng PhoBERT embeddings.

## Các Lỗi Đã Phát Hiện và Sửa

### 1. **query_encoder.py**

#### Lỗi 1.1: Regex Pattern cho Vietnamese Characters không đầy đủ
- **Vị trí**: Dòng 285
- **Vấn đề**: Regex pattern chỉ liệt kê một số ký tự tiếng Việt, có thể bỏ sót một số ký tự
- **Giải pháp**: Sử dụng Unicode ranges (`\u00C0-\u024F`, `\u1E00-\u1EFF`) để bao phủ đầy đủ các ký tự tiếng Việt
- **Mã sửa**:
```python
# Trước:
text = re.sub(r'[^\w\sàáảãạăắằẳẵặ...]', '', text)

# Sau:
text = re.sub(r'[^\w\s\u00C0-\u024F\u1E00-\u1EFF]', '', text, flags=re.UNICODE)
```

#### Lỗi 1.2: Xử lý Model Output không linh hoạt
- **Vị trí**: Dòng 340, 415
- **Vấn đề**: Code giả định model luôn trả về `last_hidden_state`, có thể fail với một số model
- **Giải pháp**: Thêm fallback để xử lý các format output khác nhau (`pooler_output`, tuple)
- **Mã sửa**:
```python
# Thêm kiểm tra và fallback
if hasattr(outputs, 'last_hidden_state'):
    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
elif hasattr(outputs, 'pooler_output'):
    embedding = outputs.pooler_output.cpu().numpy()[0]
else:
    embedding = outputs[0][:, 0, :].cpu().numpy()[0]
```

### 2. **smart_search.py**

#### Lỗi 2.1: Xử lý NaN trong String Operations
- **Vị trí**: Dòng 726, 728
- **Vấn đề**: `.str.lower()` sẽ fail nếu column chứa NaN values
- **Giải pháp**: Kiểm tra `notna()` trước khi thực hiện string operations
- **Mã sửa**:
```python
# Trước:
df = df[df['brand'].str.lower() == filters['brand'].lower()]

# Sau:
df = df[df['brand'].notna() & (df['brand'].astype(str).str.lower() == filters['brand'].lower())]
```

#### Lỗi 2.2: Division by Zero trong Quality Signal Normalization
- **Vị trí**: Dòng 634
- **Vấn đề**: Có thể chia cho 0 nếu `max_q == min_q`
- **Giải pháp**: Thêm kiểm tra trước khi chia
- **Mã sửa**:
```python
if max_q > min_q:
    signals['quality'] = (float(quality) - min_q) / (max_q - min_q)
else:
    signals['quality'] = 0.5  # Default if min == max
```

#### Lỗi 2.3: Xử lý Popularity Signal không an toàn
- **Vị trí**: Dòng 620-624
- **Vấn đề**: Không kiểm tra `max_pop > 0` và không convert sang float
- **Giải pháp**: Thêm kiểm tra và type conversion
- **Mã sửa**:
```python
if popularity and popularity > 0:
    max_pop = norm_config['popularity']['max_value']
    if max_pop > 0:
        signals['popularity'] = min(np.log1p(float(popularity)) / np.log1p(max_pop), 1.0)
    else:
        signals['popularity'] = 0.0
```

### 3. **search_index.py**

#### Lỗi 3.1: Thiếu kiểm tra Embeddings trước khi sử dụng
- **Vị trí**: Dòng 313, 439
- **Vấn đề**: Code có thể fail nếu `embeddings_norm` là None
- **Giải pháp**: Thêm kiểm tra trước khi sử dụng
- **Mã sửa**:
```python
if self.phobert_loader.embeddings_norm is None:
    logger.error("Embeddings not loaded. Cannot perform search.")
    return []
```

#### Lỗi 3.2: Thiếu kiểm tra khi build FAISS Index
- **Vị trí**: Dòng 179
- **Vấn đề**: Có thể fail nếu embeddings chưa được load hoặc rỗng
- **Giải pháp**: Thêm kiểm tra trước khi build index
- **Mã sửa**:
```python
if self.phobert_loader.embeddings_norm is None:
    logger.error("Embeddings not loaded. Cannot build FAISS index.")
    self.use_faiss = False
    return

if embeddings.size == 0:
    logger.error("Empty embeddings. Cannot build FAISS index.")
    self.use_faiss = False
    return
```

#### Lỗi 3.3: Thiếu kiểm tra khi build index
- **Vị trí**: Dòng 137
- **Vấn đề**: Có thể fail nếu `product_id_to_idx` rỗng sau khi load
- **Giải pháp**: Thêm kiểm tra sau khi load embeddings
- **Mã sửa**:
```python
if not self.phobert_loader.is_loaded() or not self.phobert_loader.product_id_to_idx:
    logger.error("Failed to load PhoBERT embeddings. Cannot build index.")
    return
```

#### Lỗi 3.4: Thiếu kiểm tra phobert_loader trong filtered search
- **Vị trí**: Dòng 425
- **Vấn đề**: Có thể fail nếu phobert_loader chưa được khởi tạo đúng
- **Giải pháp**: Thêm kiểm tra attribute trước khi sử dụng
- **Mã sửa**:
```python
if not hasattr(self.phobert_loader, 'product_id_to_idx') or not self.phobert_loader.product_id_to_idx:
    logger.error("PhoBERT loader not properly initialized. Cannot perform filtered search.")
    return []
```

## Các Tính Năng Chính

### 1. QueryEncoder
- Encode text queries thành embeddings sử dụng PhoBERT
- Hỗ trợ Vietnamese text preprocessing với abbreviation expansion
- LRU cache cho query embeddings
- Batch encoding cho hiệu suất tốt hơn

### 2. SearchIndex
- Exact cosine similarity search (cho catalog nhỏ)
- FAISS ANN search (tùy chọn, cho catalog lớn)
- Metadata filtering (brand, category, price range)
- Thread-safe operations

### 3. SmartSearchService
- Text-to-product semantic search
- Item-to-item similarity search
- User profile-based recommendations
- Hybrid search với attribute filters
- Multi-signal reranking (semantic, popularity, quality)

## Các Cải Tiến Đã Thực Hiện

1. ✅ Cải thiện xử lý Vietnamese text với Unicode ranges
2. ✅ Thêm error handling cho model output formats
3. ✅ Xử lý NaN values trong string operations
4. ✅ Tránh division by zero errors
5. ✅ Thêm validation checks cho embeddings và indices
6. ✅ Cải thiện error messages và logging

## Khuyến Nghị

1. **Testing**: Nên tạo unit tests cho từng component
2. **Performance**: Monitor cache hit rates và search latency
3. **Error Handling**: Có thể thêm retry logic cho model loading
4. **Documentation**: Cập nhật docstrings với examples cụ thể

## Files Đã Sửa

- `service/search/query_encoder.py` - 3 fixes
- `service/search/smart_search.py` - 3 fixes  
- `service/search/search_index.py` - 4 fixes

Tổng cộng: **10 lỗi đã được sửa**

