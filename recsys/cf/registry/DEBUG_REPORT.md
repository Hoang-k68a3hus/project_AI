# Báo Cáo Debug Module Registry

## Tổng Quan
Module `recsys/cf/registry` cung cấp hệ thống quản lý model registry cho CF models, bao gồm:
- Model registration và versioning
- Best model selection
- Model loading cho serving
- BERT embeddings management
- Utility functions

## Các Lỗi Đã Phát Hiện và Sửa

### 1. **registry.py** (4 lỗi)

#### Lỗi 1.1: Division by Zero trong Improvement Calculation
- **Vị trí**: Dòng 376
- **Vấn đề**: Khi `prev_value = 0`, phép chia sẽ gây lỗi hoặc kết quả không chính xác
- **Giải pháp**: Kiểm tra `prev_value > 0` trước khi chia, và xử lý trường hợp `prev_value = 0`
- **Mã sửa**:
```python
# Trước:
if prev_value:
    pct = ((best_value - prev_value) / prev_value) * 100

# Sau:
if prev_value is not None and prev_value > 0:
    pct = ((best_value - prev_value) / prev_value) * 100
elif prev_value is not None and prev_value == 0 and best_value > 0:
    improvement = "improvement=+inf%"
```

#### Lỗi 1.2: Unsafe Access đến current_best Dictionary
- **Vị trí**: Dòng 348-353
- **Vấn đề**: Không kiểm tra `prev_best` có phải là dict và có key `model_id` không
- **Giải pháp**: Thêm type checking và safe access
- **Mã sửa**:
```python
# Trước:
if prev_best:
    prev_value = self._registry['models'].get(
        prev_best['model_id'], {}
    ).get('metrics', {}).get(metric)

# Sau:
if prev_best and isinstance(prev_best, dict):
    prev_model_id = prev_best.get('model_id')
    if prev_model_id:
        prev_model = self._registry['models'].get(prev_model_id, {})
        prev_value = prev_model.get('metrics', {}).get(metric)
```

#### Lỗi 1.3: Unsafe Access trong archive_model
- **Vị trí**: Dòng 519-520
- **Vấn đề**: Không kiểm tra `current_best` có phải là dict
- **Giải pháp**: Thêm type checking
- **Mã sửa**:
```python
# Trước:
current_best = self._registry.get('current_best', {})
if current_best and current_best.get('model_id') == model_id:

# Sau:
current_best = self._registry.get('current_best')
if current_best and isinstance(current_best, dict) and current_best.get('model_id') == model_id:
```

#### Lỗi 1.4: Unsafe Access trong delete_model
- **Vị trí**: Dòng 552-553
- **Vấn đề**: Tương tự lỗi 1.3
- **Giải pháp**: Thêm type checking

### 2. **model_loader.py** (2 lỗi)

#### Lỗi 2.1: Thiếu Error Handling khi Registry không tồn tại
- **Vị trí**: Dòng 106
- **Vấn đề**: Nếu registry không tồn tại và `auto_create=False`, sẽ raise exception
- **Giải pháp**: Thêm try-except để tự động tạo registry nếu không tồn tại
- **Mã sửa**:
```python
# Trước:
self._registry = ModelRegistry(registry_path, auto_create=False)

# Sau:
try:
    self._registry = ModelRegistry(registry_path, auto_create=False)
except FileNotFoundError:
    logger.warning(f"Registry not found at {registry_path}, creating empty registry")
    self._registry = ModelRegistry(registry_path, auto_create=True)
```

#### Lỗi 2.2: Thiếu Validation khi Load Model Files
- **Vị trí**: Dòng 134-149
- **Vấn đề**: Không kiểm tra file tồn tại trước khi load, có thể gây lỗi không rõ ràng
- **Giải pháp**: Thêm validation và error messages rõ ràng
- **Mã sửa**:
```python
# Thêm checks:
if not path.exists():
    raise FileNotFoundError(f"Model path does not exist: {model_path}")

u_file = path / f"{prefix}_U.npy"
if not u_file.exists():
    raise FileNotFoundError(f"Missing file: {u_file}")
```

### 3. **bert_registry.py** (1 lỗi)

#### Lỗi 3.1: Thiếu weights_only Parameter trong torch.load
- **Vị trí**: Dòng 423, 429
- **Vấn đề**: PyTorch warning về security khi load files không có `weights_only`
- **Giải pháp**: Thêm `weights_only=False` và `map_location='cpu'` để explicit
- **Mã sửa**:
```python
# Trước:
embeddings = torch.load(path / 'product_embeddings.pt')

# Sau:
embeddings = torch.load(path / 'product_embeddings.pt', map_location='cpu', weights_only=False)
```

### 4. **utils.py** (2 lỗi)

#### Lỗi 4.1: Index Out of Range trong get_git_commit_short
- **Vị trí**: Dòng 141
- **Vấn đề**: Nếu commit hash ngắn hơn 7 ký tự, sẽ gây lỗi
- **Giải pháp**: Kiểm tra độ dài trước khi slice
- **Mã sửa**:
```python
# Trước:
return commit[:7] if commit else None

# Sau:
return commit[:7] if commit and len(commit) >= 7 else commit
```

#### Lỗi 4.2: Thiếu Error Handling trong compute_directory_hash
- **Vị trí**: Dòng 248
- **Vấn đề**: Có thể fail nếu file không đọc được (permission, locked, etc.)
- **Giải pháp**: Thêm try-except để handle gracefully
- **Mã sửa**:
```python
# Thêm:
try:
    file_hash = compute_file_hash(str(file_path))
    hasher.update(file_hash.encode())
except (OSError, IOError) as e:
    logger.warning(f"Could not hash file {file_path}: {e}")
    # Use filename only as fallback
    hasher.update(file_path.name.encode())
```

## Các Tính Năng Chính

### 1. ModelRegistry
- Register models với metadata đầy đủ
- Select best model dựa trên metrics
- List và compare models
- Archive và delete models
- Audit trail logging

### 2. ModelLoader
- Load models từ registry
- Cache embeddings trong memory
- Hot-reload không cần restart
- Thread-safe operations

### 3. BERTEmbeddingsRegistry
- Track embedding versions
- Link embeddings với CF models
- Version comparison

### 4. Utilities
- Version generation và parsing
- Git integration
- Hash computation
- Backup và restore

## Các Cải Tiến Đã Thực Hiện

1. ✅ Sửa division by zero errors
2. ✅ Thêm type checking cho dictionary access
3. ✅ Cải thiện error handling và messages
4. ✅ Thêm validation cho file operations
5. ✅ Fix PyTorch security warnings
6. ✅ Cải thiện edge case handling

## Khuyến Nghị

1. **Testing**: Nên tạo unit tests cho từng component
2. **Validation**: Thêm schema validation cho registry entries
3. **Performance**: Monitor cache hit rates
4. **Documentation**: Cập nhật examples với real use cases

## Files Đã Sửa

- `recsys/cf/registry/registry.py` - 4 fixes
- `recsys/cf/registry/model_loader.py` - 2 fixes
- `recsys/cf/registry/bert_registry.py` - 1 fix
- `recsys/cf/registry/utils.py` - 2 fixes

Tổng cộng: **9 lỗi đã được sửa**

