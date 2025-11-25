"""
Script kiểm tra kết quả embedding mới
"""
import torch
import pandas as pd

print('='*70)
print('KIỂM TRA KẾT QUẢ EMBEDDING MỚI')
print('='*70)

# Load enriched data
df = pd.read_parquet('data/processed/enriched_products.parquet')
print(f'\n1. ENRICHED DATA:')
print(f'   - Số records: {len(df)}')
print(f'   - product_id range: {df["product_id"].min()} - {df["product_id"].max()}')
print(f'   - Có bert_input_text: {"bert_input_text" in df.columns}')
print(f'   - Sample bert_input_text:')
print(f'     "{df["bert_input_text"].iloc[0][:200]}..."')

# Load embeddings
emb_data = torch.load('data/processed/content_based_embeddings/product_embeddings.pt', weights_only=False)
print(f'\n2. EMBEDDINGS:')
print(f'   - Shape: {emb_data["embeddings"].shape}')
print(f'   - Số product_ids: {len(emb_data["product_ids"])}')
print(f'   - product_id range: {min(emb_data["product_ids"])} - {max(emb_data["product_ids"])}')
print(f'   - Model: {emb_data["metadata"]["model_name"]}')
print(f'   - Dimension: {emb_data["metadata"]["embedding_dim"]}')

# Check product_id alignment
print(f'\n3. KIỂM TRA ALIGNMENT:')
emb_ids = set(emb_data['product_ids'])
df_ids = set(df['product_id'].tolist())
print(f'   - IDs trong embeddings: {len(emb_ids)}')
print(f'   - IDs trong enriched data: {len(df_ids)}')
print(f'   - IDs khớp: {len(emb_ids & df_ids)}')

# Sample embedding
print(f'\n4. SAMPLE EMBEDDING (product_id=0):')
if 0 in emb_data['product_ids']:
    idx = emb_data['product_ids'].index(0)
    sample_emb = emb_data['embeddings'][idx]
    print(f'   - Index: {idx}')
    print(f'   - Embedding shape: {sample_emb.shape}')
    print(f'   - First 5 values: {sample_emb[:5].tolist()}')
    print(f'   - Mean: {sample_emb.mean():.4f}, Std: {sample_emb.std():.4f}')
else:
    print(f'   - product_id=0 not found!')
    print(f'   - First 5 product_ids: {emb_data["product_ids"][:5]}')

print('\n' + '='*70)
print('✅ EMBEDDING MỚI ĐÃ SẴN SÀNG!')
print('='*70)
