"""
Script để kiểm tra chi tiết file merged có khớp với data_product_attribute.csv không
"""
import pandas as pd

def check_merged_data():
    # Load both files
    df_attr = pd.read_csv('data/published_data/data_product_attribute.csv', encoding='utf-8')
    df_merged = pd.read_csv('data/published_data/attribute_based_embeddings/attribute_text_filtering_merged.csv', encoding='utf-8')

    print('='*70)
    print('SO SÁNH CHI TIẾT: data_product_attribute.csv vs merged file')
    print('='*70)

    # Common columns to compare
    common_cols = ['product_id', 'shop_id', 'product_name', 'ingredient', 'feature', 
                   'skin_type', 'capacity', 'design', 'brand', 'expiry', 'origin']

    print(f'\n1. KIỂM TRA SỐ LƯỢNG:')
    print(f'   - data_product_attribute.csv: {len(df_attr)} rows')
    print(f'   - merged file: {len(df_merged)} rows')

    print(f'\n2. SO SÁNH TỪNG CỘT:')
    all_match = True
    for col in common_cols:
        if col in df_attr.columns and col in df_merged.columns:
            match = (df_attr[col].fillna('').astype(str) == df_merged[col].fillna('').astype(str)).all()
            mismatch_count = (~(df_attr[col].fillna('').astype(str) == df_merged[col].fillna('').astype(str))).sum()
            status = "✅ KHỚP" if match else f"❌ KHÔNG KHỚP ({mismatch_count} rows)"
            print(f'   - {col}: {status}')
            if not match:
                all_match = False

    print(f'\n3. CHI TIẾT CÁC DÒNG KHÔNG KHỚP (nếu có):')
    has_diff = False
    for col in common_cols:
        if col in df_attr.columns and col in df_merged.columns:
            mask = df_attr[col].fillna('').astype(str) != df_merged[col].fillna('').astype(str)
            if mask.any():
                has_diff = True
                print(f'\n   Cột [{col}] - {mask.sum()} dòng khác:')
                diff_idx = mask[mask].index[:3]  # Show first 3
                for idx in diff_idx:
                    attr_val = str(df_attr.loc[idx, col])[:60]
                    merged_val = str(df_merged.loc[idx, col])[:60]
                    print(f'      Row {idx}:')
                    print(f'         attr  : "{attr_val}"')
                    print(f'         merged: "{merged_val}"')

    if not has_diff:
        print('   Không có dòng nào khác biệt!')

    # Check extra columns in merged
    print(f'\n4. CÁC CỘT BỔ SUNG TRONG MERGED FILE:')
    extra_cols = ['type', 'skin_kind', 'is_5_star', 'num_sold_time', 'price']
    for col in extra_cols:
        if col in df_merged.columns:
            null_count = df_merged[col].isna().sum()
            unique_count = df_merged[col].nunique()
            print(f'   - {col}: {unique_count} unique values, {null_count} nulls')

    # Sample comparison
    print(f'\n5. MẪU SO SÁNH (row 0, 100, 500, 1000, 2000):')
    sample_indices = [0, 100, 500, 1000, 2000]
    for idx in sample_indices:
        if idx < len(df_attr):
            print(f'\n   Row {idx}:')
            print(f'      product_id: attr={df_attr.loc[idx, "product_id"]} | merged={df_merged.loc[idx, "product_id"]}')
            print(f'      product_name: "{df_attr.loc[idx, "product_name"][:50]}..."')
            print(f'      is_5_star: {df_merged.loc[idx, "is_5_star"]:.4f}')
            print(f'      price: {df_merged.loc[idx, "price"]}')

    print('\n' + '='*70)
    if all_match:
        print('KẾT LUẬN: ✅ TẤT CẢ CÁC CỘT ĐỀU KHỚP!')
    else:
        print('KẾT LUẬN: ❌ CÓ CỘT KHÔNG KHỚP - CẦN KIỂM TRA LẠI!')
    print('='*70)


if __name__ == '__main__':
    check_merged_data()
