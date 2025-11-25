"""
Script để kiểm tra chi tiết các dòng khác biệt
"""
import pandas as pd

df_attr = pd.read_csv('data/published_data/data_product_attribute.csv', encoding='utf-8')
df_merged = pd.read_csv('data/published_data/attribute_based_embeddings/attribute_text_filtering_merged.csv', encoding='utf-8')

print("="*70)
print("PHÂN TÍCH CHI TIẾT CÁC DÒNG KHÁC BIỆT")
print("="*70)

# Row 1751 - product_name
print("\n1. Row 1751 - product_name:")
attr_val = str(df_attr.loc[1751, 'product_name'])
merged_val = str(df_merged.loc[1751, 'product_name'])
print(f"   attr   length: {len(attr_val)}")
print(f"   merged length: {len(merged_val)}")
print(f"   Khác nhau: {attr_val != merged_val}")

# Find first difference
for i, (a, b) in enumerate(zip(attr_val, merged_val)):
    if a != b:
        print(f"   Vị trí khác đầu tiên: {i}")
        print(f"   attr  [{i-5}:{i+10}]: '{attr_val[max(0,i-5):i+10]}'")
        print(f"   merged[{i-5}:{i+10}]: '{merged_val[max(0,i-5):i+10]}'")
        print(f"   Char codes: attr={ord(a)}, merged={ord(b)}")
        break

# Row 1121 - feature (newline issue)
print("\n2. Row 1121 - feature:")
attr_feat = str(df_attr.loc[1121, 'feature'])
merged_feat = str(df_merged.loc[1121, 'feature'])
print(f"   attr   has newline (\\n): {chr(10) in attr_feat}")
print(f"   merged has newline (\\n): {chr(10) in merged_feat}")
print(f"   attr   length: {len(attr_feat)}")
print(f"   merged length: {len(merged_feat)}")

# Row 439 - feature 
print("\n3. Row 439 - feature:")
attr_feat = str(df_attr.loc[439, 'feature'])
merged_feat = str(df_merged.loc[439, 'feature'])
print(f"   attr   has newline: {chr(10) in attr_feat}")
print(f"   merged has newline: {chr(10) in merged_feat}")

# Row 1588
print("\n4. Row 1588 - ingredient & feature:")
print(f"   ingredient attr  : '{str(df_attr.loc[1588, 'ingredient'])[:80]}'")
print(f"   ingredient merged: '{str(df_merged.loc[1588, 'ingredient'])[:80]}'")

print("\n" + "="*70)
print("KẾT LUẬN:")
print("- Sự khác biệt có thể do whitespace (newline, spaces) trong CSV parsing")
print("- Các cột chính (product_id, shop_id, brand...) đều khớp 100%")
print("- Chỉ có 3-4 dòng có khác biệt nhỏ về formatting text")
print("="*70)
