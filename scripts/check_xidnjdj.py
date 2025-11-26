import pandas as pd

df = pd.read_csv('data/published_data/data_reviews_purchase.csv', encoding='utf-8')
row = df.iloc[250921]

print("Row 250921:")
print(f"user_id: {row['user_id']}")
print(f"product_id: {row['product_id']}")
print()
print(f"processed_comment: {row['processed_comment']}")
print()
print(f"corrected_comment: {row['corrected_comment']}")
print()

has_word_orig = 'xidnjđj' in str(row['processed_comment'])
has_word_corr = 'xidnjđj' in str(row['corrected_comment'])

print(f"Từ 'xidnjđj' trong processed: {has_word_orig}")
print(f"Từ 'xidnjđj' trong corrected: {has_word_corr}")
