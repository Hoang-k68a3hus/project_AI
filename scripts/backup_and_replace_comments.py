import pandas as pd
import json
from datetime import datetime
import os

# Change to project root
os.chdir('d:\\app\\IAI\\viecomrec')

# Load CSV
print("Loading CSV...")
df = pd.read_csv('data/published_data/data_reviews_purchase.csv', encoding='utf-8')
print(f"✅ Loaded {len(df):,} rows")

# Backup original processed_comment
print("\nBacking up processed_comment...")
backup_data = {
    'timestamp': datetime.now().isoformat(),
    'total_rows': len(df),
    'processed_comments': df['processed_comment'].tolist()
}

backup_file = f'data/processed/backup_processed_comment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
with open(backup_file, 'w', encoding='utf-8') as f:
    json.dump(backup_data, f, ensure_ascii=False, indent=2)
print(f"✅ Backed up to: {backup_file}")
print(f"   Size: {len(backup_data['processed_comments']):,} comments")

# Replace processed_comment with corrected_comment
print("\nReplacing processed_comment with corrected_comment...")
df['processed_comment'] = df['corrected_comment']
print(f"✅ Replaced {len(df):,} rows")

# Save updated CSV
print("Saving updated CSV...")
df.to_csv('data/published_data/data_reviews_purchase.csv', index=False, encoding='utf-8')
print(f"✅ Saved to data/published_data/data_reviews_purchase.csv")

# Verify
print("\nVerifying...")
df_verify = pd.read_csv('data/published_data/data_reviews_purchase.csv', encoding='utf-8')
print(f"✅ Total rows: {len(df_verify):,}")
print(f"✅ Columns: {list(df_verify.columns)}")

# Show sample
print(f"\n📝 Sample (first 3 rows):")
for i in range(min(3, len(df_verify))):
    print(f"   Row {i+1}: {df_verify.iloc[i]['processed_comment'][:60]}...")

print(f"\n✨ Done! Backup file: {backup_file}")
