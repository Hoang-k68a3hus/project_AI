import shutil
from datetime import datetime
import os
import glob

# Backup CSV file
backup_name = f"data_reviews_purchase_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
backup_path = f"data/processed/{backup_name}"

shutil.copy('data/published_data/data_reviews_purchase.csv', backup_path)
print(f"✅ Backed up to: {backup_path}")

# Verify
size = os.path.getsize(backup_path)
print(f"   File size: {size / (1024*1024):.2f} MB")

# List all backups
backups = sorted(glob.glob('data/processed/data_reviews_purchase_backup_*.csv'))
print(f"\n📋 Total backups: {len(backups)}")
for b in backups[-5:]:  # Show last 5
    size_mb = os.path.getsize(b) / (1024*1024)
    print(f"   - {b.split(chr(92))[-1]} ({size_mb:.2f} MB)")
