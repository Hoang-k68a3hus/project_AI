"""
Convert Product IDs to Contiguous Indices (0, 1, 2...) by Product Name

This script converts product_id in attribute_text_filtering.csv to contiguous indices
based on product_name matching.

Usage:
    python scripts/convert_product_ids_by_name.py [--output OUTPUT_FILE]
"""

import sys
import pandas as pd
from pathlib import Path
import argparse

# Fix Windows encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def normalize_product_name(name: str) -> str:
    """
    Normalize product name for matching.
    
    Removes:
    - Extra spaces
    - Special characters (optional)
    - Case differences (optional)
    """
    if pd.isna(name):
        return ""
    
    # Convert to string and strip
    name = str(name).strip()
    
    # Optional: normalize case (uncomment if needed)
    # name = name.lower()
    
    # Optional: remove special characters (uncomment if needed)
    # import re
    # name = re.sub(r'[^\w\s]', '', name)
    
    return name

def create_name_to_id_mapping(df: pd.DataFrame) -> dict:
    """
    Create mapping from normalized product_name to new contiguous ID (0, 1, 2...).
    
    Same product_name → same new_id
    """
    print("\n" + "="*80)
    print("CREATING PRODUCT NAME → NEW ID MAPPING")
    print("="*80)
    
    # Normalize product names
    print("\nNormalizing product names...")
    df['normalized_name'] = df['product_name'].apply(normalize_product_name)
    
    # Get unique normalized names
    unique_names = df['normalized_name'].unique()
    print(f"Found {len(unique_names):,} unique product names")
    
    # Create mapping: normalized_name → new_id (0, 1, 2...)
    name_to_new_id = {name: idx for idx, name in enumerate(sorted(unique_names))}
    
    print(f"Created mapping for {len(name_to_new_id):,} unique products")
    print(f"  New ID range: [0, {len(name_to_new_id)-1}]")
    
    # Show sample mappings
    print(f"\nSample mappings (first 10):")
    for i, (name, new_id) in enumerate(list(name_to_new_id.items())[:10]):
        # Truncate long names
        display_name = name[:60] + "..." if len(name) > 60 else name
        print(f"  {new_id:4d} ← {display_name}")
    
    return name_to_new_id

def load_reference_products(reference_path: Path = None) -> pd.DataFrame:
    """
    Load reference products (enriched_products.parquet or products.csv) for mapping.
    
    Returns DataFrame with product_id and product_name columns.
    """
    if reference_path is None:
        # Try to find enriched_products.parquet
        base_path = Path(__file__).parent.parent
        possible_paths = [
            base_path / "data" / "processed" / "enriched_products.parquet",
            base_path / "data" / "published_data" / "products.csv",
            base_path / "data" / "raw" / "products.csv",
        ]
        
        for path in possible_paths:
            if path.exists():
                reference_path = path
                break
        
        if reference_path is None:
            return None
    
    print(f"\nLoading reference products from: {reference_path}")
    
    if reference_path.suffix == '.parquet':
        df_ref = pd.read_parquet(reference_path)
    else:
        try:
            df_ref = pd.read_csv(reference_path, encoding='utf-8')
        except UnicodeDecodeError:
            df_ref = pd.read_csv(reference_path, encoding='latin-1')
    
    # Check for product_name column
    name_cols = ['product_name', 'name', 'Name', 'PRODUCT_NAME']
    name_col = None
    for col in name_cols:
        if col in df_ref.columns:
            name_col = col
            break
    
    if name_col is None:
        print(f"  WARNING: No product_name column found. Available columns: {df_ref.columns.tolist()}")
        return None
    
    # Check for product_id column
    id_cols = ['product_id', 'id', 'Id', 'PRODUCT_ID']
    id_col = None
    for col in id_cols:
        if col in df_ref.columns:
            id_col = col
            break
    
    if id_col is None:
        print(f"  WARNING: No product_id column found. Available columns: {df_ref.columns.tolist()}")
        return None
    
    # Select relevant columns
    df_ref = df_ref[[id_col, name_col]].copy()
    df_ref.columns = ['product_id', 'product_name']
    df_ref = df_ref.dropna(subset=['product_name'])
    
    print(f"  Loaded {len(df_ref):,} reference products")
    print(f"  Product ID range: [{df_ref['product_id'].min()}, {df_ref['product_id'].max()}]")
    
    return df_ref

def convert_product_ids(
    input_path: Path,
    output_path: Path = None,
    reference_path: Path = None,
    create_backup: bool = True
) -> pd.DataFrame:
    """
    Convert product_id to contiguous indices based on product_name.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save output CSV (if None, overwrites input)
        create_backup: Whether to create backup of original file
    
    Returns:
        DataFrame with converted product_id
    """
    print("="*80)
    print("CONVERT PRODUCT IDs BY PRODUCT NAME")
    print("="*80)
    
    # Load CSV
    print(f"\nLoading file: {input_path}")
    try:
        df = pd.read_csv(input_path, encoding='utf-8')
    except UnicodeDecodeError:
        # Try with different encoding
        df = pd.read_csv(input_path, encoding='latin-1')
    
    print(f"Loaded {len(df):,} rows")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check required columns
    if 'product_name' not in df.columns:
        raise ValueError("Column 'product_name' not found in CSV")
    if 'product_id' not in df.columns:
        raise ValueError("Column 'product_id' not found in CSV")
    
    # Show original product_id stats
    print(f"\nOriginal product_id statistics:")
    print(f"  Unique product_ids: {df['product_id'].nunique():,}")
    print(f"  Range: [{df['product_id'].min()}, {df['product_id'].max()}]")
    print(f"  Sample: {df['product_id'].head(10).tolist()}")
    
    # Create backup
    if create_backup and output_path is None:
        backup_path = input_path.parent / f"{input_path.stem}_BACKUP.csv"
        df.to_csv(backup_path, index=False, encoding='utf-8')
        print(f"\nCreated backup: {backup_path.name}")
    
    # Load reference products if provided
    df_ref = None
    if reference_path:
        df_ref = load_reference_products(reference_path)
    else:
        # Try to auto-detect
        df_ref = load_reference_products()
    
    if df_ref is not None:
        # Map using reference products
        print(f"\n" + "="*80)
        print("MAPPING USING REFERENCE PRODUCTS")
        print("="*80)
        
        # Normalize reference product names
        print(f"\nNormalizing reference product names...")
        df_ref['normalized_name'] = df_ref['product_name'].apply(normalize_product_name)
        
        # Create mapping: normalized_name → reference_product_id
        ref_name_to_id = dict(zip(df_ref['normalized_name'], df_ref['product_id']))
        print(f"  Created {len(ref_name_to_id):,} reference mappings")
        
        # Normalize input product names
        print(f"\nNormalizing input product names...")
        df['normalized_name'] = df['product_name'].apply(normalize_product_name)
        
        # Map to reference product_id
        print(f"\nMapping to reference product_ids...")
        df['reference_product_id'] = df['normalized_name'].map(ref_name_to_id)
        
        # Count matches
        matched = df['reference_product_id'].notna().sum()
        unmatched = df['reference_product_id'].isna().sum()
        print(f"  Matched: {matched:,} rows ({matched/len(df)*100:.1f}%)")
        print(f"  Unmatched: {unmatched:,} rows ({unmatched/len(df)*100:.1f}%)")
        
        if unmatched > 0:
            print(f"\n  Sample unmatched product names (first 5):")
            unmatched_names = df[df['reference_product_id'].isna()]['product_name'].head(5).tolist()
            for name in unmatched_names:
                display_name = name[:60] + "..." if len(name) > 60 else name
                print(f"    - {display_name}")
        
        # Create contiguous mapping for matched products
        matched_df = df[df['reference_product_id'].notna()].copy()
        unique_ref_ids = sorted(matched_df['reference_product_id'].unique())
        ref_id_to_new_id = {ref_id: idx for idx, ref_id in enumerate(unique_ref_ids)}
        
        print(f"\nCreating contiguous ID mapping...")
        print(f"  Unique matched products: {len(unique_ref_ids):,}")
        print(f"  New ID range: [0, {len(unique_ref_ids)-1}]")
        
        # Map to new contiguous IDs
        df['new_product_id'] = df['reference_product_id'].map(ref_id_to_new_id)
        
        # For unmatched products, assign IDs starting from max+1
        if unmatched > 0:
            max_id = len(unique_ref_ids) - 1
            unmatched_mask = df['new_product_id'].isna()
            unmatched_indices = unmatched_mask.sum()
            df.loc[unmatched_mask, 'new_product_id'] = range(max_id + 1, max_id + 1 + unmatched_indices)
            print(f"  Unmatched products assigned IDs: [{max_id + 1}, {max_id + unmatched_indices}]")
        
        df['new_product_id'] = df['new_product_id'].astype(int)
        
    else:
        # Fallback: Use original method (map by name within file)
        print(f"\nNo reference products found. Using file-internal mapping...")
        name_to_new_id = create_name_to_id_mapping(df)
        
        # Map product_id based on normalized_name
        print(f"\nMapping product_ids...")
        df['new_product_id'] = df['normalized_name'].map(name_to_new_id)
    
    # Check for missing mappings
    missing = df['new_product_id'].isna().sum()
    if missing > 0:
        print(f"  WARNING: {missing} rows have missing normalized_name")
        # Fill with -1 for missing
        df['new_product_id'] = df['new_product_id'].fillna(-1).astype(int)
    else:
        df['new_product_id'] = df['new_product_id'].astype(int)
    
    # Replace old product_id with new_product_id
    old_product_id = df['product_id'].copy()
    df['product_id'] = df['new_product_id']
    
    # Drop temporary columns
    cols_to_drop = ['new_product_id', 'normalized_name']
    if 'reference_product_id' in df.columns:
        cols_to_drop.append('reference_product_id')
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    # Show new product_id stats
    print(f"\nNew product_id statistics:")
    print(f"  Unique product_ids: {df['product_id'].nunique():,}")
    print(f"  Range: [{df['product_id'].min()}, {df['product_id'].max()}]")
    print(f"  Sample: {df['product_id'].head(10).tolist()}")
    
    # Show mapping summary
    print(f"\nMapping summary:")
    print(f"  Original unique product_ids: {old_product_id.nunique():,}")
    print(f"  New unique product_ids: {df['product_id'].nunique():,}")
    print(f"  Rows processed: {len(df):,}")
    
    # Show products that got same new_id (grouped by name)
    print(f"\nProducts grouped by name (showing groups with >1 product):")
    name_groups = df.groupby('product_name')['product_id'].nunique()
    multi_id_groups = name_groups[name_groups > 1]
    if len(multi_id_groups) > 0:
        print(f"  WARNING: {len(multi_id_groups)} product names have multiple IDs")
        print(f"  Sample (first 5):")
        for name, count in multi_id_groups.head(5).items():
            display_name = name[:60] + "..." if len(name) > 60 else name
            print(f"    '{display_name}': {count} different IDs")
    else:
        print(f"  All products with same name have same new ID")
    
    # Save output
    if output_path is None:
        output_path = input_path
    
    print(f"\nSaving to: {output_path}")
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Saved {len(df):,} rows")
    
    return df

def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description="Convert product_id to contiguous indices by product_name"
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/published_data/attribute_based_embeddings/attribute_text_filtering.csv',
        help='Input CSV file path'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file path (if not specified, overwrites input)'
    )
    parser.add_argument(
        '--reference',
        type=str,
        default=None,
        help='Reference products file (enriched_products.parquet or products.csv) for mapping'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip creating backup file'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None
    reference_path = Path(args.reference) if args.reference else None
    
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)
    
    try:
        df = convert_product_ids(
            input_path=input_path,
            output_path=output_path,
            reference_path=reference_path,
            create_backup=not args.no_backup
        )
        
        print("\n" + "="*80)
        print("CONVERSION COMPLETE!")
        print("="*80)
        print(f"\nOutput file: {output_path or input_path}")
        print(f"Total rows: {len(df):,}")
        print(f"Unique product_ids: {df['product_id'].nunique():,}")
        print(f"Product ID range: [{df['product_id'].min()}, {df['product_id'].max()}]")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

