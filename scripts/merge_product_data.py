"""
Script to merge data from data_product_attribute.csv and attribute_text_filtering.csv
- Takes sequential product_id (0, 1, 2...) from data_product_attribute.csv
- Takes full data (type, skin_kind, is_5_star, num_sold_time, price) from attribute_text_filtering.csv
- Matches by product_name
"""

import pandas as pd
import argparse
from pathlib import Path


def merge_product_data(
    attr_file: str,
    filter_file: str,
    output_file: str,
    encoding: str = 'utf-8'
) -> pd.DataFrame:
    """
    Merge product data from two CSV files.
    
    Args:
        attr_file: Path to data_product_attribute.csv (has sequential product_id)
        filter_file: Path to attribute_text_filtering.csv (has full data)
        output_file: Path to save merged output
        encoding: File encoding (default: utf-8)
    
    Returns:
        Merged DataFrame
    """
    print("=" * 60)
    print("MERGING PRODUCT DATA")
    print("=" * 60)
    
    # Load data_product_attribute.csv
    print(f"\n1. Loading {attr_file}...")
    df_attr = pd.read_csv(attr_file, encoding=encoding)
    print(f"   - Rows: {len(df_attr)}")
    print(f"   - Columns: {list(df_attr.columns)}")
    print(f"   - product_id range: {df_attr['product_id'].min()} - {df_attr['product_id'].max()}")
    
    # Load attribute_text_filtering.csv
    print(f"\n2. Loading {filter_file}...")
    df_filter = pd.read_csv(filter_file, encoding=encoding)
    print(f"   - Rows: {len(df_filter)}")
    print(f"   - Columns: {list(df_filter.columns)}")
    
    # Identify extra columns in filter file
    extra_cols = ['type', 'skin_kind', 'is_5_star', 'num_sold_time', 'price']
    existing_extra_cols = [c for c in extra_cols if c in df_filter.columns]
    print(f"\n3. Extra columns to merge: {existing_extra_cols}")
    
    # Rename columns in df_filter to avoid conflicts
    # Keep only product_name and extra columns from filter file
    df_filter_subset = df_filter[['product_name'] + existing_extra_cols].copy()
    
    # Remove duplicates - keep first occurrence
    dup_count = df_filter_subset.duplicated(subset=['product_name']).sum()
    print(f"   - Duplicates in filter file (by product_name): {dup_count}")
    df_filter_subset = df_filter_subset.drop_duplicates(subset=['product_name'], keep='first')
    print(f"   - After dedup: {len(df_filter_subset)} unique products")
    
    # Merge on product_name
    print("\n4. Merging on product_name...")
    df_merged = df_attr.merge(
        df_filter_subset,
        on='product_name',
        how='left',
        indicator=True
    )
    
    # Check merge results
    matched = (df_merged['_merge'] == 'both').sum()
    left_only = (df_merged['_merge'] == 'left_only').sum()
    print(f"   - Matched: {matched} ({matched/len(df_attr)*100:.1f}%)")
    print(f"   - Not matched (left only): {left_only} ({left_only/len(df_attr)*100:.1f}%)")
    
    # Drop merge indicator
    df_merged = df_merged.drop(columns=['_merge'])
    
    # Reorder columns: put product_id first, then other columns
    cols_order = ['product_id', 'shop_id', 'product_name', 'ingredient', 'feature', 
                  'skin_type', 'capacity', 'design', 'brand', 'expiry', 'origin'] + existing_extra_cols
    cols_order = [c for c in cols_order if c in df_merged.columns]
    df_merged = df_merged[cols_order]
    
    # Fill NaN for extra columns with defaults
    default_values = {
        'type': 'no_type',
        'skin_kind': 'no_skin',
        'is_5_star': 0.0,
        'num_sold_time': 0.0,
        'price': 0.0
    }
    for col, default in default_values.items():
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].fillna(default)
    
    # Save output
    print(f"\n5. Saving to {output_file}...")
    df_merged.to_csv(output_file, index=False, encoding=encoding)
    print(f"   - Saved {len(df_merged)} rows")
    print(f"   - Final columns: {list(df_merged.columns)}")
    
    # Show sample
    print("\n6. Sample of merged data (first 3 rows):")
    print(df_merged.head(3).to_string())
    
    # Show unmatched if any
    if left_only > 0:
        print(f"\n7. WARNING: {left_only} products not matched!")
        unmatched = df_attr[~df_attr['product_name'].isin(df_filter['product_name'])]
        print("   First 5 unmatched product names:")
        for i, name in enumerate(unmatched['product_name'].head(5)):
            print(f"   {i+1}. {name[:80]}...")
    
    print("\n" + "=" * 60)
    print("MERGE COMPLETED!")
    print("=" * 60)
    
    return df_merged


def main():
    parser = argparse.ArgumentParser(description='Merge product data files')
    parser.add_argument(
        '--attr-file',
        default='data/published_data/data_product_attribute.csv',
        help='Path to data_product_attribute.csv'
    )
    parser.add_argument(
        '--filter-file',
        default='data/published_data/attribute_based_embeddings/attribute_text_filtering.csv',
        help='Path to attribute_text_filtering.csv'
    )
    parser.add_argument(
        '--output',
        default='data/published_data/attribute_based_embeddings/attribute_text_filtering_merged.csv',
        help='Output file path'
    )
    
    args = parser.parse_args()
    
    merge_product_data(
        attr_file=args.attr_file,
        filter_file=args.filter_file,
        output_file=args.output
    )


if __name__ == '__main__':
    main()
