"""
Data Reader Module for Collaborative Filtering

This module handles loading raw CSV data files with proper encoding
and schema validation.
"""

import os
import logging
from typing import Dict, List

import pandas as pd


logger = logging.getLogger("data_layer")


class DataReader:
    """
    Class for reading and loading raw data files.
    
    This class handles:
    - Loading CSV files with UTF-8 encoding
    - Schema validation
    - File existence checks
    """
    
    def __init__(self, base_path: str = "data/published_data"):
        """
        Initialize DataReader.
        
        Args:
            base_path: Base directory containing raw CSV files
        """
        self.base_path = base_path
        self.file_paths = {
            'interactions': os.path.join(base_path, 'data_reviews_purchase.csv'),
            'products': os.path.join(base_path, 'data_product.csv'),
            'attributes': os.path.join(base_path, 'data_product_attribute.csv'),
            'shops': os.path.join(base_path, 'data_shop.csv')
        }
        
        # Expected columns for schema validation
        self.expected_columns = {
            'interactions': ['user_id', 'product_id', 'rating', 'cmt_date'],
            'products': ['product_id', 'product_name'],
            'attributes': ['product_id'],
            'shops': []  # Optional validation
        }
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all raw CSV files from published_data directory.
        
        Returns:
            Dictionary with keys: 'interactions', 'products', 'attributes', 'shops'
        
        Raises:
            FileNotFoundError: If required CSV files are missing
            ValueError: If CSV schema is invalid
        """
        logger.info("="*80)
        logger.info("STARTING DATA LOADING PROCESS")
        logger.info("="*80)
        
        # Check file existence
        self._check_files_exist()
        
        # Load all DataFrames
        dataframes = self._load_dataframes()
        
        # Validate schema
        self._validate_schema(dataframes)
        
        logger.info("\n" + "="*80)
        logger.info("DATA LOADING COMPLETED SUCCESSFULLY")
        logger.info("="*80 + "\n")
        
        return dataframes
    
    def load_interactions(self) -> pd.DataFrame:
        """
        Load only interactions data.
        
        Returns:
            Interactions DataFrame
        """
        logger.info("Loading interactions data...")
        df = pd.read_csv(self.file_paths['interactions'], encoding='utf-8')
        logger.info(f"✓ Loaded {len(df)} interaction records")
        return df
    
    def load_products(self) -> pd.DataFrame:
        """
        Load only products data.
        
        Returns:
            Products DataFrame
        """
        logger.info("Loading products data...")
        df = pd.read_csv(self.file_paths['products'], encoding='utf-8')
        logger.info(f"✓ Loaded {len(df)} product records")
        return df
    
    def load_attributes(self) -> pd.DataFrame:
        """
        Load only attributes data.
        
        Returns:
            Attributes DataFrame
        """
        logger.info("Loading attributes data...")
        df = pd.read_csv(self.file_paths['attributes'], encoding='utf-8')
        logger.info(f"✓ Loaded {len(df)} attribute records")
        return df
    
    def load_shops(self) -> pd.DataFrame:
        """
        Load only shops data.
        
        Returns:
            Shops DataFrame
        """
        logger.info("Loading shops data...")
        df = pd.read_csv(self.file_paths['shops'], encoding='utf-8')
        logger.info(f"✓ Loaded {len(df)} shop records")
        return df
    
    def _check_files_exist(self) -> None:
        """
        Check if all required files exist.
        
        Raises:
            FileNotFoundError: If any required file is missing
        """
        for name, path in self.file_paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required file not found: {path}")
            logger.info(f"Found {name} file: {path}")
    
    def _load_dataframes(self) -> Dict[str, pd.DataFrame]:
        """
        Load all DataFrames with UTF-8 encoding.
        
        Returns:
            Dictionary of loaded DataFrames
        
        Raises:
            Exception: If loading fails
        """
        dataframes = {}
        
        try:
            logger.info("\n" + "-"*80)
            
            logger.info("Loading interactions data...")
            dataframes['interactions'] = pd.read_csv(
                self.file_paths['interactions'], 
                encoding='utf-8'
            )
            logger.info(f"✓ Loaded {len(dataframes['interactions'])} interaction records")
            
            logger.info("\nLoading products data...")
            dataframes['products'] = pd.read_csv(
                self.file_paths['products'], 
                encoding='utf-8'
            )
            logger.info(f"✓ Loaded {len(dataframes['products'])} product records")
            
            logger.info("\nLoading attributes data...")
            dataframes['attributes'] = pd.read_csv(
                self.file_paths['attributes'], 
                encoding='utf-8'
            )
            logger.info(f"✓ Loaded {len(dataframes['attributes'])} attribute records")
            
            logger.info("\nLoading shops data...")
            dataframes['shops'] = pd.read_csv(
                self.file_paths['shops'], 
                encoding='utf-8'
            )
            logger.info(f"✓ Loaded {len(dataframes['shops'])} shop records")
            
        except Exception as e:
            logger.error(f"Failed to load CSV files: {str(e)}")
            raise
        
        return dataframes
    
    def _validate_schema(self, dataframes: Dict[str, pd.DataFrame]) -> None:
        """
        Validate that loaded DataFrames have expected columns.
        
        Args:
            dataframes: Dict of loaded DataFrames
        
        Raises:
            ValueError: If required columns are missing
        """
        logger.info("\n" + "-"*80)
        logger.info("VALIDATING DATA SCHEMA")
        logger.info("-"*80)
        
        for name, expected_cols in self.expected_columns.items():
            if not expected_cols:
                continue
                
            df = dataframes[name]
            missing_cols = set(expected_cols) - set(df.columns)
            
            if missing_cols:
                raise ValueError(
                    f"Missing required columns in {name}: {missing_cols}\n"
                    f"Available columns: {list(df.columns)}"
                )
            
            logger.info(f"✓ {name}: All required columns present")
            logger.info(f"  Columns: {list(df.columns)[:10]}{'...' if len(df.columns) > 10 else ''}")
