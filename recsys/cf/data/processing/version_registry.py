"""
Step 7: Data Versioning - Version Registry Management

This module implements version registry for tracking data versions, changes,
and enabling reproducibility. It maintains a registry of all processed data
versions with metadata including hashes, timestamps, filters, and file lists.

Key Features:
- Version creation with auto-generated version IDs
- MD5 hash calculation for data version tracking (already implemented in DataSaver)
- Timestamp tracking for data staleness detection
- Filter configuration tracking for reproducibility
- File list tracking for all artifacts in each version
- Version lookup and comparison
- Stale data detection based on timestamps
- Version rollback support

Author: VieComRec Team
Created: November 2025
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import hashlib
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VersionRegistry:
    """
    Manages version registry for processed data artifacts.
    
    This class provides functionality to:
    - Create new data versions with metadata
    - Load and query existing versions
    - Compare versions to detect changes
    - Detect stale data based on timestamps
    - Track filter configurations for reproducibility
    - Maintain complete audit trail of data processing
    
    The registry is stored in a JSON file (versions.json) with structure:
    {
        "v1": {
            "hash": "abc123...",
            "timestamp": "2025-01-15T10:30:00",
            "filters": {"min_user_pos": 2, "min_item_pos": 5},
            "files": ["interactions.parquet", "mappings.json", ...],
            "stats": {...},
            "git_commit": "def456..." (optional)
        },
        "v2": {...},
        ...
    }
    """
    
    def __init__(self, registry_path: str = 'data/processed/versions.json'):
        """
        Initialize VersionRegistry.
        
        Args:
            registry_path: Path to the version registry JSON file
        """
        self.registry_path = Path(registry_path)
        self.registry_dir = self.registry_path.parent
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing registry or create new one
        self.registry = self._load_registry()
        
        logger.info(f"Initialized VersionRegistry at {registry_path}")
        logger.info(f"Current versions in registry: {len(self.registry)}")
    
    def _load_registry(self) -> Dict[str, Dict]:
        """
        Load existing registry from disk or create empty registry.
        
        Returns:
            Dictionary containing all versions
        """
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r', encoding='utf-8') as f:
                    registry = json.load(f)
                logger.info(f"Loaded existing registry with {len(registry)} versions")
                return registry
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse registry JSON: {e}")
                logger.warning("Creating new empty registry")
                return {}
            except Exception as e:
                logger.error(f"Failed to load registry: {e}")
                return {}
        else:
            logger.info("No existing registry found, creating new one")
            return {}
    
    def _save_registry(self):
        """Save registry to disk."""
        try:
            with open(self.registry_path, 'w', encoding='utf-8') as f:
                json.dump(self.registry, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved registry to {self.registry_path}")
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
            raise
    
    def _generate_version_id(self) -> str:
        """
        Generate unique version ID based on timestamp.
        
        Returns:
            Version ID string (e.g., "v1_20250115_103000")
        """
        # Get current version count
        version_count = len(self.registry) + 1
        
        # Generate timestamp-based suffix
        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        
        # Combine into version ID
        version_id = f"v{version_count}_{timestamp_str}"
        
        # Ensure uniqueness
        while version_id in self.registry:
            version_count += 1
            version_id = f"v{version_count}_{timestamp_str}"
        
        return version_id
    
    def create_version(
        self,
        data_hash: str,
        filters: Dict[str, Any],
        files: List[str],
        stats: Optional[Dict] = None,
        git_commit: Optional[str] = None,
        description: Optional[str] = None
    ) -> str:
        """
        Create a new data version entry in the registry.
        
        Args:
            data_hash: MD5 hash of the raw data files
            filters: Dictionary of filter configurations used
                    (e.g., {"min_user_pos": 2, "min_item_pos": 5})
            files: List of artifact filenames created in this version
            stats: Optional statistics summary for this version
            git_commit: Optional git commit hash for code version tracking
            description: Optional human-readable description
        
        Returns:
            Version ID string
        
        Example:
            >>> registry = VersionRegistry()
            >>> version_id = registry.create_version(
            ...     data_hash="abc123...",
            ...     filters={"min_user_pos": 2, "min_item_pos": 5},
            ...     files=["interactions.parquet", "mappings.json"],
            ...     stats={"train_size": 65000, "test_size": 26000},
            ...     description="Initial baseline version"
            ... )
            >>> print(version_id)
            'v1_20250115_103000'
        """
        # Generate version ID
        version_id = self._generate_version_id()
        
        # Create version entry
        version_entry = {
            "hash": data_hash,
            "timestamp": datetime.now().isoformat(),
            "filters": filters,
            "files": files,
            "stats": stats or {},
            "description": description or ""
        }
        
        # Add optional fields
        if git_commit:
            version_entry["git_commit"] = git_commit
        
        # Add to registry
        self.registry[version_id] = version_entry
        
        # Save to disk
        self._save_registry()
        
        logger.info(f"Created new version: {version_id}")
        logger.info(f"  Data hash: {data_hash[:16]}...")
        logger.info(f"  Filters: {filters}")
        logger.info(f"  Files: {len(files)} artifacts")
        
        return version_id
    
    def get_version(self, version_id: str) -> Optional[Dict]:
        """
        Get version entry by ID.
        
        Args:
            version_id: Version identifier
        
        Returns:
            Version entry dictionary or None if not found
        """
        return self.registry.get(version_id)
    
    def get_latest_version(self) -> Optional[Tuple[str, Dict]]:
        """
        Get the most recent version entry.
        
        Returns:
            Tuple of (version_id, version_entry) or None if registry is empty
        """
        if not self.registry:
            return None
        
        # Sort by timestamp (most recent first)
        sorted_versions = sorted(
            self.registry.items(),
            key=lambda x: x[1]['timestamp'],
            reverse=True
        )
        
        return sorted_versions[0]
    
    def list_versions(self, limit: Optional[int] = None) -> List[Tuple[str, Dict]]:
        """
        List all versions in chronological order (newest first).
        
        Args:
            limit: Optional limit on number of versions to return
        
        Returns:
            List of (version_id, version_entry) tuples
        """
        # Sort by timestamp (most recent first)
        sorted_versions = sorted(
            self.registry.items(),
            key=lambda x: x[1]['timestamp'],
            reverse=True
        )
        
        if limit:
            sorted_versions = sorted_versions[:limit]
        
        return sorted_versions
    
    def compare_versions(
        self,
        version_id1: str,
        version_id2: str
    ) -> Dict[str, Any]:
        """
        Compare two versions to detect differences.
        
        Args:
            version_id1: First version ID
            version_id2: Second version ID
        
        Returns:
            Dictionary containing comparison results:
            {
                "hash_changed": bool,
                "filters_changed": bool,
                "filter_differences": {...},
                "files_added": [...],
                "files_removed": [...],
                "stats_changed": bool
            }
        
        Example:
            >>> comparison = registry.compare_versions("v1_20250115_103000", "v2_20250116_140000")
            >>> if comparison['hash_changed']:
            ...     print("Raw data changed - models need retraining")
            >>> if comparison['filters_changed']:
            ...     print(f"Filter changes: {comparison['filter_differences']}")
        """
        v1 = self.get_version(version_id1)
        v2 = self.get_version(version_id2)
        
        if v1 is None or v2 is None:
            raise ValueError(f"Version not found: {version_id1 if v1 is None else version_id2}")
        
        comparison = {
            "hash_changed": v1['hash'] != v2['hash'],
            "filters_changed": v1['filters'] != v2['filters'],
            "filter_differences": {},
            "files_added": [],
            "files_removed": [],
            "stats_changed": v1.get('stats', {}) != v2.get('stats', {})
        }
        
        # Find filter differences
        if comparison['filters_changed']:
            for key in set(v1['filters'].keys()) | set(v2['filters'].keys()):
                val1 = v1['filters'].get(key)
                val2 = v2['filters'].get(key)
                if val1 != val2:
                    comparison['filter_differences'][key] = {
                        'old': val1,
                        'new': val2
                    }
        
        # Find file differences
        files1 = set(v1['files'])
        files2 = set(v2['files'])
        comparison['files_added'] = list(files2 - files1)
        comparison['files_removed'] = list(files1 - files2)
        
        return comparison
    
    def is_stale(
        self,
        version_id: str,
        max_age_hours: int = 24
    ) -> bool:
        """
        Check if a version is stale based on its timestamp.
        
        Args:
            version_id: Version identifier
            max_age_hours: Maximum age in hours before considering stale
        
        Returns:
            True if version is stale, False otherwise
        
        Example:
            >>> if registry.is_stale("v1_20250115_103000", max_age_hours=24):
            ...     print("Data is over 24 hours old - consider retraining")
        """
        version = self.get_version(version_id)
        if version is None:
            raise ValueError(f"Version not found: {version_id}")
        
        # Parse timestamp
        version_time = datetime.fromisoformat(version['timestamp'])
        age_hours = (datetime.now() - version_time).total_seconds() / 3600
        
        return age_hours > max_age_hours
    
    def find_version_by_hash(self, data_hash: str) -> Optional[str]:
        """
        Find version ID by data hash.
        
        Args:
            data_hash: MD5 hash to search for
        
        Returns:
            Version ID or None if not found
        """
        for version_id, version_entry in self.registry.items():
            if version_entry['hash'] == data_hash:
                return version_id
        return None
    
    def find_versions_by_filters(self, filters: Dict[str, Any]) -> List[str]:
        """
        Find all versions with matching filter configurations.
        
        Args:
            filters: Filter configuration to match
        
        Returns:
            List of version IDs with matching filters
        """
        matching_versions = []
        for version_id, version_entry in self.registry.items():
            if version_entry['filters'] == filters:
                matching_versions.append(version_id)
        return matching_versions
    
    def delete_version(self, version_id: str):
        """
        Delete a version from the registry.
        
        Args:
            version_id: Version identifier to delete
        
        Note:
            This only removes the version from the registry,
            it does not delete the actual data files.
        """
        if version_id not in self.registry:
            raise ValueError(f"Version not found: {version_id}")
        
        del self.registry[version_id]
        self._save_registry()
        
        logger.info(f"Deleted version: {version_id}")
    
    def get_registry_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics about the registry.
        
        Returns:
            Dictionary with summary information:
            {
                "num_versions": int,
                "oldest_version": str,
                "newest_version": str,
                "total_files": int,
                "unique_hashes": int
            }
        """
        if not self.registry:
            return {
                "num_versions": 0,
                "oldest_version": None,
                "newest_version": None,
                "total_files": 0,
                "unique_hashes": 0
            }
        
        # Sort by timestamp
        sorted_versions = sorted(
            self.registry.items(),
            key=lambda x: x[1]['timestamp']
        )
        
        # Count unique hashes
        unique_hashes = len(set(v['hash'] for v in self.registry.values()))
        
        # Count total files
        all_files = set()
        for version_entry in self.registry.values():
            all_files.update(version_entry['files'])
        
        return {
            "num_versions": len(self.registry),
            "oldest_version": sorted_versions[0][0],
            "newest_version": sorted_versions[-1][0],
            "total_files": len(all_files),
            "unique_hashes": unique_hashes
        }
    
    def export_to_csv(self, output_path: str):
        """
        Export registry to CSV format for analysis.
        
        Args:
            output_path: Path to output CSV file
        """
        import csv
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'version_id', 'timestamp', 'hash', 
                'num_files', 'filters', 'description'
            ])
            
            # Write rows
            for version_id, version_entry in sorted(
                self.registry.items(),
                key=lambda x: x[1]['timestamp']
            ):
                writer.writerow([
                    version_id,
                    version_entry['timestamp'],
                    version_entry['hash'],
                    len(version_entry['files']),
                    json.dumps(version_entry['filters']),
                    version_entry.get('description', '')
                ])
        
        logger.info(f"Exported registry to {output_path}")


# Standalone utility functions

def compute_file_hash(file_path: str) -> str:
    """
    Compute MD5 hash of a single file.
    
    Args:
        file_path: Path to file
    
    Returns:
        MD5 hash string
    """
    hash_md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def compute_data_hash_from_files(file_paths: List[str]) -> str:
    """
    Compute combined MD5 hash from multiple files (sorted concatenation).
    
    Args:
        file_paths: List of file paths to hash
    
    Returns:
        Combined MD5 hash string
    
    Example:
        >>> hash_val = compute_data_hash_from_files([
        ...     'data/published_data/data_reviews_purchase.csv',
        ...     'data/published_data/data_product.csv'
        ... ])
    """
    hash_md5 = hashlib.md5()
    
    # Sort file paths for consistent ordering
    sorted_paths = sorted(file_paths)
    
    for file_path in sorted_paths:
        if not os.path.exists(file_path):
            logger.warning(f"File not found for hashing: {file_path}")
            continue
        
        # Read file in chunks and update hash
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
    
    return hash_md5.hexdigest()


def get_git_commit_hash() -> Optional[str]:
    """
    Get current git commit hash if in a git repository.
    
    Returns:
        Git commit hash or None if not in git repo
    """
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


if __name__ == '__main__':
    """
    Example usage and testing.
    """
    print("="*80)
    print("Step 7: Data Versioning - VersionRegistry Example")
    print("="*80)
    
    # Initialize registry
    registry = VersionRegistry(registry_path='data/processed/versions.json')
    
    # Example 1: Create a new version
    print("\n[Example 1] Creating new version...")
    version_id = registry.create_version(
        data_hash="abc123def456",
        filters={
            "min_user_pos": 2,
            "min_item_pos": 5,
            "positive_threshold": 4,
            "hard_negative_threshold": 3
        },
        files=[
            "interactions.parquet",
            "user_item_mappings.json",
            "X_train_confidence.npz",
            "user_pos_train.pkl",
            "item_popularity.npy",
            "data_stats.json"
        ],
        stats={
            "train_size": 65000,
            "test_size": 26000,
            "num_users": 26000,
            "num_items": 2200,
            "sparsity": 0.0011
        },
        description="Initial baseline version"
    )
    print(f"Created version: {version_id}")
    
    # Example 2: Get version details
    print("\n[Example 2] Retrieving version details...")
    version = registry.get_version(version_id)
    print(f"Version: {version_id}")
    print(f"  Timestamp: {version['timestamp']}")
    print(f"  Hash: {version['hash']}")
    print(f"  Filters: {version['filters']}")
    print(f"  Files: {len(version['files'])} artifacts")
    
    # Example 3: Get registry summary
    print("\n[Example 3] Registry summary...")
    summary = registry.get_registry_summary()
    print(f"Total versions: {summary['num_versions']}")
    print(f"Unique data hashes: {summary['unique_hashes']}")
    print(f"Total unique files: {summary['total_files']}")
    
    print("\n" + "="*80)
    print("VersionRegistry module ready for integration!")
    print("="*80)
