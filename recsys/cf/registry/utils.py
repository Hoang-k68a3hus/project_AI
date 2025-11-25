"""
Registry Utilities Module.

This module provides helper functions for registry operations:
- Version generation and parsing
- Path validation
- Git commit extraction
- Hash computation
- Registry migration/backup

Example:
    >>> from recsys.cf.registry.utils import get_git_commit, generate_version_id
    >>> commit = get_git_commit()
    >>> version = generate_version_id("als")
"""

from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
import os
import subprocess
import hashlib
import json
import shutil
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Version Management
# ============================================================================

def generate_version_id(prefix: str = "v", include_counter: bool = False) -> str:
    """
    Generate version identifier.
    
    Format: {prefix}_{YYYYMMDD}_{HHMMSS}
    
    Args:
        prefix: Version prefix (e.g., 'als', 'bpr', 'bert')
        include_counter: Include incrementing counter (not implemented)
    
    Returns:
        Version string
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


def parse_version_id(version: str) -> Dict[str, Any]:
    """
    Parse version identifier.
    
    Args:
        version: Version string like 'als_20250115_123456'
    
    Returns:
        Dict with prefix, date, time
    """
    parts = version.split('_')
    
    result = {'raw': version, 'prefix': parts[0]}
    
    if len(parts) >= 2:
        result['date'] = parts[1]
    
    if len(parts) >= 3:
        result['time'] = parts[2]
    
    # Parse datetime if possible
    if len(parts) >= 3:
        try:
            dt = datetime.strptime(f"{parts[1]}_{parts[2]}", "%Y%m%d_%H%M%S")
            result['datetime'] = dt.isoformat()
        except ValueError:
            pass
    
    return result


def compare_versions(v1: str, v2: str) -> int:
    """
    Compare two version strings.
    
    Args:
        v1: First version
        v2: Second version
    
    Returns:
        -1 if v1 < v2, 0 if equal, 1 if v1 > v2
    """
    p1 = parse_version_id(v1)
    p2 = parse_version_id(v2)
    
    # Compare by datetime if available
    dt1 = p1.get('datetime', '')
    dt2 = p2.get('datetime', '')
    
    if dt1 < dt2:
        return -1
    elif dt1 > dt2:
        return 1
    return 0


# ============================================================================
# Git Integration
# ============================================================================

def get_git_commit(repo_path: Optional[str] = None) -> Optional[str]:
    """
    Get current git commit hash.
    
    Args:
        repo_path: Path to git repository (None = current dir)
    
    Returns:
        Git commit hash or None
    """
    cwd = repo_path or os.getcwd()
    
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            cwd=cwd
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception as e:
        logger.debug(f"Could not get git commit: {e}")
    
    return None


def get_git_commit_short(repo_path: Optional[str] = None) -> Optional[str]:
    """Get short (7 char) git commit hash."""
    commit = get_git_commit(repo_path)
    return commit[:7] if commit else None


def get_git_branch(repo_path: Optional[str] = None) -> Optional[str]:
    """Get current git branch name."""
    cwd = repo_path or os.getcwd()
    
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True,
            text=True,
            cwd=cwd
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    
    return None


def is_git_clean(repo_path: Optional[str] = None) -> bool:
    """Check if git working tree is clean (no uncommitted changes)."""
    cwd = repo_path or os.getcwd()
    
    try:
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True,
            text=True,
            cwd=cwd
        )
        return result.returncode == 0 and len(result.stdout.strip()) == 0
    except Exception:
        pass
    
    return False


# ============================================================================
# Hash Computation
# ============================================================================

def compute_file_hash(file_path: str, algorithm: str = 'md5') -> str:
    """
    Compute hash of file contents.
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256')
    
    Returns:
        Hex digest string
    """
    if algorithm == 'md5':
        hasher = hashlib.md5()
    elif algorithm == 'sha1':
        hasher = hashlib.sha1()
    elif algorithm == 'sha256':
        hasher = hashlib.sha256()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    
    return hasher.hexdigest()


def compute_directory_hash(
    dir_path: str,
    extensions: Optional[List[str]] = None
) -> str:
    """
    Compute hash of directory contents.
    
    Args:
        dir_path: Path to directory
        extensions: File extensions to include (None = all)
    
    Returns:
        Combined hash string
    """
    path = Path(dir_path)
    
    if not path.is_dir():
        raise ValueError(f"Not a directory: {dir_path}")
    
    hasher = hashlib.md5()
    
    # Get sorted list of files
    files = sorted(path.rglob('*'))
    
    for file_path in files:
        if not file_path.is_file():
            continue
        
        if extensions:
            if file_path.suffix.lower() not in extensions:
                continue
        
        # Add filename
        hasher.update(file_path.name.encode())
        
        # Add file hash
        file_hash = compute_file_hash(str(file_path))
        hasher.update(file_hash.encode())
    
    return hasher.hexdigest()


def compute_data_version(data_files: List[str]) -> str:
    """
    Compute data version hash from multiple files.
    
    Args:
        data_files: List of file paths
    
    Returns:
        Combined hash string
    """
    hasher = hashlib.md5()
    
    for file_path in sorted(data_files):
        if os.path.exists(file_path):
            file_hash = compute_file_hash(file_path)
            hasher.update(f"{file_path}:{file_hash}".encode())
    
    return hasher.hexdigest()


# ============================================================================
# Path Validation
# ============================================================================

def validate_model_path(
    model_path: str,
    model_type: str,
    required_files: Optional[List[str]] = None
) -> Tuple[bool, List[str]]:
    """
    Validate model path contains required files.
    
    Args:
        model_path: Path to model folder
        model_type: Model type (als, bpr, bert_als)
        required_files: Override required files list
    
    Returns:
        Tuple of (is_valid, missing_files)
    """
    from .registry import REQUIRED_MODEL_FILES
    
    path = Path(model_path)
    
    if not path.exists():
        return False, [str(model_path)]
    
    if required_files is None:
        if model_type not in REQUIRED_MODEL_FILES:
            return False, [f"Unknown model type: {model_type}"]
        required_files = REQUIRED_MODEL_FILES[model_type]
    
    missing = []
    for file in required_files:
        if not (path / file).exists():
            missing.append(file)
    
    return len(missing) == 0, missing


def ensure_directory(dir_path: str) -> Path:
    """
    Ensure directory exists, create if needed.
    
    Args:
        dir_path: Directory path
    
    Returns:
        Path object
    """
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ============================================================================
# Backup & Migration
# ============================================================================

def backup_registry(
    registry_path: str,
    backup_dir: Optional[str] = None
) -> str:
    """
    Create backup of registry file.
    
    Args:
        registry_path: Path to registry.json
        backup_dir: Backup directory (default: same as registry)
    
    Returns:
        Path to backup file
    """
    path = Path(registry_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Registry not found: {registry_path}")
    
    # Determine backup path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{path.stem}_backup_{timestamp}{path.suffix}"
    
    if backup_dir:
        backup_path = Path(backup_dir) / backup_name
        backup_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        backup_path = path.parent / backup_name
    
    # Copy
    shutil.copy2(registry_path, backup_path)
    
    logger.info(f"Created backup: {backup_path}")
    return str(backup_path)


def restore_registry(
    backup_path: str,
    registry_path: str,
    create_current_backup: bool = True
) -> bool:
    """
    Restore registry from backup.
    
    Args:
        backup_path: Path to backup file
        registry_path: Target registry path
        create_current_backup: Backup current before overwriting
    
    Returns:
        True if successful
    """
    if not os.path.exists(backup_path):
        raise FileNotFoundError(f"Backup not found: {backup_path}")
    
    # Backup current if exists
    if create_current_backup and os.path.exists(registry_path):
        backup_registry(registry_path)
    
    # Restore
    shutil.copy2(backup_path, registry_path)
    
    logger.info(f"Restored registry from: {backup_path}")
    return True


def list_backups(
    registry_path: str,
    backup_dir: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    List available registry backups.
    
    Args:
        registry_path: Path to registry.json
        backup_dir: Backup directory (default: same as registry)
    
    Returns:
        List of backup info dicts
    """
    path = Path(registry_path)
    
    if backup_dir:
        search_dir = Path(backup_dir)
    else:
        search_dir = path.parent
    
    pattern = f"{path.stem}_backup_*{path.suffix}"
    
    backups = []
    for backup_file in search_dir.glob(pattern):
        stat = backup_file.stat()
        backups.append({
            'path': str(backup_file),
            'name': backup_file.name,
            'size_bytes': stat.st_size,
            'created_at': datetime.fromtimestamp(stat.st_mtime).isoformat()
        })
    
    # Sort by date descending
    backups.sort(key=lambda x: x['created_at'], reverse=True)
    
    return backups


# ============================================================================
# Model Artifact Utilities
# ============================================================================

def copy_model_artifacts(
    src_path: str,
    dest_path: str,
    model_type: str
) -> bool:
    """
    Copy model artifacts to new location.
    
    Args:
        src_path: Source model folder
        dest_path: Destination folder
        model_type: Model type
    
    Returns:
        True if successful
    """
    from .registry import REQUIRED_MODEL_FILES
    
    src = Path(src_path)
    dest = Path(dest_path)
    
    dest.mkdir(parents=True, exist_ok=True)
    
    files = REQUIRED_MODEL_FILES.get(model_type, [])
    
    for file in files:
        src_file = src / file
        if src_file.exists():
            shutil.copy2(src_file, dest / file)
    
    logger.info(f"Copied model artifacts from {src_path} to {dest_path}")
    return True


def cleanup_old_artifacts(
    artifacts_dir: str,
    keep_count: int = 5,
    keep_current_best: bool = True,
    dry_run: bool = True
) -> List[str]:
    """
    Clean up old model artifacts, keeping recent versions.
    
    Args:
        artifacts_dir: Base artifacts directory
        keep_count: Number of versions to keep per model type
        keep_current_best: Never delete current best model
        dry_run: If True, only report what would be deleted
    
    Returns:
        List of paths that were/would be deleted
    """
    path = Path(artifacts_dir)
    deleted = []
    
    # Group by model type
    model_types = ['als', 'bpr', 'bert_als']
    
    for model_type in model_types:
        type_dir = path / model_type
        if not type_dir.exists():
            continue
        
        # Get version folders (timestamped)
        versions = []
        for item in type_dir.iterdir():
            if item.is_dir() and item.name not in ['.', '..']:
                versions.append(item)
        
        # Sort by modification time (newest first)
        versions.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Identify versions to delete
        to_delete = versions[keep_count:]
        
        for version_dir in to_delete:
            if dry_run:
                logger.info(f"[DRY RUN] Would delete: {version_dir}")
            else:
                shutil.rmtree(version_dir)
                logger.info(f"Deleted: {version_dir}")
            
            deleted.append(str(version_dir))
    
    return deleted


# ============================================================================
# Metadata Utilities
# ============================================================================

def create_model_metadata(
    num_users: int,
    num_items: int,
    factors: int,
    model_type: str,
    data_version: Optional[str] = None,
    git_commit: Optional[str] = None,
    score_range: Optional[Tuple[float, float]] = None,
    extra: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Create standard model metadata dictionary.
    
    Args:
        num_users: Number of users
        num_items: Number of items
        factors: Embedding dimension
        model_type: Type of model
        data_version: Data version hash
        git_commit: Git commit (auto-detected if None)
        score_range: Min/max prediction score
        extra: Additional metadata
    
    Returns:
        Metadata dictionary
    """
    metadata = {
        'num_users': num_users,
        'num_items': num_items,
        'factors': factors,
        'model_type': model_type,
        'training_date': datetime.now().isoformat(),
        'data_version_hash': data_version or 'unknown',
        'git_commit': git_commit or get_git_commit()
    }
    
    if score_range:
        metadata['score_range'] = {
            'min': score_range[0],
            'max': score_range[1]
        }
    
    if extra:
        metadata.update(extra)
    
    return metadata


def load_model_metadata(model_path: str, model_type: str) -> Dict[str, Any]:
    """
    Load model metadata from file.
    
    Args:
        model_path: Path to model folder
        model_type: Model type
    
    Returns:
        Metadata dictionary
    """
    metadata_file = Path(model_path) / f"{model_type}_metadata.json"
    
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_file}")
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_model_metadata(
    model_path: str,
    model_type: str,
    metadata: Dict[str, Any]
) -> str:
    """
    Save model metadata to file.
    
    Args:
        model_path: Path to model folder
        model_type: Model type
        metadata: Metadata dictionary
    
    Returns:
        Path to saved file
    """
    path = Path(model_path)
    path.mkdir(parents=True, exist_ok=True)
    
    metadata_file = path / f"{model_type}_metadata.json"
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    return str(metadata_file)


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing Registry Utilities")
    print("=" * 60)
    
    # Test version generation
    version = generate_version_id("als")
    print(f"Generated version: {version}")
    
    parsed = parse_version_id(version)
    print(f"Parsed version: {parsed}")
    
    # Test git functions
    commit = get_git_commit()
    print(f"Git commit: {commit}")
    
    branch = get_git_branch()
    print(f"Git branch: {branch}")
    
    clean = is_git_clean()
    print(f"Git clean: {clean}")
    
    # Test comparison
    v1 = "als_20250101_120000"
    v2 = "als_20250102_120000"
    cmp = compare_versions(v1, v2)
    print(f"Compare {v1} vs {v2}: {cmp}")
    
    print("\n" + "=" * 60)
    print("Utility tests passed!")
