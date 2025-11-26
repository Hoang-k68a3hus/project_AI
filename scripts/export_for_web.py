"""
Export files needed for web project API.

This script copies all necessary files for the recommendation API service
to a separate directory, excluding data processing and training code.

Usage:
    python scripts/export_for_web.py --output ../web_project_IAI
    
Or to create a list of files:
    python scripts/export_for_web.py --list-only
"""

import os
import shutil
import argparse
from pathlib import Path


# Files and directories to export for web API
EXPORT_STRUCTURE = {
    # ============================================================
    # SERVICE LAYER (Core API)
    # ============================================================
    "service": {
        "include": [
            "service/__init__.py",
            "service/api.py",
            "service/dashboard.py",
        ],
        "subdirs": {
            "service/recommender": [
                "__init__.py",
                "cache.py",
                "fallback.py",
                "filters.py",
                "loader.py",
                "phobert_loader.py",
                "recommender.py",
                "rerank.py",
            ],
            "service/search": [
                "__init__.py",
                "query_encoder.py",
                "search_index.py",
                "smart_search.py",
            ],
            "service/config": [
                "rerank_config.yaml",
                "search_config.yaml",
                "serving_config.yaml",
            ],
        }
    },
    
    # ============================================================
    # RECSYS MODULE (Only serving-related parts)
    # ============================================================
    "recsys": {
        "include": [
            "recsys/__init__.py",
        ],
        "subdirs": {
            "recsys/cf": [
                "__init__.py",
                "logging_utils.py",  # For ServiceMetricsDB
            ],
        }
    },
    
    # ============================================================
    # CONFIGURATION
    # ============================================================
    "config": {
        "include": [
            "config/serving_config.yaml",
            "config/alerts_config.yaml",
        ]
    },
    
    # ============================================================
    # ARTIFACTS (Pre-trained models)
    # ============================================================
    "artifacts": {
        "include": [
            "artifacts/cf/registry.json",
        ],
        "subdirs": {
            "artifacts/cf/als": [
                "als_U.npy",
                "als_V.npy",
                "als_params.json",
                "als_metadata.json",
                "als_metrics.json",
            ],
            "artifacts/cf/bert_als": "ALL",  # Copy all files
            "artifacts/cf/bpr": "ALL",  # Copy all files if exists
        }
    },
    
    # ============================================================
    # DATA (Only processed data needed for serving)
    # ============================================================
    "data": {
        "subdirs": {
            "data/processed": [
                "user_item_mappings.json",
                "trainable_user_mapping.json",
                "user_metadata.pkl",
                "user_pos_train.pkl",
                "item_popularity.npy",
                "top_k_popular_items.json",
                "data_stats.json",
                "interactions.parquet",  # For user history lookup
            ],
            "data/processed/content_based_embeddings": "ALL",  # PhoBERT embeddings
            "data/published_data": [
                "data_product.csv",  # Product metadata for display
                "data_product_attribute.csv",  # For filtering
            ],
        }
    },
}

# Root-level files
ROOT_FILES = [
    ".gitignore",
    "README.md",
]


def get_files_to_export(project_root: Path) -> list:
    """Get list of all files to export."""
    files = []
    
    # Root files
    for f in ROOT_FILES:
        path = project_root / f
        if path.exists():
            files.append(str(path.relative_to(project_root)))
    
    # Process export structure
    for category, config in EXPORT_STRUCTURE.items():
        # Direct includes
        if "include" in config:
            for f in config["include"]:
                path = project_root / f
                if path.exists():
                    files.append(f)
        
        # Subdirectories
        if "subdirs" in config:
            for subdir, items in config["subdirs"].items():
                subdir_path = project_root / subdir
                
                if items == "ALL":
                    # Copy all files in directory
                    if subdir_path.exists():
                        for f in subdir_path.iterdir():
                            if f.is_file() and not f.name.startswith('.') and f.name != 'desktop.ini':
                                files.append(str(f.relative_to(project_root)))
                else:
                    # Copy specific files
                    for item in items:
                        path = subdir_path / item
                        if path.exists():
                            files.append(str(path.relative_to(project_root)))
    
    return sorted(set(files))


def export_files(project_root: Path, output_dir: Path, dry_run: bool = False):
    """Export files to output directory."""
    files = get_files_to_export(project_root)
    
    print(f"\n{'=' * 60}")
    print(f"EXPORT FOR WEB PROJECT")
    print(f"{'=' * 60}")
    print(f"Source: {project_root}")
    print(f"Target: {output_dir}")
    print(f"Files: {len(files)}")
    print(f"{'=' * 60}\n")
    
    if dry_run:
        print("DRY RUN - No files will be copied\n")
    
    # Create output directory
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy files
    copied = 0
    skipped = 0
    errors = []
    
    for rel_path in files:
        src = project_root / rel_path
        dst = output_dir / rel_path
        
        if not src.exists():
            print(f"  ⚠️  SKIP (not found): {rel_path}")
            skipped += 1
            continue
        
        if dry_run:
            print(f"  📄 {rel_path}")
            copied += 1
        else:
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                print(f"  ✓ {rel_path}")
                copied += 1
            except Exception as e:
                print(f"  ✗ {rel_path}: {e}")
                errors.append((rel_path, str(e)))
    
    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Copied: {copied}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors: {len(errors)}")
    
    if errors:
        print("\nErrors:")
        for path, err in errors:
            print(f"  - {path}: {err}")
    
    return copied, skipped, errors


def create_requirements(output_dir: Path):
    """Create requirements.txt for web project."""
    requirements = """# Web API Requirements
# Core
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
pydantic>=2.0.0

# Data
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
pyarrow>=12.0.0

# ML/Embeddings
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.0

# Caching
cachetools>=5.3.0

# Optional: FAISS for fast similarity search
# faiss-cpu>=1.7.4

# Monitoring (optional)
prometheus-client>=0.17.0
"""
    
    req_path = output_dir / "requirements.txt"
    req_path.write_text(requirements)
    print(f"✓ Created: requirements.txt")


def create_dockerfile(output_dir: Path):
    """Create Dockerfile for web project."""
    dockerfile = """# Vietnamese Cosmetics Recommendation API
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run the API
CMD ["uvicorn", "service.api:app", "--host", "0.0.0.0", "--port", "8000"]
"""
    
    docker_path = output_dir / "Dockerfile"
    docker_path.write_text(dockerfile)
    print(f"✓ Created: Dockerfile")


def create_docker_compose(output_dir: Path):
    """Create docker-compose.yml for web project."""
    compose = """version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data:ro
      - ./artifacts:/app/artifacts:ro
      - ./logs:/app/logs
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  # Optional: Nginx reverse proxy
  # nginx:
  #   image: nginx:alpine
  #   ports:
  #     - "80:80"
  #   volumes:
  #     - ./nginx.conf:/etc/nginx/nginx.conf:ro
  #   depends_on:
  #     - api
"""
    
    compose_path = output_dir / "docker-compose.yml"
    compose_path.write_text(compose)
    print(f"✓ Created: docker-compose.yml")


def create_readme(output_dir: Path):
    """Create README for web project."""
    readme = """# Vietnamese Cosmetics Recommendation API

API service cho hệ thống gợi ý mỹ phẩm Việt Nam.

## Features

- **CF Recommendation**: ALS/BPR-based collaborative filtering
- **Smart Search**: Semantic search với PhoBERT embeddings  
- **Similar Items**: Tìm sản phẩm tương tự
- **User Profile Search**: Gợi ý dựa trên lịch sử

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run API
uvicorn service.api:app --host 0.0.0.0 --port 8000 --reload
```

### Docker

```bash
# Build and run
docker-compose up -d

# Check logs
docker-compose logs -f
```

## API Endpoints

### Recommendation
- `POST /recommend` - Get recommendations for a user
- `POST /batch_recommend` - Batch recommendations
- `POST /similar_items` - Find similar products

### Smart Search
- `POST /search` - Semantic search with Vietnamese query
- `POST /search/similar` - Find similar products by ID
- `POST /search/profile` - Search based on user history
- `GET /search/filters` - Available filter options

### System
- `GET /health` - Health check
- `GET /stats` - Service statistics
- `POST /reload_model` - Hot-reload model

## Example Usage

```python
import requests

# Recommend for user
response = requests.post("http://localhost:8000/recommend", json={
    "user_id": 12345,
    "topk": 10,
    "exclude_seen": True
})
print(response.json())

# Search products
response = requests.post("http://localhost:8000/search", json={
    "query": "kem dưỡng ẩm cho da khô",
    "topk": 10
})
print(response.json())
```

## Project Structure

```
.
├── service/                 # API service
│   ├── api.py              # FastAPI endpoints
│   ├── recommender/        # CF recommendation logic
│   └── search/             # Smart search module
├── data/
│   ├── processed/          # Processed data for serving
│   └── published_data/     # Product metadata
├── artifacts/
│   └── cf/                 # Trained models
├── config/                 # Configuration files
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Configuration

Edit `service/config/serving_config.yaml` for:
- Model selection
- Cache settings
- Reranking weights

## License

Private - Internal use only.
"""
    
    readme_path = output_dir / "README.md"
    readme_path.write_text(readme, encoding='utf-8')
    print(f"✓ Created: README.md")


def create_gitignore(output_dir: Path):
    """Create .gitignore for web project."""
    gitignore = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
.env

# IDE
.vscode/
.idea/
*.swp
*.swo

# Logs
logs/
*.log

# Data (large files - use Git LFS or exclude)
# Uncomment if not using Git LFS:
# *.npy
# *.pkl
# *.parquet
# *.pt

# Desktop.ini (Windows)
desktop.ini
Thumbs.db

# Docker
.docker/

# Cache
.cache/
*.cache
"""
    
    gitignore_path = output_dir / ".gitignore"
    gitignore_path.write_text(gitignore)
    print(f"✓ Created: .gitignore")


def main():
    parser = argparse.ArgumentParser(description="Export files for web project")
    parser.add_argument("--output", "-o", type=str, default="../web_project_IAI",
                        help="Output directory")
    parser.add_argument("--list-only", "-l", action="store_true",
                        help="Only list files, don't copy")
    parser.add_argument("--no-extras", action="store_true",
                        help="Don't create Dockerfile, requirements, etc.")
    
    args = parser.parse_args()
    
    # Get paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    output_dir = Path(args.output)
    
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    
    # Export files
    copied, skipped, errors = export_files(
        project_root, 
        output_dir, 
        dry_run=args.list_only
    )
    
    # Create extra files
    if not args.list_only and not args.no_extras:
        print(f"\nCreating additional files...")
        create_requirements(output_dir)
        create_dockerfile(output_dir)
        create_docker_compose(output_dir)
        create_readme(output_dir)
        create_gitignore(output_dir)
    
    print(f"\n✓ Export complete!")
    
    if not args.list_only:
        print(f"\nNext steps:")
        print(f"  cd {output_dir}")
        print(f"  git init")
        print(f"  git remote add origin https://github.com/Hoang-k68a3hus/web_project_IAI.git")
        print(f"  git add .")
        print(f"  git commit -m 'Initial commit: API service'")
        print(f"  git push -u origin main")


if __name__ == "__main__":
    main()
