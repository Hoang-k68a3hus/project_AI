"""
Test Script for Model Registry Module.

This script tests all components of the registry module:
- ModelRegistry: registration, selection, listing
- ModelLoader: loading, caching, hot-reload
- BERTEmbeddingsRegistry: BERT embeddings management
- Utility functions: versioning, git, hashing

Run:
    python scripts/test_registry_module.py
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_imports():
    """Test all imports from registry module."""
    print("\n=== Test 1: Imports ===")
    
    from recsys.cf.registry import (
        # Registry
        ModelRegistry,
        create_empty_registry,
        validate_registry_schema,
        get_registry,
        register_model,
        select_best_model,
        DEFAULT_REGISTRY_PATH,
        REQUIRED_MODEL_FILES,
        MODEL_STATUS,
        
        # Model Loader
        ModelLoader,
        ModelState,
        LoaderStats,
        get_loader,
        reset_loader,
        load_model_from_registry,
        
        # BERT Registry
        BERTEmbeddingsRegistry,
        get_bert_registry,
        DEFAULT_BERT_REGISTRY_PATH,
        
        # Utilities
        generate_version_id,
        parse_version_id,
        compare_versions,
        get_git_commit,
        get_git_commit_short,
        get_git_branch,
        compute_file_hash,
        validate_model_path,
        ensure_directory,
        backup_registry,
        list_backups,
        create_model_metadata,
    )
    
    print("✓ All imports successful")
    print(f"  - DEFAULT_REGISTRY_PATH: {DEFAULT_REGISTRY_PATH}")
    print(f"  - REQUIRED_MODEL_FILES keys: {list(REQUIRED_MODEL_FILES.keys())}")
    print(f"  - MODEL_STATUS: {MODEL_STATUS}")
    
    return True


def test_version_utilities():
    """Test version generation and parsing."""
    print("\n=== Test 2: Version Utilities ===")
    
    from recsys.cf.registry import (
        generate_version_id,
        parse_version_id,
        compare_versions,
    )
    
    # Generate version
    version = generate_version_id("als")
    print(f"  Generated version: {version}")
    assert version.startswith("als_"), f"Expected prefix 'als_', got {version}"
    
    # Parse version
    parsed = parse_version_id(version)
    print(f"  Parsed: {parsed}")
    assert parsed['prefix'] == 'als', f"Expected prefix 'als', got {parsed['prefix']}"
    assert 'date' in parsed, "Expected 'date' in parsed"
    assert 'datetime' in parsed, "Expected 'datetime' in parsed"
    
    # Compare versions
    v1 = "als_20250101_120000"
    v2 = "als_20250102_120000"
    v3 = "als_20250101_120000"
    
    cmp1 = compare_versions(v1, v2)
    cmp2 = compare_versions(v2, v1)
    cmp3 = compare_versions(v1, v3)
    
    assert cmp1 == -1, f"Expected v1 < v2, got {cmp1}"
    assert cmp2 == 1, f"Expected v2 > v1, got {cmp2}"
    assert cmp3 == 0, f"Expected v1 == v3, got {cmp3}"
    
    print("✓ Version utilities passed")
    return True


def test_git_utilities():
    """Test git integration utilities."""
    print("\n=== Test 3: Git Utilities ===")
    
    from recsys.cf.registry import (
        get_git_commit,
        get_git_commit_short,
        get_git_branch,
        is_git_clean,
    )
    
    # Get commit
    commit = get_git_commit()
    print(f"  Git commit: {commit}")
    
    if commit:
        short = get_git_commit_short()
        print(f"  Short commit: {short}")
        assert len(short) == 7, f"Expected 7 char short commit, got {len(short)}"
    
    # Get branch
    branch = get_git_branch()
    print(f"  Git branch: {branch}")
    
    # Check clean
    clean = is_git_clean()
    print(f"  Git clean: {clean}")
    
    print("✓ Git utilities passed")
    return True


def test_model_registry():
    """Test ModelRegistry class."""
    print("\n=== Test 4: ModelRegistry ===")
    
    from recsys.cf.registry import (
        ModelRegistry,
        create_empty_registry,
        validate_registry_schema,
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        registry_path = os.path.join(tmpdir, 'registry.json')
        
        # Create registry
        registry = ModelRegistry(registry_path, auto_create=True)
        print(f"  Created registry at: {registry_path}")
        
        # Verify file exists
        assert os.path.exists(registry_path), "Registry file not created"
        
        # Check empty registry
        stats = registry.get_registry_stats()
        print(f"  Stats: {stats}")
        assert stats['total_models'] == 0, "Expected 0 models"
        assert stats['active_models'] == 0, "Expected 0 active models"
        
        # Current best should be None
        current = registry.get_current_best()
        assert current is None, "Expected no current best"
        
        # List models (should be empty)
        models_df = registry.list_models()
        print(f"  Models DataFrame shape: {models_df.shape}")
        assert models_df.shape[0] == 0, "Expected empty DataFrame"
        
        # Test schema validation
        empty_reg = create_empty_registry()
        assert validate_registry_schema(empty_reg), "Empty registry should be valid"
        
        invalid_reg = {'foo': 'bar'}
        assert not validate_registry_schema(invalid_reg), "Invalid registry should fail"
        
    print("✓ ModelRegistry tests passed")
    return True


def test_model_registration_with_mock():
    """Test model registration with mock artifacts."""
    print("\n=== Test 5: Model Registration with Mock Artifacts ===")
    
    from recsys.cf.registry import (
        ModelRegistry,
        REQUIRED_MODEL_FILES,
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock model artifacts
        model_path = os.path.join(tmpdir, 'als_model')
        os.makedirs(model_path)
        
        # Create required files
        for file in REQUIRED_MODEL_FILES['als']:
            file_path = os.path.join(model_path, file)
            if file.endswith('.npy'):
                np.save(file_path.replace('.npy', ''), np.random.randn(10, 64))
            else:
                with open(file_path, 'w') as f:
                    json.dump({'test': 'data'}, f)
        
        print(f"  Created mock artifacts at: {model_path}")
        
        # Create registry
        registry_path = os.path.join(tmpdir, 'registry.json')
        registry = ModelRegistry(registry_path, auto_create=True)
        
        # Validate artifacts
        is_valid, missing = registry.validate_artifacts(model_path, 'als')
        print(f"  Artifacts valid: {is_valid}, missing: {missing}")
        assert is_valid, f"Expected valid artifacts, missing: {missing}"
        
        # Register model
        model_id = registry.register_model(
            artifacts_path=model_path,
            model_type='als',
            hyperparameters={'factors': 64, 'regularization': 0.1, 'alpha': 10},
            metrics={'ndcg@10': 0.25, 'recall@10': 0.30, 'coverage': 0.85},
            training_info={'training_time_seconds': 120, 'num_iterations': 15}
        )
        
        print(f"  Registered model: {model_id}")
        assert model_id.startswith('als_'), f"Expected 'als_' prefix, got {model_id}"
        
        # Check stats
        stats = registry.get_registry_stats()
        print(f"  Stats after registration: {stats}")
        assert stats['total_models'] == 1, "Expected 1 model"
        assert stats['active_models'] == 1, "Expected 1 active model"
        
        # Get model
        model = registry.get_model(model_id)
        assert model is not None, "Model should exist"
        assert model['model_type'] == 'als', "Model type should be 'als'"
        assert model['metrics']['ndcg@10'] == 0.25, "Metrics should match"
        
        # Select best
        best = registry.select_best_model(metric='ndcg@10')
        print(f"  Selected best: {best['model_id']}")
        assert best['model_id'] == model_id, "Best should be our model"
        
        # Get current best
        current = registry.get_current_best()
        assert current is not None, "Current best should exist"
        assert current['model_id'] == model_id, "Current best should match"
        
        # List models
        models_df = registry.list_models()
        print(f"  Models DataFrame:\n{models_df}")
        assert len(models_df) == 1, "Expected 1 model in list"
        
    print("✓ Model registration tests passed")
    return True


def test_model_loader():
    """Test ModelLoader class."""
    print("\n=== Test 6: ModelLoader ===")
    
    from recsys.cf.registry import (
        ModelRegistry,
        ModelLoader,
        REQUIRED_MODEL_FILES,
        reset_loader,
    )
    
    # Reset singleton
    reset_loader()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock artifacts
        model_path = os.path.join(tmpdir, 'als_model')
        os.makedirs(model_path)
        
        # Create embeddings and config files
        U = np.random.randn(100, 64).astype(np.float32)
        V = np.random.randn(50, 64).astype(np.float32)
        
        np.save(os.path.join(model_path, 'als_U'), U)
        np.save(os.path.join(model_path, 'als_V'), V)
        
        params = {'factors': 64, 'regularization': 0.1, 'alpha': 10}
        with open(os.path.join(model_path, 'als_params.json'), 'w') as f:
            json.dump(params, f)
        
        metadata = {
            'num_users': 100,
            'num_items': 50,
            'factors': 64,
            'training_date': datetime.now().isoformat()
        }
        with open(os.path.join(model_path, 'als_metadata.json'), 'w') as f:
            json.dump(metadata, f)
        
        # Create and register model
        registry_path = os.path.join(tmpdir, 'registry.json')
        registry = ModelRegistry(registry_path, auto_create=True)
        
        model_id = registry.register_model(
            artifacts_path=model_path,
            model_type='als',
            hyperparameters=params,
            metrics={'ndcg@10': 0.25},
            training_info={'training_time_seconds': 60}
        )
        
        registry.select_best_model(metric='ndcg@10')
        print(f"  Registered and selected model: {model_id}")
        
        # Create loader
        loader = ModelLoader(registry_path, auto_load=True)
        
        # Check current model
        info = loader.get_model_info()
        print(f"  Loaded model info: {info}")
        assert info['model_id'] == model_id, "Model ID should match"
        assert info['num_users'] == 100, "Num users should be 100"
        assert info['num_items'] == 50, "Num items should be 50"
        assert info['factors'] == 64, "Factors should be 64"
        
        # Get embeddings
        U_loaded, V_loaded = loader.get_embeddings()
        print(f"  U shape: {U_loaded.shape}, V shape: {V_loaded.shape}")
        assert U_loaded.shape == (100, 64), f"U shape mismatch: {U_loaded.shape}"
        assert V_loaded.shape == (50, 64), f"V shape mismatch: {V_loaded.shape}"
        
        # Check stats
        stats = loader.get_stats()
        print(f"  Loader stats: {stats}")
        assert stats['total_loads'] >= 1, "Should have at least 1 load"
        
        # Test reload
        changed = loader.reload_model()
        print(f"  Reload changed model: {changed}")
        assert not changed, "Model should not have changed"
        
        stats_after = loader.get_stats()
        assert stats_after['reload_count'] == 1, "Reload count should be 1"
        
    print("✓ ModelLoader tests passed")
    return True


def test_bert_embeddings_registry():
    """Test BERTEmbeddingsRegistry class."""
    print("\n=== Test 7: BERTEmbeddingsRegistry ===")
    
    from recsys.cf.registry import (
        BERTEmbeddingsRegistry,
        get_bert_registry,
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock embeddings
        emb_path = os.path.join(tmpdir, 'embeddings_v1')
        os.makedirs(emb_path)
        
        # Create embedding file
        embeddings = np.random.randn(100, 768).astype(np.float32)
        np.save(os.path.join(emb_path, 'product_embeddings'), embeddings)
        
        print(f"  Created mock embeddings at: {emb_path}")
        
        # Create registry
        registry_path = os.path.join(tmpdir, 'bert_registry.json')
        registry = BERTEmbeddingsRegistry(registry_path, auto_create=True)
        
        # Validate embeddings
        is_valid, missing = registry.validate_embedding_files(emb_path)
        print(f"  Embeddings valid: {is_valid}")
        assert is_valid, f"Expected valid embeddings, missing: {missing}"
        
        # Register embeddings
        version = registry.register_embeddings(
            embedding_path=emb_path,
            model_name='vinai/phobert-base',
            num_items=100,
            embedding_dim=768,
            text_fields_used=['product_name', 'description', 'ingredient']
        )
        
        print(f"  Registered embeddings: {version}")
        assert version.startswith('bert_'), f"Expected 'bert_' prefix, got {version}"
        
        # Get current best
        current = registry.get_current_best()
        print(f"  Current best: {current}")
        assert current is not None, "Should have current best"
        assert current['version'] == version, "Version should match"
        
        # List embeddings
        embeddings_list = registry.list_embeddings()
        print(f"  Embeddings list: {embeddings_list}")
        assert len(embeddings_list) == 1, "Expected 1 embedding"
        assert embeddings_list[0]['is_current'] == True, "Should be current"
        
        # Get stats
        stats = registry.get_stats()
        print(f"  Stats: {stats}")
        assert stats['total_embeddings'] == 1, "Expected 1 embedding"
        
        # Link to model
        registry.link_to_model(version, 'bert_als_v1')
        emb_info = registry.get_embeddings(version)
        assert 'bert_als_v1' in emb_info['linked_models'], "Model should be linked"
        
    print("✓ BERTEmbeddingsRegistry tests passed")
    return True


def test_backup_utilities():
    """Test backup and restore utilities."""
    print("\n=== Test 8: Backup Utilities ===")
    
    from recsys.cf.registry import (
        ModelRegistry,
        backup_registry,
        restore_registry,
        list_backups,
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        registry_path = os.path.join(tmpdir, 'registry.json')
        
        # Create registry
        registry = ModelRegistry(registry_path, auto_create=True)
        
        # Create backup
        backup_path = backup_registry(registry_path)
        print(f"  Created backup: {backup_path}")
        assert os.path.exists(backup_path), "Backup file should exist"
        
        # List backups
        backups = list_backups(registry_path)
        print(f"  Available backups: {len(backups)}")
        assert len(backups) == 1, "Should have 1 backup"
        
        # Modify registry
        with open(registry_path, 'r') as f:
            data = json.load(f)
        data['metadata']['test_key'] = 'test_value'
        with open(registry_path, 'w') as f:
            json.dump(data, f)
        
        # Restore (with backup of current)
        restore_registry(backup_path, registry_path, create_current_backup=False)
        
        # Verify restore
        with open(registry_path, 'r') as f:
            restored = json.load(f)
        assert 'test_key' not in restored['metadata'], "Should be restored"
        
    print("✓ Backup utilities tests passed")
    return True


def test_metadata_utilities():
    """Test metadata creation utilities."""
    print("\n=== Test 9: Metadata Utilities ===")
    
    from recsys.cf.registry import (
        create_model_metadata,
        save_model_metadata,
        load_model_metadata,
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create metadata
        metadata = create_model_metadata(
            num_users=26050,
            num_items=1423,
            factors=64,
            model_type='als',
            data_version='abc123',
            score_range=(-5.0, 15.0),
            extra={'custom_field': 'value'}
        )
        
        print(f"  Created metadata: {list(metadata.keys())}")
        assert metadata['num_users'] == 26050, "Num users should match"
        assert metadata['factors'] == 64, "Factors should match"
        assert 'score_range' in metadata, "Should have score_range"
        assert metadata['custom_field'] == 'value', "Should have custom field"
        
        # Save metadata
        model_path = os.path.join(tmpdir, 'model')
        saved_path = save_model_metadata(model_path, 'als', metadata)
        print(f"  Saved to: {saved_path}")
        assert os.path.exists(saved_path), "Metadata file should exist"
        
        # Load metadata
        loaded = load_model_metadata(model_path, 'als')
        print(f"  Loaded metadata keys: {list(loaded.keys())}")
        assert loaded['num_users'] == 26050, "Loaded num_users should match"
        
    print("✓ Metadata utilities tests passed")
    return True


def test_integration_with_existing_artifacts():
    """Test integration with existing artifact structure."""
    print("\n=== Test 10: Integration with Existing Artifacts ===")
    
    from recsys.cf.registry import (
        ModelRegistry,
        validate_model_path,
        load_model_metadata,
    )
    
    # Check if existing artifacts exist
    existing_artifacts = [
        'artifacts/cf/als',
        'artifacts/cf/bert_als',
        'artifacts/cf/bpr',
    ]
    
    found_artifacts = []
    for path in existing_artifacts:
        full_path = PROJECT_ROOT / path
        if full_path.exists():
            found_artifacts.append(str(full_path))
            print(f"  Found existing artifacts: {path}")
    
    if not found_artifacts:
        print("  No existing artifacts found, skipping integration test")
        print("✓ Integration test skipped (no artifacts)")
        return True
    
    # Try to validate existing artifacts
    for artifact_path in found_artifacts:
        model_type = Path(artifact_path).name
        
        # Check if it's a timestamped folder structure
        subdirs = list(Path(artifact_path).iterdir())
        
        for subdir in subdirs[:2]:  # Check first 2
            if subdir.is_dir() and not subdir.name.startswith('.'):
                is_valid, missing = validate_model_path(str(subdir), model_type)
                print(f"    {subdir.name}: valid={is_valid}, missing={missing}")
            elif subdir.is_file() and subdir.suffix == '.npy':
                # Direct file structure
                is_valid, missing = validate_model_path(artifact_path, model_type)
                print(f"    Direct structure: valid={is_valid}, missing={missing}")
                break
    
    print("✓ Integration test passed")
    return True


def run_all_tests():
    """Run all registry module tests."""
    print("=" * 70)
    print("MODEL REGISTRY MODULE TESTS")
    print("=" * 70)
    
    tests = [
        ("Imports", test_imports),
        ("Version Utilities", test_version_utilities),
        ("Git Utilities", test_git_utilities),
        ("ModelRegistry", test_model_registry),
        ("Model Registration", test_model_registration_with_mock),
        ("ModelLoader", test_model_loader),
        ("BERTEmbeddingsRegistry", test_bert_embeddings_registry),
        ("Backup Utilities", test_backup_utilities),
        ("Metadata Utilities", test_metadata_utilities),
        ("Integration", test_integration_with_existing_artifacts),
    ]
    
    results = []
    
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed, None))
        except Exception as e:
            print(f"✗ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False, str(e)))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, p, _ in results if p)
    total = len(results)
    
    for name, success, error in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {name}")
        if error:
            print(f"         Error: {error}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
