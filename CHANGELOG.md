# Changelog

All notable changes to the Vietnamese Cosmetics Recommendation System will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-11-25

### 🚀 Initial Production Release

This is the first production-ready release of the hybrid recommendation system combining Collaborative Filtering (ALS/BPR) with content-based filtering (PhoBERT) for Vietnamese cosmetics.

### Added

#### Data Layer (Task 01)
- **DataProcessor**: Unified orchestration interface for data pipeline
- **DataValidator**: Input validation with strict timestamp/rating checks
- **FeatureEngineer**: Comment quality scoring, positive/hard-negative labeling
- **UserFilter**: User segmentation (trainable ≥2 interactions vs cold-start)
- **IDMapper**: Contiguous ID mapping (user_id/product_id → u_idx/i_idx)
- **TemporalSplitter**: Leave-one-out temporal splitting for evaluation
- **MatrixBuilder**: CSR sparse matrix construction for ALS/BPR
- **DataSaver**: Parquet, NPZ, JSON, PKL artifact persistence
- **VersionRegistry**: Data versioning with MD5 hash tracking

#### Training Pipelines (Task 02)
- **ALS Training**: Alternating Least Squares with sentiment-enhanced confidence
- **BPR Training**: Bayesian Personalized Ranking with hard negative sampling
- **BERT-ALS**: PhoBERT initialization for item embeddings
- Grid search support for hyperparameter optimization
- Colab notebooks for GPU training

#### Evaluation (Task 03)
- Recall@K, NDCG@K, Precision@K metrics
- Popularity baseline comparison
- Coverage and diversity analysis
- Cold-start vs trainable user breakdown

#### Model Registry (Task 04)
- JSON-based model registry with versioning
- Automatic best model selection
- Model metadata and score range tracking
- Hot-reload support for production

#### Serving Layer (Task 05)
- **CFRecommender**: Main recommendation engine
- **CFModelLoader**: Singleton model loader with hot-reload
- **FallbackRecommender**: Cold-start handling strategies
- **PhoBERTEmbeddingLoader**: Content embedding management
- FastAPI REST endpoints:
  - `POST /recommend`: Single user recommendations
  - `POST /batch_recommend`: Batch recommendations
  - `POST /similar_items`: Item-item similarity
  - `POST /reload_model`: Hot-reload model
  - `GET /health`: Health check
  - `GET /stats`: Service statistics

#### Monitoring & Logging (Task 06)
- Request latency tracking
- Fallback usage monitoring
- Cache hit rate metrics
- Structured logging with rotation

#### Hybrid Reranking (Task 08)
- **HybridReranker**: Multi-signal reranking engine
- Configurable weights for CF, content, popularity, quality
- Global normalization for consistent scoring
- Diversity penalty using BERT similarity
- Adaptive weights for trainable vs cold-start users

#### Deployment Optimization
- **CacheManager**: LRU caching with TTL support
  - User profile cache (50K entries, 1h TTL)
  - Item similarity cache (5K entries, 24h TTL)
  - Fallback result cache (10K entries, 30min TTL)
- **Warm-up Strategy**: Pre-compute popular items and similarities
- Load testing scripts with latency profiling
- Deployment optimization analyzer

### Configuration Files
- `config/serving_config.yaml`: Serving layer configuration
- `config/als_config.yaml`: ALS training configuration
- `config/bert_als_config.yaml`: BERT-ALS configuration
- `config/alerts_config.yaml`: Alerting thresholds

### Documentation
- `docs/DEPLOYMENT_OPTIMIZATION.md`: Performance tuning guide
- `docs/OPERATIONS_GUIDE.md`: Operations manual (this release)
- `tasks/*.md`: Architecture specifications (00-08)

### Performance Characteristics
- **Trainable Users**: ~26K users (8.6%) with ≥2 interactions
- **Cold-Start Users**: ~274K users (91.4%) with <2 interactions
- **Target Latency**: <200ms P99 for all paths
- **Expected Metrics**:
  - ALS Recall@10: >0.20
  - BPR Recall@10: >0.22
  - Hybrid Recall@10: ~0.28-0.35

### Known Limitations
- Cold-start path handles 91.4% of traffic (content-based only)
- PhoBERT embeddings required for content similarity
- Item-item similarity matrix must fit in memory (<3K items for pre-computation)

### Dependencies
- Python 3.9+
- FastAPI 0.100+
- NumPy, SciPy, Pandas
- PyTorch (for PhoBERT embeddings)
- implicit (for ALS)
- aiohttp (for load testing)

---

## [Unreleased]

### Planned
- FAISS integration for approximate nearest neighbor search
- Redis distributed caching for multi-instance deployment
- A/B testing framework
- Real-time model updates with Kafka
- User preference learning from implicit feedback

---

## Version History

| Version | Date | Description |
|---------|------|-------------|
| 1.0.0 | 2025-11-25 | Initial production release |

