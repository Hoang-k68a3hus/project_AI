# Strategic Update: Trainable User Threshold Adjustment (≥2 Interactions)

**Date**: November 19, 2025  
**Decision**: Lower trainable user threshold from ≥3 to ≥2 interactions  
**Impact**: ~26,000 trainable users (~8.6% coverage), ~274,000 cold-start users (~91.4%)

---

## Executive Summary

After analyzing the dataset distribution, we've identified that with a **≥2 interaction threshold**, the system will have:
- **26,000 trainable users (8.6%)** eligible for CF training
- **Matrix density ~0.11%** (26k × 2.2k items with ~65k interactions)
- **90%+ traffic** will be served via content-based (PhoBERT) + popularity

This is a **strategic sweet spot** that balances:
✅ Sufficient statistical base for CF (26k users)  
✅ BERT initialization compensates for sparsity  
✅ Realistic traffic distribution (content-first approach)  

---

## Key Metrics

| Metric | Value | Context |
|--------|-------|---------|
| Total users | ~300,000 | From raw data |
| Trainable users (≥2) | ~26,000 (8.6%) | Eligible for CF training |
| Cold-start users | ~274,000 (91.4%) | Content-based only |
| Avg interactions/trainable user | ~2.5 | Estimated |
| Matrix density | ~0.11% | 65k interactions / (26k × 2.2k) |
| Total interactions | ~369,000 | From raw data |

---

## Strategic Adjustments

### 1. Data Layer (Task 01) ✅
**Changes**:
- User filtering: `min_user_interactions: 2` (down from 3)
- Additional requirement: `min_user_positives: 1` (at least 1 rating ≥4)
- **Edge case**: Users with 2 interactions but both negative (<4) → Force to cold-start
- Updated `data_stats.json` to reflect 26k trainable users

**Files Updated**:
- `tasks/01_data_layer.md` - Step 2.3 (User Filtering)
- `tasks/01_data_layer.md` - Step 4.3 (Edge Cases)
- `tasks/01_data_layer.md` - Step 6.4 (Statistics Summary)

---

### 2. Training Pipelines (Task 02) ✅
**Changes**:
- **Higher regularization**: `λ = 0.05-0.15` (up from 0.01) to anchor sparse user vectors to BERT semantic space
- **Lower alpha**: `α = 5-10` (down from 20-40) due to confidence_score range 1-6
- Grid search expanded: 54 configs (was 36) to cover higher regularization values

**Rationale**:
With users having only 2 interactions, regularization prevents vectors from drifting too far from BERT-initialized semantic space. This is critical for maintaining recommendation quality.

**Files Updated**:
- `tasks/02_training_pipelines.md` - Grid Search Strategy
- `tasks/02_training_pipelines.md` - Stage 1 Coarse Search
- `tasks/02_training_pipelines.md` - Strategy 1 BERT Initialization (added note)

---

### 3. Serving Layer (Task 05) ✅
**Changes**:
- Updated routing logic: Check `is_trainable_user` flag (≥2 interactions + ≥1 positive)
- Traffic distribution updated:
  - **CF path**: ~8.6% traffic (trainable users)
  - **Content-based path**: ~91.4% traffic (cold-start users)
- Added performance note: Content-based infrastructure must be optimized for majority traffic

**Files Updated**:
- `tasks/05_serving_layer.md` - Context section
- `tasks/05_serving_layer.md` - Serving Flow diagram
- `tasks/05_serving_layer.md` - Benefits section

---

### 4. Hybrid Reranking (Task 08) ✅
**Changes**:
- **Trainable users** (≥2, ~8.6% traffic):
  - `w_content=0.40`, `w_cf=0.30`, `w_popularity=0.20`, `w_quality=0.10`
- **Cold-start users** (~91.4% traffic):
  - `w_content=0.60`, `w_popularity=0.30`, `w_quality=0.10`, `w_cf=0.0`
- Added critical note: 91.4% traffic uses cold-start path → optimize content-based latency

**Files Updated**:
- `tasks/08_hybrid_reranking.md` - Content-First Hybrid Strategy section

---

## Technical Implementation Notes

### A. BERT Initialization Strategy
```python
# Higher regularization anchors user vectors to semantic space
als_model = AlternatingLeastSquares(
    factors=64,
    regularization=0.1,  # Higher for sparse users (was 0.01)
    iterations=15,
    alpha=10  # Lower due to confidence_score range 1-6 (was 40)
)

# Initialize item factors from BERT
V_init = project_bert_to_factors(bert_embeddings, target_dim=64)
als_model.item_factors = V_init
```

### B. Test Set Handling for 2-Interaction Users
```python
# User with exactly 2 interactions
if user_interaction_count == 2:
    interactions = sorted_interactions_by_timestamp(user_id)
    
    if interactions[0]['rating'] >= 4 and interactions[1]['rating'] >= 4:
        # Both positive → Use older for train, newer for test
        train_set.append(interactions[0])
        test_set.append(interactions[1])
        
    elif interactions[0]['rating'] >= 4 or interactions[1]['rating'] >= 4:
        # 1 positive, 1 negative → Positive to train, no test data
        positive = [i for i in interactions if i['rating'] >= 4][0]
        train_set.append(positive)
        # User excluded from evaluation (no test)
        
    else:
        # Both negative → Force to cold-start
        user_metadata[user_id]['is_trainable_user'] = False
```

### C. Regularization Impact
With λ=0.1 and BERT initialization:
- User vectors U stay closer to weighted average of their interacted items' BERT vectors
- Prevents "random drift" that happens with only 2 data points
- Trade-off: Slightly lower personalization, but more stable and semantically meaningful

---

## Validation Checklist

Before implementing, verify:

- [ ] **Data Layer**:
  - [ ] `min_user_interactions = 2` in `data_config.yaml`
  - [ ] `min_user_positives = 1` enforced in `preprocess_interactions()`
  - [ ] Edge case: 2 interactions, both negative → `is_trainable_user = False`
  - [ ] `data_stats.json` logs trainable user count ~26k

- [ ] **Training**:
  - [ ] ALS regularization in grid: `[0.05, 0.1, 0.15]`
  - [ ] ALS alpha in grid: `[5, 10, 20]`
  - [ ] BERT initialization + higher regularization tested together
  - [ ] Training logs report matrix density ~0.11%

- [ ] **Serving**:
  - [ ] `user_metadata.pkl` loaded at startup
  - [ ] Routing checks `is_trainable_user` flag
  - [ ] Cold-start users (91.4%) routed to content-based path
  - [ ] Latency monitoring for both paths

- [ ] **Hybrid Reranking**:
  - [ ] Trainable weights: `[0.40, 0.30, 0.20, 0.10]`
  - [ ] Cold-start weights: `[0.60, 0.30, 0.10, 0.0]`
  - [ ] Weight selection based on `is_trainable_user` flag

---

## Expected Outcomes

### Positive Impacts
1. **More training data**: 26k users vs previous ~10-15k (estimated with ≥3 threshold)
2. **Statistical significance**: 26k samples sufficient for CF learning with BERT support
3. **Realistic architecture**: Acknowledges 90%+ traffic is cold-start → optimize accordingly
4. **Better cold-start**: Content-first approach gets explicit focus and optimization

### Trade-offs
1. **Slightly noisier data**: Users with only 2 interactions have weak collaborative signal
2. **Mitigation**: Higher regularization (λ=0.1) + BERT init prevents overfitting to noise
3. **Evaluation challenge**: Many users have only 1 test sample (the 2nd interaction)

### Performance Expectations
- **CF Recall@10**: ~0.15-0.20 (lower than typical CF due to sparsity)
- **Content Recall@10**: ~0.25-0.30 (primary workhorse)
- **Hybrid Recall@10**: ~0.28-0.35 (combined strength)

---

## Monitoring & Iteration

### Key Metrics to Track
1. **Coverage**:
   - % requests hitting CF path (~8.6%)
   - % requests hitting content-based path (~91.4%)

2. **Performance by User Type**:
   - CF users: Recall@10, NDCG@10, latency
   - Cold-start users: Content similarity scores, diversity, latency

3. **Drift Detection**:
   - Trainable user percentage over time (should stay ~8-10%)
   - Matrix density (should stay ~0.1-0.2%)

### Iteration Plan
- **Week 1-2**: Implement ≥2 threshold, measure baseline metrics
- **Week 3-4**: Tune regularization (λ) via grid search
- **Week 5-6**: Optimize content-based latency (primary bottleneck)
- **Week 7+**: A/B test CF vs content-only for trainable users

---

## References

**Updated Task Files**:
1. `tasks/01_data_layer.md` - User filtering, edge cases, statistics
2. `tasks/02_training_pipelines.md` - Regularization, grid search, BERT init
3. `tasks/05_serving_layer.md` - Routing logic, traffic distribution
4. `tasks/08_hybrid_reranking.md` - Weight strategies for trainable vs cold-start

**Key Sections**:
- Task 01, Step 2.3: User Filtering (≥2 threshold)
- Task 02, Grid Search: Higher regularization (0.05-0.15)
- Task 05, Context: Traffic distribution (8.6% CF, 91.4% content)
- Task 08, Content-First Strategy: Dual weight configurations

---

## Conclusion

The **≥2 interaction threshold** with **26,000 trainable users** is the optimal balance for this dataset:
- Large enough for statistical learning (26k samples)
- Compensated by BERT initialization + higher regularization
- Realistic traffic distribution (90%+ cold-start) drives content-first architecture
- Clear separation between CF and content-based paths enables targeted optimization

This strategy acknowledges the **data reality** (high sparsity, rating skew) and builds an architecture that **thrives on content-based signals** while using CF as a **supporting component** for the minority of users with sufficient interaction history.

**Ready to proceed with implementation of Task 08 (Hybrid Reranking)** using these parameters. 🚀
