# XGBoost Hyperparameter Tuning Workflow for Sepsis Prediction

## Overview
This document outlines a structured workflow to tune XGBoost hyperparameters for the sepsis prediction problem using Bayesian Optimization. The goal is to maximize PR AUC (Precision-Recall AUC) while maintaining good generalization on validation and test sets.

---

## Context: Your Problem
- **Dataset**: Imbalanced sepsis classification (many negative cases, fewer positive)
- **Current Baseline**: 
  - Train: ROC AUC 0.9977, PR AUC 0.9397
  - Validation: ROC AUC 0.9902, PR AUC 0.8760
  - Test: ROC AUC 0.9916, PR AUC 0.8991
- **Target**: Improve PR AUC on validation/test without overfitting to train

---

## Tuning Strategy: 3-Phase Approach

### Phase 1: Tree Structure Parameters
**Goal**: Find the right tree depth and complexity without overfitting

**Hyperparameters to tune**:
- `max_depth`: Tree depth (range: 3-10, start: 5)
- `min_child_weight`: Minimum leaf node weight (range: 1-10, start: 1)
- `gamma`: Minimum loss reduction for split (range: 0-5, start: 0)

**Prompt 1**: 
```
Create a tuning script that:
1. Uses GridSearchCV or Bayesian Optimization (hyperopt) to search over max_depth, min_child_weight, gamma
2. Uses validation PR AUC as the optimization metric
3. Trains on X_train/y_train with eval_set=(X_val, y_val)
4. Prints best parameters and their corresponding train/val PR AUC scores
5. Shows if model is overfitting (train PR AUC >> val PR AUC)
```

---

### Phase 2: Regularization & Subsampling
**Goal**: Add regularization to prevent overfitting while keeping predictive power

**Hyperparameters to tune** (with Phase 1 best params fixed):
- `subsample`: Fraction of samples for each tree (range: 0.5-1.0, start: 0.8)
- `colsample_bytree`: Fraction of features for each tree (range: 0.5-1.0, start: 0.8)
- `lambda` (reg_lambda): L2 regularization (range: 0-5, start: 1)
- `alpha` (reg_alpha): L1 regularization (range: 0-5, start: 0)

**Prompt 2**:
```
Create a tuning script that:
1. Fixes the best params from Phase 1
2. Searches over subsample, colsample_bytree, lambda, alpha
3. Uses validation PR AUC as the metric
4. Prints best regularization params and improvement over Phase 1
5. Generates a plot showing PR AUC vs each regularization parameter
```

---

### Phase 3: Learning Rate & Boosting
**Goal**: Fine-tune learning rate and number of boosting rounds for final model

**Hyperparameters to tune** (with Phase 1 & 2 best params fixed):
- `learning_rate` (eta): Step size shrinkage (range: 0.01-0.3, start: 0.05)
- `n_estimators`: Number of boosting rounds (range: 100-500, start: 200)

**Prompt 3**:
```
Create a tuning script that:
1. Fixes the best params from Phase 1 & 2
2. Searches over learning_rate and n_estimators
3. Uses early stopping: train on X_train with eval_set=(X_val, y_val)
4. Uses validation PR AUC as the early stopping metric
5. Reports final best params and test PR AUC performance
6. Generates a learning curve showing train/val PR AUC over boosting rounds
```

---

## Baseline Configuration (Current)
```python
XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    scale_pos_weight=3.45,  # ratio of negative to positive
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
    eval_metric="aucpr",
)
```

---

## Success Criteria
- ✅ Validation PR AUC improves from 0.8760 to >= 0.90
- ✅ Train PR AUC remains close to validation (< 0.05 gap)
- ✅ Test PR AUC >= 0.90 (generalization check)
- ✅ Model trains in < 5 minutes per configuration

---

## Implementation Requirements
1. **File Structure**: Create `xgboost_tuning.py` in the project root
2. **Modular Functions**: 
   - `phase1_tune_tree_structure()` 
   - `phase2_tune_regularization()`
   - `phase3_tune_learning()`
   - `train_final_model(best_params)`
3. **Output**: Save results to `tuning_results.json` and `tuning_plots/` folder
4. **Reproducibility**: Use `random_state=42`, fix train/val/test splits

---

## Prompt Summary
**Ready to proceed with Phase 1 tuning script?**

When approved, I will create:
1. `xgboost_tuning.py` with all three phases
2. Integration with your existing `train_model.py` 
3. Visualization of tuning progress
4. Final model comparison (baseline vs tuned)

---

## Questions Before Implementation
- Should I use GridSearchCV (simpler) or Bayesian Optimization with hyperopt (more powerful)?
- How much time are you willing to spend on tuning? (Quick: 10 min, Thorough: 1 hour)
- Should I tune on PR AUC only, or also consider recall at a specific precision target?
