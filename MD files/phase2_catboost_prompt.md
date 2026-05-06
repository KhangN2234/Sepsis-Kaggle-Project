# Phase 2: CatBoost Model Swap on Existing XGBoost Pipeline

## Overview
This document outlines the plan and prompts used to extend Khang's tuned XGBoost pipeline (Phase 1) by swapping the model to CatBoost while holding the feature pipeline, splits, and preprocessing identical. The change is motivated by the PHEMS organizers' published Lessons-Learned report, not by any specific top-team's solution.

---

## Context

**Phase 1 status (Khang's tuned XGBoost):**
- Train PR-AUC: 0.881
- Validation PR-AUC: 0.784
- Test PR-AUC: 0.308
- Train→Test gap: 0.573 (substantial overfitting)

**Trigger for Phase 2:** the PHEMS Consortium's *Pediatric Sepsis Prediction Hackathon: Results and Lessons Learned* report (Zenodo record 17045913, 2025) explicitly names CatBoost with `auto_class_weights='Balanced'` as the model the highest-scoring teams used, outperforming XGBoost, LightGBM, and transformer-based alternatives. The report contains conceptual descriptions only — no code, no specific hyperparameters.

**Goal:** test that recommendation on our own pipeline by swapping XGBoost for CatBoost, holding everything else identical. This isolates the effect of the model choice from the effect of any feature changes.

---

## Constraints

- **Do not modify Khang's pipeline files.** `data_prep.py`, `missing_values.py`, `XGBoost/` are read-only for this phase.
- Reuse `load_training_data()`, `split_data_person_aware()`, and `score_model()` from `XGBoost/train_XGBoost.py` for an apples-to-apples comparison with Phase 1.
- Use the same person-aware train/val/test split (random_state=42, test_size=0.2, val_size=0.2).
- No hyperparameter tuning in Phase 2. The point is to isolate the model-swap effect.

---

## Strategy: Direct Drop-in Replacement

### Model configuration
Settings come from the organizers' report and CatBoost defaults:
- `auto_class_weights='Balanced'` — the report's named recommendation for class imbalance
- `eval_metric='PRAUC'` — competition metric, used for early stopping
- `loss_function='Logloss'`
- Tree settings: depth 6, learning rate 0.05, l2_leaf_reg 3.0, 500 iterations
- Early stopping: 30 rounds without PR-AUC improvement on the validation set

### Prompt 1: Notebook scaffolding
```
Create phase2_catboost.ipynb that mirrors phase1_xgboost.ipynb's
structure but replaces XGBoost with CatBoostClassifier.

Requirements:
1. Import data_prep, prepare_for_xgboost, and the helpers from
   XGBoost/train_XGBoost.py (load_training_data, split_data_person_aware,
   score_model). Do not duplicate Khang's code.
2. Apply the same person-aware split (random_state=42).
3. Use CatBoost with auto_class_weights='Balanced', eval_metric='PRAUC',
   depth=6, learning_rate=0.05, l2_leaf_reg=3.0, 500 iterations,
   early_stopping_rounds=30, random_seed=42.
4. Pass the validation set as eval_set for early stopping.
5. Print the best iteration after training.
```

### Prompt 2: Evaluation
```
Add an evaluation cell that calls Khang's score_model() function for
each of train / val / test splits. score_model uses threshold=0.2 for
the threshold-dependent metrics (recall, precision, accuracy) and
reports PR-AUC and ROC-AUC.

Format the output to match the style of Phase 1's evaluation block so
the two phases can be visually compared.
```

### Prompt 3: Kaggle submission
```
Add a submission cell that:
1. Reads artifacts/test_features.csv (Khang's engineered Kaggle test
   features, produced by data_prep.py).
2. Builds the person_id_datetime key in the same format as
   SepsisLabel_sample_submission.csv.
3. Applies prepare_for_xgboost() with the same indicator/summary
   settings used during training.
4. Reindexes columns to match the training feature column set
   (so the model sees exactly the columns it was trained on).
5. Writes the submission to artifacts/submission_catboost.csv with
   the assertion that row count and key set match the sample.
```

---

## Success Criteria
- Phase 2 test PR-AUC within 0.05 of Phase 1 (model-swap alone unlikely to dramatically move ranking)
- Validation-to-test gap should *shrink* relative to Phase 1, evidence that CatBoost is less prone to validation overfitting
- Submission file passes format validation (130,483 rows, matching keys, probabilities in [0,1])

---

## Actual Result
- Test PR-AUC: 0.292 (within 0.05 of Phase 1's 0.308 — confirmed)
- Val→Test gap: 0.108 vs Phase 1's 0.476 (substantial reduction — confirmed)
- Conclusion: CatBoost generalizes more conservatively. Test-set ranking unchanged.
- Decision: model swap alone is not enough. Move to Phase 3 (feature engineering).

---

## Files Produced
- `phase2_catboost.ipynb` — Phase 2 notebook
- `artifacts/submission_catboost.csv` — Kaggle submission (130,483 rows)
