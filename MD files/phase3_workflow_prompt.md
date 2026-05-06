# Phase 3: Clinical Workflow Features for Sepsis Prediction

## Overview
This document outlines the plan and prompts used to extend Phase 2 by adding nine clinical workflow features to the CatBoost feature set. The motivation comes from a structural property of gradient-boosted decision trees, not from any specific recommendation in the PHEMS organizers' report.

---

## Hypothesis

Gradient-boosted decision trees (XGBoost, CatBoost, LightGBM) split on individual columns of individual rows. At each split, a tree compares one column's value against a threshold and routes the row left or right. This is excellent for finding patterns in clinical *values* — vital signs, lab results, drug counts at a single hour — but it cannot derive aggregates over time without explicit feature engineering.

Specifically, a tree model cannot learn:
- **How long has this patient been in the ICU at this hour?**
- **How many drug administrations occurred in the past 6 hours?**
- **How many distinct lab tests have been ordered in the past 24 hours?**
- **Are invasive devices currently in use?**

These signals require integrating events across many rows of a patient's history or aggregating from a separate source table. They are about *clinical decision-making patterns* over time, not about any single measurement value. If we precompute these aggregates as columns and add them to the feature table, the tree gains the ability to split on them.

**Prediction:** adding workflow features will move test PR-AUC meaningfully (>0.05 improvement), and several workflow features will appear in the top-15 importance ranking.

---

## Context

**Phase 2 status (CatBoost on Khang's features):**
- Train PR-AUC: 0.810
- Validation PR-AUC: 0.400
- Test PR-AUC: 0.292
- Val→Test gap: 0.108 (good — small overfitting)

**What Phase 2 lacked:** the model has no view of clinical-decision metadata. Every feature is a single value at a single hour. Adding workflow features bridges that gap.

---

## Constraints

- **Do not modify Khang's pipeline files.** `data_prep.py`, `missing_values.py`, `XGBoost/` are read-only.
- All workflow features must be **causal**: at any prediction time t, only use information from time ≤ t.
- Workflow features must align row-for-row with `load_training_data()`'s output. Khang's `train_features.csv` has 30 rows with duplicate `(person_id, measurement_datetime)` keys; the merge logic must dedupe before computing rolling statistics, then expand back to the original row count via left-join.
- Use the same person-aware train/val/test split as Phase 1/2.
- Use the same CatBoost hyperparameters as Phase 2 (no tuning) — isolating the effect of the feature change.

---

## The Nine Workflow Features

### Group 1: Admission timing (3 features)
- `hour_in_stay` — hours since the patient's earliest record (proxy for hours-since-admission, computed per `person_id`)
- `log_hour_in_stay` — `log(1 + hour_in_stay)` to compress the long tail
- `is_first_24h` — binary, 1 if `hour_in_stay < 24`, 0 otherwise

### Group 2: Care intensity in rolling windows (3 features)
- `drug_admin_count_6h` — number of `drugsexposure` events for this patient in past 6 hours
- `drug_admin_count_24h` — same for past 24 hours
- `proc_count_24h` — number of `proceduresoccurrences` events in past 24 hours

### Group 3: Device burden (3 features)
- `n_active_devices_24h` — count of distinct device types from `devices` recorded in past 24 hours
- `has_et_tube_24h` — binary flag: was an Endotracheal tube recorded in past 24 hours?
- `has_cvl_24h` — binary flag: was a Central venous catheter recorded in past 24 hours?

---

## Strategy: Layered helper functions

### Prompt 1: Causal rolling-event-count helper
```
Write a helper function add_event_count_in_window(df, source_df,
source_dt_col, window_hours, feat_name) that:

1. Takes a master DataFrame with columns [person_id, measurement_datetime].
2. Takes a source DataFrame of events (e.g., drugsexposure_train.csv)
   with a person_id column and a datetime column.
3. Floors source datetimes to the hour.
4. Aggregates source events to per-(person, hour) counts.
5. Left-joins those counts onto the master, filling missing with 0.
6. Sorts by (person_id, measurement_datetime) and applies a rolling sum
   of `window_hours` rows within each patient (using groupby + rolling).
7. Drops the intermediate count column and returns df with one new
   column named feat_name.

The rolling sum must look only backward in time — pandas .rolling()
default behavior. Verify by spot-checking a known patient.
```

### Prompt 2: Distinct-device-count and invasive-device-flag helpers
```
Add two more helper functions:

1. add_distinct_devices_in_window(df, devices_df, window_hours,
   feat_name): like add_event_count_in_window, but counts the number
   of *distinct* device types per (person, hour) and rolling-sums those
   distinct counts over the window. Use nunique() during the per-hour
   aggregation step.

2. add_invasive_device_flag(df, devices_df, device_name, lookback_hours,
   feat_name): produces a binary flag for whether a specific device
   (e.g., 'Endotracheal tube') appeared in the past `lookback_hours`
   for this patient. Uses the same rolling-sum pattern but wraps the
   result with (rolling > 0).astype('int8').
```

### Prompt 3: Hour-in-stay
```
Add add_hour_in_stay(df) that:

1. Sorts df by (person_id, measurement_datetime) and resets index.
2. Computes first_t = group min of measurement_datetime per patient.
3. Adds three columns:
   - hour_in_stay = (measurement_datetime - first_t) in hours, float32
   - log_hour_in_stay = np.log1p(hour_in_stay), float32
   - is_first_24h = (hour_in_stay < 24).astype(int8)

This is the only causally-cheap feature in the set since first_t is
fixed at the patient's earliest record (in the past relative to any
prediction time).
```

### Prompt 4: Integration with Khang's pipeline
```
In phase3_workflow.ipynb:

1. Call load_training_data() to get Khang's df, base_features, labels.
2. Build workflow features keyed on the *unique* (person_id,
   measurement_datetime) pairs from df. Khang's table has 30 duplicate
   key rows; deduping before workflow computation prevents a
   cross-product blowup. Then left-join wf_aligned back onto df's
   row order (which preserves duplicates correctly).
3. Concat workflow columns to base_features to get augmented_features
   (267 columns: Khang's 258 + 9 workflow).
4. Call split_data_person_aware(df, augmented_features, labels, ...)
   for the same train/val/test split used in Phases 1 and 2.
5. Train CatBoost with the same hyperparameters as Phase 2.
6. Print feature_importances and verify workflow features appear in
   the top-15.
```

### Prompt 5: Kaggle submission
```
Build the submission with the same approach as Phase 2, but apply
build_workflow_features() to the test event tables (drugsexposure_test,
proceduresoccurrences_test, devices_test) and merge those workflow
columns onto X_kaggle before predicting. Reindex columns to match
augmented_features.columns so the model sees exactly the column set
it was trained on.
```

---

## Success Criteria
- Phase 3 test PR-AUC > Phase 2 test PR-AUC (by at least 0.05; ideally >0.10)
- At least three workflow features appear in the top-15 by CatBoost feature importance
- Submission passes format validation (130,483 rows, matching keys, probabilities in [0,1])
- Train PR-AUC should *not* increase dramatically — a big train-only jump would suggest the workflow features are leaking. Test-set improvement is the real signal.

---

## Actual Result
- Test PR-AUC: 0.438 (vs Phase 2's 0.292 — **+0.146**, exceeding the 0.05 success threshold by a wide margin)
- Train PR-AUC: 0.746 (lower than Phase 2's 0.810 — opposite of what leakage would produce; consistent with workflow features adding generalizable signal rather than memorization)
- Top-15 importance ranking: 4 of 15 are workflow features
  - #2: `drug_admin_count_6h` (importance 13.18)
  - #3: `log_hour_in_stay` (importance 12.18)
  - #11: `n_active_devices_24h`
  - #15: `proc_count_24h`
- The two highest-importance features after `route_Intravenous` are both workflow features.

---

## Files Produced
- `phase3_workflow.ipynb` — Phase 3 notebook with all helper functions inline
- `artifacts/submission_workflow.csv` — Kaggle submission (130,483 rows)

---

## Lessons / Reflections
1. **Feature engineering > model swap.** Phase 1 → 2 (model swap, default hyperparameters): −0.016. Phase 2 → 3 (workflow features, same model): +0.146. The model choice was less important than the feature change for this dataset.
2. **The duplicate-key bug** in Khang's master table is not a bug per se — multiple visits per patient at the same hour can produce duplicates. The fix is to dedupe during workflow feature computation, then left-join (which preserves duplicates correctly) when merging back. Spent ~10 minutes debugging the resulting length-mismatch error.
3. **Causality is a design property, not just a rule.** All nine workflow features are causal because they're either (a) per-patient minimum-time references (`hour_in_stay`) that are fixed in the past, or (b) backward-only rolling aggregates of events with concrete timestamps. Pandas `.rolling()` is causal by default; we did not need any special enforcement.
