# Pediatric Sepsis Prediction — Detailed Spec (post-EDA)

**Status:** Plan revised after running EDA on 2026-05-03. Replaces the proposal where they conflict.
**Deadline:** 2026-05-05 (~2 days).

---

## 1. EDA findings that change the plan

### 1.1 Class imbalance is more extreme than the proposal anticipated

| Level | Positive rate | Counts |
|---|---|---|
| Hour (prediction unit) | **2.07%** | 6,874 / 331,653 |
| Patient | **3.66%** | 97 / 2,649 |

**Implication:** SMOTE at the row level is risky — it would synthesize "positive" hours for patients who never went septic, which is medically nonsensical and creates leakage across the patient boundary. The proposal's mention of SMOTE should be downgraded to a fallback. **Primary approach: `scale_pos_weight` in XGBoost + threshold tuning on PR-AUC.**

### 1.2 Lead-time distribution is bimodal

Of the 97 septic patients:
- **15** are septic on arrival (lead time = 0)
- **24** become septic within the first 6 hours
- **54** have ≥24h of healthy history before going septic
- Median lead time: 40 hours

**Implication:** The model has to perform under both regimes. Features that depend on long history (24-hour rolling stats) will be NaN for ~25% of positive cases. We need short-window features (1h, 3h, 6h) AND long-window features (12h, 24h), and missingness indicators for windows that don't yet exist.

### 1.3 Feature-table coverage is sparse

Fraction of label hours that have at least one row in each source table:

| Table | Coverage |
|---|---|
| Vitals (`measurement_meds`) | **75.5%** |
| Devices | 55.1% |
| Procedures | 59.0% |
| **Labs** | **20.5%** |
| **Drugs** | **20.0%** |
| Observations | 14.9% |

**Implication:** point-in-time joins will be NaN-dominated. We need:
- **Forward-fill within patient** (carry last known value forward) for vitals + labs
- **Time-since-last-measurement** features (a long gap since last lab is itself informative)
- Keep raw NaNs after forward-fill so XGBoost can route them — don't median-impute

### 1.4 Within-table missingness is uneven

**Vitals** (within rows that exist in vitals table):
- Body temperature: 21% missing — usable
- HR / SpO2: 62% — usable
- Respiratory rate: 68% — usable
- SBP / DBP: 84% — borderline; treat as "available when measured"
- FiO2: 99% — drop or use as indicator only

**Labs**: 6 of 36 columns are >96% missing and should be **dropped**:
- D-dimer (96.3%)
- Interleukin 6 (99.9%)
- Ionised calcium (100%)
- Blood venous pH (99.5%)
- Arterial blood gas panel (PaO2, PaCO2, HCO3-arterial, base excess) — all 96.3%

The remaining ~30 labs have 50–90% missingness *within their own table*. Combined with the 20% table-coverage stat, this means most label hours will have zero lab data. **Forward-fill is essential.**

### 1.5 Event tables are low-cardinality (good news)

- Devices: 5 unique types
- Procedures: 7 unique types
- Drugs: 50 concept IDs, 9 routes

The partner's pivot-to-wide approach in `data_prep.py` is appropriate here — no top-K filtering needed. The top-50 cap on `observation_concept_name` may also be unnecessary (need to check observation cardinality).

### 1.6 No patient overlap between train and test

Confirmed: the test set is patient-disjoint. We can split train any way we want for CV without temporal-leakage concerns within train. **Use `StratifiedGroupKFold(groups=person_id)`** so positive-patient counts are balanced across folds.

---

## 2. What changes vs the original proposal

| Proposal | Revised plan | Why |
|---|---|---|
| **PCA for dimensionality reduction** | **Drop PCA.** Use XGBoost feature importance + L1-regularized logistic-regression baseline for selection if needed. | Final feature count is ~200–400 after engineering — well within XGBoost's comfort zone. PCA destroys interpretability (a hard requirement per our proposal's "interpretable feature importances") and rarely helps tree models. |
| **KNN baseline** | **Logistic regression baseline** (with `class_weight='balanced'`). | KNN with 300+ features, NaNs everywhere, and 2% positive class is a poor baseline. We'd have to fully impute, which contradicts our missing-as-feature design. LogReg gives a defensible linear baseline that handles imbalance natively. |
| **XGBoost primary** | **XGBoost primary** + LightGBM if time permits. | Unchanged. XGBoost's native NaN handling is exactly what this dataset needs. |
| **SMOTE OR class weighting** | **`scale_pos_weight` only**, plus threshold tuning. | SMOTE on row-level data with patient-grouped positives is leakage-prone. Class weighting is simpler and won't fabricate clinically impossible rows. |
| **Rolling averages, trends, rate-of-change** | Same, **plus**: forward-fill within patient, time-since-last-measurement, multiple window sizes (1h/3h/6h/12h/24h). | The 20% coverage on labs/drugs forces forward-fill to make point-in-time predictions feasible. |
| **AUC-ROC + PR metrics, stratified k-fold** | Same, **but** `StratifiedGroupKFold(groups=person_id)` for proper patient-level holdouts. **Primary metric: AUC-ROC** (assumed Kaggle metric — verify on competition page); **secondary: PR-AUC** (more honest given imbalance). | Row-level stratification leaks: same patient in train and val makes the val score optimistic. |
| Missing-data patterns as features | Done already by partner in `missing_values.py`; **integrate, don't rewrite**. | Code exists. |

---

## 3. Pipeline architecture (what we're actually building)

```
raw CSVs
  │
  ├─► [data_prep.py]  → hourly per-patient feature table (master left-join on labels)
  │     └─ fix bug: compute top-K observation names on TRAIN only, apply to BOTH splits
  │
  ├─► [time_series_features.py]  (NEW)
  │     ├─ forward-fill numeric cols within (person_id, time-sorted)
  │     ├─ rolling stats (mean/std/min/max) over [3, 6, 12, 24] hours
  │     ├─ rate-of-change: x(t) − x(t−1h), x(t) − x(t−6h)
  │     └─ time-since-last-measurement per column
  │
  ├─► [missing_values.py]  (existing) → ±inf → NaN, missing indicators, row-level missing summary
  │
  ├─► [train_model.py]  (rewrite)
  │     ├─ StratifiedGroupKFold(n_splits=5, groups=person_id, shuffle=True, seed=42)
  │     ├─ baseline: LogisticRegression(class_weight='balanced', max_iter=1000)
  │     ├─ primary: XGBoost(scale_pos_weight=neg/pos, eval_metric='auc',
  │     │                    early_stopping_rounds=50)
  │     ├─ tune: max_depth ∈ {4,6,8}, lr ∈ {0.05,0.1}, n_est ∈ {200,500,1000}
  │     │       subsample=0.8, colsample_bytree=0.8
  │     └─ save per-fold AUC-ROC and PR-AUC, mean ± std
  │
  └─► [predict_and_submit.py]  (NEW)
        ├─ refit best model on FULL train
        ├─ predict_proba on test features → second column
        ├─ build submission: person_id_datetime = f"{person_id}_{measurement_datetime}"
        └─ write submission.csv (validate row count == 130,483)
```

---

## 4. Phased plan (mapped to remaining time)

### Phase 1 — Verify pipeline works end-to-end (today, May 3, 2–3 hours)
1. Run partner's `data_prep.main()`. Check `artifacts/train_features.csv` and `test_features.csv` exist.
2. **Fix bug:** in `load_observation_table`, compute top-K names from train, then pass that fixed list when loading test. Otherwise train and test feature columns won't align.
3. Verify train/test column sets are identical (after one-hot encoding gender). Add an assertion.
4. Run `prepare_for_xgboost` end-to-end. Confirm shape.
5. **Submit a baseline (constant 0.05 or LogReg) to Kaggle today.** Pinning down the submission format now eliminates one risk vector.

### Phase 2 — Time-series features (May 4 morning, 2–3 hours)
1. New module `time_series_features.py`. Operates on the merged hourly table.
2. Forward-fill within `person_id` (sorted by time) for vitals + labs only.
3. Rolling stats per patient: 3h, 6h, 12h, 24h windows over key vitals + top labs.
4. Rate-of-change features for HR, RR, SBP, temperature, lactate, WBC, CRP.
5. Time-since-last-measurement per source table (1 feature per table, not per column — keep it manageable).

### Phase 3 — Model training & tuning (May 4 afternoon, 3 hours)
1. LogReg baseline → record CV AUC-ROC + PR-AUC.
2. XGBoost with sensible defaults → record CV scores.
3. Coarse hyperparameter sweep (grid above is small enough to brute-force in <30 min).
4. Pick best config. Refit on full train.

### Phase 4 — Final submission + iterate (May 4 evening)
1. Predict on test, build submission, upload.
2. Screenshot submission confirmation + leaderboard rank.
3. If time: try LightGBM, try ensemble (avg of LogReg + XGBoost probabilities).

### Phase 5 — Report + slides + video (May 5)
1. Report PDF (8–12 pages): Abstract / Intro / Methodology / Results / Conclusion. Cite this spec, the EDA notebook, the partner's AI-assisted scaffolding (per academic-honesty rule), and the missing-data paper already referenced in README (arxiv 2411.09591).
2. Slides + ~10-min recorded video.
3. Upload everything to Canvas.

---

## 5. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Submission format wrong → 0 score | Medium | Submit a constant-probability baseline TODAY to validate format end-to-end |
| Train/test column mismatch (top-K obs bug) | High if not fixed | Phase 1 step 2 + assertion |
| Patient-level CV variance (only 97 positives) | High | Use 5-fold StratifiedGroupKFold; report both mean and per-fold scores; consider 3-fold if folds are too noisy |
| Forward-fill creates leakage | Low | Only fill within `(person_id, time-ascending)`; never across patients or backwards in time |
| Kaggle metric isn't AUC-ROC | Medium | Check the competition's overview/evaluation page before final submission. Plan B is whatever metric they specify (likely AUC, possibly PR-AUC) |
| Time runs out before report | Medium | Hard-stop modeling work end of May 4; reserve all of May 5 for write-up |

---

## 6. Open questions (resolve before Phase 2)

1. **Confirm the official Kaggle evaluation metric.** Proposal assumed AUC-ROC + PR-AUC. Need to read the competition's "Evaluation" tab.
2. Is observation `top_k=50` the right cap? Need to count `observation_concept_name.nunique()` in `observation_train.csv`. If it's <100 total, drop the cap.
3. Should we model at hour resolution (current plan) or aggregate to e.g. 4-hour blocks? Hour is the submission resolution, so default is hour. Aggregation could help if predictions are too noisy.

---

## 7. Deliverables checklist

- [ ] Working pipeline with verified train/test column alignment
- [ ] Time-series features module
- [ ] CV results table (LogReg vs XGBoost, AUC + PR-AUC, mean ± std)
- [ ] At least one valid Kaggle submission with screenshot of confirmation
- [ ] Final Kaggle submission with screenshot of leaderboard rank
- [ ] Final report PDF (8–12 pp, single column, ≤11pt) with all 5 sections
- [ ] Slide deck
- [ ] ~10-minute recorded presentation video
- [ ] All source code committed, cleaned, commented
