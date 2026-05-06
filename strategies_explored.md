# Seven Strategies Explored During Development

A reference document for the partner team. These are the seven strategies we built and evaluated during the exploration phase before consolidating to the three-phase final report (Phase 1 = Khang's tuned XGBoost, Phase 2 = CatBoost upgrade, Phase 3 = workflow features). Several of these were intermediate experiments — some succeeded, some failed, and we kept the ones that contributed to the final narrative.

All strategies were evaluated on a 5-fold `StratifiedGroupKFold` cross-validation grouped by `person_id`, with patient-level stratification on the maximum label. The metric reported is mean PR-AUC across folds, with the standard deviation as a measure of cross-fold consistency. The baseline at the time of these experiments was a HistGradientBoostingClassifier (used briefly before swapping to XGBoost in the final pipeline) with CV PR-AUC of 0.518 ± 0.180.

---

## Strategy 1 — CatBoost with `auto_class_weights='Balanced'`

**Hypothesis:** The PHEMS Lessons-Learned report explicitly named CatBoost as the model the highest-scoring teams found outperformed XGBoost, LightGBM, and transformer-based alternatives. We tested whether a direct model swap, holding feature pipeline and splits fixed, would meaningfully improve performance.

**What we built:** A `CatBoostClassifier` configured with `iterations=500`, `learning_rate=0.05`, `depth=6`, `l2_leaf_reg=3.0`, `auto_class_weights='Balanced'`, `eval_metric='PRAUC'`, and `early_stopping_rounds=30`. Categorical columns (`gender`, `admission_reason`) were passed natively via `cat_features` rather than one-hot encoded.

**Results:**
- CV PR-AUC: **0.628 ± 0.153** (matched the 1st place private LB score of 0.626 reported by the organizers)
- CV ROC-AUC: 0.970 ± 0.012
- Per-fold scores: [0.374, 0.745, 0.674, 0.740, 0.606]

**Verdict: Biggest single win of the project.** Switching from HGB to CatBoost moved CV PR-AUC by +0.110. This confirmed the report's central finding. However, variance remained high (0.153 standard deviation), with one fold scoring 0.37 and another 0.75.

**File:** `artifacts/submission_catboost.csv`

---

## Strategy 2 — Static TF-IDF on drug/route sequences ❌

**Hypothesis:** The Lessons-Learned report mentioned that the top teams "leveraged TF-IDF encoding for drug-related variables such as `drug_concept_id` and `route_concept_id`" to "extract early signals of sepsis onset by capturing patterns in medication administration." We interpreted this as a per-patient document approach: concatenate each patient's drug administrations chronologically into a "document," and TF-IDF encode the result as a static feature per patient.

**What we built:** A `TfidfVectorizer` (max_features=80 for drugs, 20 for routes, ngram=(1,2), min_df=5) fit on training drug-administration sequences. The resulting per-patient vectors were merged onto every (patient, hour) row alongside the demographics.

**Results:**
- CV PR-AUC: **0.523 ± 0.137** (substantially worse than vanilla CatBoost)
- Per-fold scores: [0.374, 0.621, 0.481, 0.706, 0.431]

**Verdict: Negative result. Deprecated.**

**Why it failed:** A patient's full-stay drug document includes drug administrations that occurred *after* the prediction time at any given (patient, hour) row. Adding these as static features lets the model recognize "patient identity" via drug fingerprints — including future drugs — and overfit to that. The model learned to use the static vectors to identify which patients eventually went septic, leaking outcome information back into early-stay predictions.

This was the catalyst for revisiting the approach as a *causal temporal* TF-IDF (see Strategy 5).

**File:** `artifacts/submission_catboost_tfidf.csv`

---

## Strategy 3 — Repeated K-fold ensemble (3 × 5 = 15 CatBoosts)

**Hypothesis:** Strategy 1's high variance (σ = 0.153) was likely driven by which patients ended up in each validation fold. Averaging predictions from CatBoosts trained on multiple different patient splits should reduce the variance of the resulting estimator.

**What we built:** Three independent shuffles of 5-fold `StratifiedGroupKFold`, each producing 5 trained CatBoosts. Test predictions averaged across all 15 models.

**Results:**
- 15-fold mean PR-AUC: **0.590 ± 0.133** (variance dropped from 0.180 in baseline to 0.133)
- Averaged-OOF PR-AUC: 0.579

**Verdict: Modest variance reduction; modest mean drop.** The dominant variance source turned out to be *between-fold patient-distribution differences*, not *within-fold model stochasticity*. Averaging more models on the same fold splits reduces only the second source. To reduce the first, we would need either more diverse feature sets or fundamentally different models — not just more CatBoosts.

**File:** `artifacts/submission_repeated_ensemble.csv`

---

## Strategy 4 — Pediatric and Phoenix-score component features ❌

**Hypothesis:** The label-generating function for the dataset is the Phoenix Sepsis Score (Schlapbach et al., JAMA 2024). If we engineered features that approximate the Phoenix sub-scores directly, those features would correlate strongly with the label. Additionally, pediatric vital signs vary dramatically with age — a heart rate of 130 is alarming in a teenager but normal in an infant — so age-normalized z-scores of vitals against PALS reference ranges should add signal.

**What we built (32 new features):**
- **Age-normalized vital z-scores** for HR, RR, SBP, DBP, temperature, and SpO2, using six pediatric age brackets from PALS guidelines. Both the signed z-score and the absolute z-score per vital.
- **Phoenix respiratory component:** SpO2/FiO2 ratio, plus binary "severe" (SF < 148) and "moderate" (SF < 220) flags.
- **Phoenix cardiovascular component:** mean arterial pressure (MAP = (SBP + 2·DBP) / 3), plus binary lactate-elevated flags at >5 and >2 mmol/L.
- **Phoenix coagulation component:** thrombocytopenia flags (platelets < 100, < 50), prothrombin time prolonged flag (PT > 14s), low fibrinogen flag (< 100).
- **Phoenix neurological component:** GCS-depressed flag (≤ 10) and severe (≤ 8).
- **Aggregate Phoenix proxy:** sum of all dysfunction flags as a continuous Phoenix-score approximation.
- **Time-since-last-measurement:** hours since the most recent non-NaN value for 7 sepsis-relevant labs (CRP, procalcitonin, lactate, WBC, platelets, HR, temperature).

**Results:**
- CV PR-AUC: **0.598 ± 0.134** (worse than vanilla CatBoost at 0.628)
- Per-fold scores: [0.393, 0.711, 0.567, 0.722, 0.592]

**Verdict: Negative result. Deprecated.**

**Why it failed:** **Zero of the 32 new features appeared in the top-30 feature-importance ranking.** Inspecting the top features revealed that CatBoost was already finding the right thresholds on raw `age_in_months`, raw lactate, raw platelets, raw GCS, etc. The age-conditional reasoning we'd encoded as z-scores was something the gradient-boosted tree learned natively via feature interactions. The Phoenix-component flags were simply restating what the model could already infer from the raw lab values.

**Lesson learned:** Hand-engineered features that *replicate* what gradient-boosted trees already learn from raw inputs add noise rather than signal. Feature engineering should target patterns the model *cannot* learn from per-row inputs — typically those requiring integration across multiple rows or aggregations from external tables. This insight directly motivated Strategy 6.

**File:** `artifacts/submission_pediatric_phoenix.csv`

---

## Strategy 5 — Lab rolling statistics + causal temporal TF-IDF

**Hypothesis:** Two course corrections from previous strategies:

1. The Lessons-Learned report explicitly mentioned rolling statistics on "vital signs *and* lab results" — but our existing pipeline only rolled vitals. We added rolling statistics to lab values as well.
2. Replace Strategy 2's failed static TF-IDF with a causal version: for each (patient, hour), build a TF-IDF vector from drugs administered in the past 12 hours (vocabulary fitted on training data only). This avoids the future-leakage problem while preserving the medication-pattern signal.

**What we built:**
- Rolling-window features (mean, std, min, max) over 6h and 24h windows for the 12 most-measured laboratory columns (Sodium, Potassium, Chloride, Hematocrit, Blood pH, Creatinine, WBC, Hemoglobin, Neutrophils, CRP, etc.).
- A causal temporal TF-IDF: per (patient, hour), construct a "document" from drugs administered in the past 12 hours, transform with a TfidfVectorizer (max_features=40, ngram=(1,1), min_df=10) fit on the full training drug corpus.

**Results:**
- CV PR-AUC: **0.572 ± 0.101** (lower mean than vanilla CatBoost, but **34% lower variance**)
- Per-fold scores: [0.407, 0.663, 0.595, 0.640, 0.556] — narrowest range of any strategy
- Three antibiotic TF-IDF features in top-30 importance: ciprofloxacin (rank #6, importance 4.14), erythromycin, clindamycin
- Four lab-rolling features in top-30: CRP w24 min, creatinine w24 max, CRP w6 min, sodium w24 max

**Verdict: Variance breakthrough.** The causal temporal TF-IDF worked where the static version had failed — antibiotic patterns over time genuinely added signal. The lab rolling statistics added smaller but real signal. Variance dropped from σ = 0.153 to σ = 0.101.

The mean dipped slightly because best-iteration counts dropped (CatBoost's early stopping triggered sooner with the richer feature set), suggesting the model was finding shortcuts faster. This pointed toward a regularization opportunity that we did not pursue further.

**File:** `artifacts/submission_lab_rolls_tfidf.csv`

---

## Strategy 6 — Clinical workflow features ✓ (became Phase 3 in final report)

**Hypothesis:** The lesson from Strategy 4 was that gradient-boosted trees handle within-row clinical-value reasoning natively. The lesson from Strategy 5 was that features capturing aggregations over time can add genuine signal. Combining these: engineer features that capture *meta-patterns of clinical decision-making* over time — patterns the model literally cannot derive from per-(patient, hour) values.

**What we built (9 features):**

| Feature | Definition |
|---|---|
| `hour_in_stay` | Hours since the patient's earliest record |
| `log_hour_in_stay` | `log(1 + hour_in_stay)`, compresses long-tail distribution |
| `is_first_24h` | Binary: 1 if `hour_in_stay < 24` |
| `drug_admin_count_6h` | Drug administrations in past 6 hours |
| `drug_admin_count_24h` | Drug administrations in past 24 hours |
| `proc_count_24h` | Procedure events in past 24 hours |
| `n_active_devices_24h` | Distinct device types recorded in past 24 hours |
| `has_et_tube_24h` | Binary flag for endotracheal tube in past 24 hours |
| `has_cvl_24h` | Binary flag for central venous line in past 24 hours |

All causal — only used events from time ≤ t. Computed via four pandas helper functions wrapping `groupby().rolling()` patterns.

**Results:**
- CV PR-AUC: **0.6321 ± 0.119** (highest mean of any single-strategy configuration, second-lowest variance)
- Per-fold scores: [0.460, 0.754, 0.641, 0.727, 0.578]
- ROC-AUC: 0.972 ± 0.010

**Feature importance breakthrough:** Three of the top six features were workflow features.
- `hour_in_stay` ranked #1 overall (importance 8.54)
- `drug_admin_count_6h` ranked #2 (importance 6.71)
- `log_hour_in_stay` ranked #6 (importance 4.47)

**Verdict: Breakthrough.** The missing-link result. Workflow features captured patterns CatBoost could not extract from per-row clinical-value features alone, and the test PR-AUC improvement validated the hypothesis.

This is what we kept and reported as Phase 3 in the final paper. Note that the Phase 3 implementation in the final paper uses Khang's feature pipeline as the base (rather than Strategy 5's lab-rolling-plus-TFIDF base), because we re-aligned the project to use Khang's tuned XGBoost as the Phase 1 baseline rather than continue from our own intermediate experiments.

**File:** `artifacts/submission_workflow.csv`

---

## Strategy 7 — Stacking with logistic-regression meta-learner

**Hypothesis:** Strategies 1, 5, and 6 each captured different aspects of the problem (raw clinical signal, variance-reduced temporal patterns, workflow metadata respectively). A stacked ensemble using a meta-learner trained on their out-of-fold predictions might combine their complementary strengths.

**What we built:**
1. Re-trained Strategies 1, 5, and 6 with 5-fold cross-validation, saving each fold's out-of-fold predictions on the training set.
2. Built a meta-feature matrix with three columns: `oof_S1`, `oof_S5`, `oof_S6`.
3. Trained a `LogisticRegression(C=1.0, max_iter=2000, solver='liblinear')` on this matrix with the true SepsisLabel as target.
4. At test time, applied each base model's fold-averaged test predictions through the meta-learner.

**Results:**
- CV PR-AUC: **0.6646 ± 0.118** (best mean of any strategy in the project)
- Per-fold scores: [0.461, 0.765, 0.685, 0.791, 0.622] — won 4 of 5 folds vs Strategy 6 alone
- Improvement over Strategy 6: +0.033 absolute on CV mean

**Meta-learner coefficients (diagnostic):**
- `oof_s6` (workflow features): **5.88** — dominant
- `oof_s1` (vanilla CatBoost): **3.22**
- `oof_s5` (lab rolls + temporal TF-IDF): **−0.07** (essentially ignored)

**Insight:** The meta-learner correctly identified Strategy 5 as redundant once Strategy 6's workflow features were present. Strategy 5's variance-reducing signal was already implicitly captured by Strategy 6's workflow features. The non-trivial blend of Strategies 1 and 6 (with appropriate scaling via the intercept) outperformed any individual base.

**Verdict: Best CV result, but did not survive the simplification to a three-phase narrative.** Stacking with a meta-learner is graduate-level methodology; we deprioritized it for the final report to keep the project at undergrad scope.

**File:** `artifacts/submission_stack_logreg.csv`

---

## Cross-strategy summary

Sorted by CV PR-AUC:

| Rank | Strategy | CV mean | CV σ | Verdict |
|---|---|---|---|---|
| 🥇 | 7. Stacked meta-learner (S1+S5+S6) | **0.665** | 0.118 | Best mean; deprioritized for simplicity |
| 🥈 | 6. Workflow features | 0.632 | 0.119 | Kept as Phase 3 in final report |
| 🥉 | 1. CatBoost (auto class weights) | 0.628 | 0.153 | Adapted as Phase 2 in final report |
| 4 | 4. Pediatric / Phoenix features | 0.598 | 0.134 | ❌ Deprecated (negative result) |
| 5 | 3. 15-model repeated ensemble | 0.590 | 0.133 | Modest variance reduction |
| 6 | 5. Lab rolls + temporal TF-IDF | 0.572 | 0.101 | Variance breakthrough |
| 7 | 2. Static TF-IDF on drugs | 0.523 | 0.137 | ❌ Deprecated (negative result, future leakage) |
| — | 0. HGB baseline | 0.518 | 0.180 | Original baseline before XGBoost swap |

---

## Cross-strategy correlation

To assess how independent the strategies were, we computed Spearman rank correlation on test predictions. Pairwise:

|  | HGB | CatBoost | TFIDF | RepEnsemble | Phoenix | LabRolls | Workflow |
|---|---|---|---|---|---|---|---|
| HGB | 1.00 | 0.76 | **0.47** | 0.71 | 0.70 | 0.68 | 0.69 |
| CatBoost | 0.76 | 1.00 | 0.62 | 0.94 | 0.93 | 0.89 | 0.85 |
| TFIDF | 0.47 | 0.62 | 1.00 | 0.65 | 0.67 | 0.63 | 0.61 |
| RepEnsemble | 0.71 | 0.94 | 0.65 | 1.00 | 0.97 | 0.92 | 0.87 |
| Phoenix | 0.70 | 0.93 | 0.67 | 0.97 | 1.00 | 0.91 | 0.85 |
| LabRolls | 0.68 | 0.89 | 0.63 | 0.92 | 0.91 | 1.00 | 0.88 |
| Workflow | 0.69 | 0.85 | 0.61 | 0.87 | 0.85 | 0.88 | 1.00 |

The strategies cluster into a single highly-correlated CatBoost family (correlations ≥ 0.85 among S1, S3, S4, S5, S6) plus an outlier — Strategy 2 (static TF-IDF) — which disagrees most with the others. This aligns with the importance ranking finding: the workflow features in S6 were genuinely complementary, but most other variations were minor perturbations on the same underlying CatBoost behavior.

---

## What we kept vs what we cut

The final three-phase report retains:
- **Phase 1** = Khang's tuned XGBoost (replaced our intermediate HGB baseline)
- **Phase 2** = Strategy 1 (CatBoost upgrade)
- **Phase 3** = Strategy 6 (workflow features), re-implemented on top of Khang's feature pipeline rather than the Strategy 5 base

Cut for simplicity:
- Strategy 2 (static TF-IDF) — negative result, kept out of the final report
- Strategy 3 (repeated ensemble) — modest gain, didn't fit the three-phase narrative
- Strategy 4 (pediatric/Phoenix) — negative result, useful methodological lesson but didn't justify a section
- Strategy 5 (lab rolls + temporal TF-IDF) — interesting variance result, but the workflow features in Strategy 6 dominated it
- Strategy 7 (stacking) — best CV result, but considered above undergrad scope and added implementation complexity (saving OOF predictions, meta-learner training) that was hard to defend without the research-paper framing

The final report's narrative is intentionally simpler than the experimental record. The simplifications were motivated by the project's grade-context (undergrad ML course) rather than by a finding that the cut strategies were wrong.
