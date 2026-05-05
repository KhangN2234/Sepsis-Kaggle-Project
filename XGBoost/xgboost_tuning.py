import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from skopt import gp_minimize, space
from skopt.utils import use_named_args
import json
import time
from pathlib import Path
from joblib import dump as joblib_dump
from missing_values import prepare_for_xgboost
from XGBoost.train_XGBoost import fit_xgboost

# Try to import StratifiedGroupKFold if available; fallback to StratifiedKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:
    StratifiedGroupKFold = None
from sklearn.model_selection import StratifiedKFold

 # Global variables to store data across phases
X_train, X_val, X_test, y_train, y_val, y_test = [None] * 6
GLOBAL_SPW = None
baseline_config = {
    'n_estimators': 500,
    'learning_rate': 0.05,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'gamma': 0,
    'reg_lambda': 1,
    'reg_alpha': 0,
    'objective': 'binary:logistic',
}


def load_split_data(group_stratify: bool = True, test_size: float = 0.2, val_size: float = 0.2, random_state: int = 42):
    """Load engineered features and perform group-aware stratified outer splits.

    Reads `artifacts/train_features.csv` to preserve `person_id` for group splits,
    then applies `prepare_for_xgboost` to feature columns (excluding `person_id`).
    """
    global X_train, X_val, X_test, y_train, y_val, y_test, GLOBAL_SPW

    path = Path('artifacts') / 'train_features.csv'
    df = pd.read_csv(path)

    if 'person_id' not in df.columns:
        raise RuntimeError('person_id column required for group-aware splitting')

    # person-level label for stratification: whether the person ever had sepsis
    person_label = df.groupby('person_id')['SepsisLabel'].max()
    persons = person_label.index.to_numpy()
    person_labels = person_label.values

    # Stratified split of persons into train_val and test (group-aware stratification)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(sss.split(persons, person_labels))
    train_val_persons = persons[train_val_idx]
    test_persons = persons[test_idx]

    # Further split train_val into train and val (stratified by person-level label)
    remaining = train_val_persons
    remaining_labels = person_label.loc[remaining].values
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size / (1.0 - test_size), random_state=random_state)
    train_idx, val_idx = next(sss2.split(remaining, remaining_labels))
    train_persons = remaining[train_idx]
    val_persons = remaining[val_idx]

    train_df = df[df['person_id'].isin(train_persons)].copy()
    val_df = df[df['person_id'].isin(val_persons)].copy()
    test_df = df[df['person_id'].isin(test_persons)].copy()

    # Prepare features (drop person_id, measurement_datetime before preparing)
    def prep(df_in):
        y = df_in['SepsisLabel'].astype(int)
        X = df_in.drop(columns=['SepsisLabel', 'measurement_datetime']) if 'measurement_datetime' in df_in.columns else df_in.drop(columns=['SepsisLabel'])
        return X

    X_train_raw = prep(train_df)
    X_val_raw = prep(val_df)
    X_test_raw = prep(test_df)

    # Apply feature preparation on copies that exclude person_id
    X_train_proc = X_train_raw.drop(columns=['person_id']).copy()
    X_val_proc = X_val_raw.drop(columns=['person_id']).copy()
    X_test_proc = X_test_raw.drop(columns=['person_id']).copy()

    X_train_proc = prepare_for_xgboost(X_train_proc, label_column=None, add_indicators=True, add_summary=True)
    X_val_proc = prepare_for_xgboost(X_val_proc, label_column=None, add_indicators=True, add_summary=True)
    X_test_proc = prepare_for_xgboost(X_test_proc, label_column=None, add_indicators=True, add_summary=True)

    y_train = train_df['SepsisLabel'].astype(int)
    y_val = val_df['SepsisLabel'].astype(int)
    y_test = test_df['SepsisLabel'].astype(int)

    # Assign globals
    X_train, X_val, X_test = X_train_proc, X_val_proc, X_test_proc
    globals()['y_train'] = y_train
    globals()['y_val'] = y_val
    globals()['y_test'] = y_test

    # Save group arrays aligned to processed feature indices for inner CV
    globals()['TRAIN_GROUPS'] = train_df.loc[X_train_proc.index, 'person_id'].to_numpy()
    globals()['VAL_GROUPS'] = val_df.loc[X_val_proc.index, 'person_id'].to_numpy()
    globals()['TEST_GROUPS'] = test_df.loc[X_test_proc.index, 'person_id'].to_numpy()

    # Compute global scale_pos_weight from training set (fixed across folds)
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    GLOBAL_SPW = float(neg / pos) if pos > 0 else 1.0

    print(f"Data loaded: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")
    print(f"Global scale_pos_weight set to {GLOBAL_SPW:.3f}")


def evaluate_params_via_cv(model_params, n_splits=5, random_state=42):
    """Evaluate given XGBoost params via inner CV on the training set using PR-AUC with fold-level early stopping.

    Uses StratifiedGroupKFold when available, otherwise StratifiedKFold (groups ignored).
    For each fold, splits fold-training data into fold-train (80%) and fold-early-stop (20%)
    to enable early stopping without validation-set reuse.
    Returns mean PR-AUC across folds (higher is better).
    """
    X = X_train.copy()
    y = y_train.copy()
    groups = globals().get('TRAIN_GROUPS', None)

    pr_scores = []

    if StratifiedGroupKFold is not None and groups is not None:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = splitter.split(X, y, groups)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = splitter.split(X, y)

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        # Outer fold split: fold-train and fold-val
        X_fold_train_full = X.iloc[train_idx].copy()
        y_fold_train_full = y.iloc[train_idx].copy()
        X_fold_val = X.iloc[val_idx].to_numpy()
        y_fold_val = y.iloc[val_idx].to_numpy()

        # Inner fold split (within fold-train): fold-train (80%) and fold-early-stop (20%)
        es_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state + fold_idx)
        es_train_idx, es_stop_idx = next(es_splitter.split(X_fold_train_full, y_fold_train_full))
        
        X_fold_train = X_fold_train_full.iloc[es_train_idx].to_numpy()
        y_fold_train = y_fold_train_full.iloc[es_train_idx].to_numpy()
        X_fold_early_stop = X_fold_train_full.iloc[es_stop_idx].to_numpy()
        y_fold_early_stop = y_fold_train_full.iloc[es_stop_idx].to_numpy()

        # Ensure scale_pos_weight passed
        params = model_params.copy()
        params['scale_pos_weight'] = GLOBAL_SPW

        model = xgb.XGBClassifier(
            **params,
            tree_method='hist',
            eval_metric='aucpr',
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )

        # Fit with fold-level early stopping on fold-early-stop
        try:
            model.fit(
                X_fold_train, y_fold_train,
                eval_set=[(X_fold_early_stop, y_fold_early_stop)],
                early_stopping_rounds=20,
                verbose=False,
            )
        except TypeError:
            # Fallback if early_stopping_rounds not supported by XGBoost wrapper
            model.fit(X_fold_train, y_fold_train)

        y_fold_val_proba = model.predict_proba(X_fold_val)[:, 1]
        _, pr = compute_pr_auc(y_fold_val, y_fold_val_proba)
        pr_scores.append(pr)

    mean_pr = float(np.mean(pr_scores))
    return -mean_pr, mean_pr

def compute_pr_auc(y_true, y_pred_proba):
    """Compute PR-AUC (average precision). Return negative for minimization and raw PR-AUC."""
    pr_auc = average_precision_score(y_true, y_pred_proba)
    return -pr_auc, pr_auc

def phase1_tune_tree_structure(max_evals=20):
    """
    Phase 1: Optimize tree structure parameters (max_depth, min_child_weight, gamma).
    Uses Bayesian Optimization to find best tree complexity without overfitting.
    """
    print("\n" + "="*60)
    print("PHASE 1: Tuning Tree Structure Parameters")
    print("="*60)
    
    # Define search space for Phase 1
    search_space = [
        space.Integer(3, 10, name='max_depth'),
        space.Integer(1, 10, name='min_child_weight'),
        space.Real(0, 5, name='gamma'),
    ]
    
    best_result = {'loss': float('inf'), 'params': None}
    call_count = [0]  # Use list to track across function calls
    
    @use_named_args(search_space)
    def objective(max_depth, min_child_weight, gamma):
        call_count[0] += 1
        
        # Build model with Phase 1 params + baseline for others
        model_params = baseline_config.copy()
        model_params.update({
            'max_depth': max_depth,
            'min_child_weight': min_child_weight,
            'gamma': gamma,
        })
        # Evaluate via inner CV on training set (PR-AUC)
        neg_mean_pr, mean_pr = evaluate_params_via_cv(model_params, n_splits=5)

        # Train on full training set for diagnostics and evaluate on outer val
        model = xgb.XGBClassifier(
            **model_params,
            tree_method='hist',
            eval_metric='aucpr',
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(X_train.to_numpy(), y_train.to_numpy())

        y_val_pred_proba = model.predict_proba(X_val.to_numpy())[:, 1]
        _, val_pr = compute_pr_auc(y_val, y_val_pred_proba)

        y_train_pred_proba = model.predict_proba(X_train.to_numpy())[:, 1]
        _, train_pr = compute_pr_auc(y_train, y_train_pred_proba)

        print(f"  [{call_count[0]}/{max_evals}] max_depth={max_depth}, min_child_weight={min_child_weight}, gamma={gamma:.2f} "
              f"| CV mean PR: {mean_pr:.4f} | Val PR: {val_pr:.4f} | Train PR: {train_pr:.4f}")

        # Track best result (based on CV mean PR)
        if neg_mean_pr < best_result['loss']:
            best_result['loss'] = neg_mean_pr
            best_result['params'] = {
                'max_depth': max_depth,
                'min_child_weight': min_child_weight,
                'gamma': gamma,
            }
            best_result['val_pr_auc'] = val_pr
            best_result['train_pr_auc'] = train_pr

        return neg_mean_pr
    
    # Run Bayesian Optimization
    result = gp_minimize(
        objective,
        search_space,
        n_calls=max_evals,
        random_state=42,
        n_initial_points=5,
        verbose=0,
    )
    
    best_params = best_result['params']
    
    print(f"\nPhase 1 Best Parameters: {best_params}")
    print(f"  Validation PR AUC: {best_result['val_pr_auc']:.4f}")
    print(f"  Train PR AUC: {best_result['train_pr_auc']:.4f}")
    print(f"  Overfitting gap (train - val PR AUC): {best_result['train_pr_auc'] - best_result['val_pr_auc']:.4f}")
    
    return best_params

def phase2_tune_regularization(phase1_params, max_evals=20):
    """
    Phase 2: Optimize regularization & subsampling (subsample, colsample_bytree, lambda, alpha).
    Keeps Phase 1 best params fixed.
    """
    print("\n" + "="*60)
    print("PHASE 2: Tuning Regularization & Subsampling Parameters")
    print("="*60)
    
    # Define search space for Phase 2
    search_space = [
        space.Real(0.5, 1.0, name='subsample'),
        space.Real(0.5, 1.0, name='colsample_bytree'),
        space.Real(0, 5, name='reg_lambda'),
        space.Real(0, 5, name='reg_alpha'),
    ]
    
    best_result = {'loss': float('inf'), 'params': None}
    call_count = [0]
    
    @use_named_args(search_space)
    def objective(subsample, colsample_bytree, reg_lambda, reg_alpha):
        call_count[0] += 1
        
        # Build model with Phase 1 + Phase 2 params
        model_params = baseline_config.copy()
        model_params.update(phase1_params)
        model_params.update({
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'reg_lambda': reg_lambda,
            'reg_alpha': reg_alpha,
        })
        # Evaluate via inner CV on training set (PR-AUC)
        neg_mean_pr, mean_pr = evaluate_params_via_cv(model_params, n_splits=5)

        # Train on full training set for diagnostics
        model = xgb.XGBClassifier(
            **model_params,
            tree_method='hist',
            eval_metric='aucpr',
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(X_train.to_numpy(), y_train.to_numpy())

        y_val_proba = model.predict_proba(X_val.to_numpy())[:, 1]
        _, val_pr = compute_pr_auc(y_val, y_val_proba)

        y_train_proba = model.predict_proba(X_train.to_numpy())[:, 1]
        _, train_pr = compute_pr_auc(y_train, y_train_proba)

        print(f"  [{call_count[0]}/{max_evals}] subsample={subsample:.2f}, colsample={colsample_bytree:.2f}, "
              f"lambda={reg_lambda:.2f}, alpha={reg_alpha:.2f} | CV mean PR: {mean_pr:.4f} | Val PR: {val_pr:.4f}")

        if neg_mean_pr < best_result['loss']:
            best_result['loss'] = neg_mean_pr
            best_result['params'] = {
                'subsample': subsample,
                'colsample_bytree': colsample_bytree,
                'reg_lambda': reg_lambda,
                'reg_alpha': reg_alpha,
            }
            best_result['val_pr_auc'] = val_pr
            best_result['train_pr_auc'] = train_pr

        return neg_mean_pr
    
    # Run Bayesian Optimization
    result = gp_minimize(
        objective,
        search_space,
        n_calls=max_evals,
        random_state=42,
        n_initial_points=5,
        verbose=0,
    )
    
    best_params = best_result['params']
    
    print(f"\nPhase 2 Best Parameters: {best_params}")
    print(f"  Validation PR AUC: {best_result['val_pr_auc']:.4f}")
    print(f"  Train PR AUC: {best_result['train_pr_auc']:.4f}")
    
    return best_params

def phase3_tune_learning(phase1_params, phase2_params, max_evals=15):
    """
    Phase 3: Optimize learning rate & n_estimators with early stopping.
    Keeps Phase 1 & 2 best params fixed.
    """
    print("\n" + "="*60)
    print("PHASE 3: Tuning Learning Rate & Boosting Rounds (with Early Stopping)")
    print("="*60)
    
    # Define search space for Phase 3
    # Increased n_estimators range since fold-level early stopping now picks optimal rounds
    search_space = [
        space.Real(0.01, 0.3, prior='log-uniform', name='learning_rate'),
        space.Integer(200, 2000, name='n_estimators'),
    ]
    
    best_result = {'loss': float('inf'), 'params': None}
    call_count = [0]
    
    @use_named_args(search_space)
    def objective(learning_rate, n_estimators):
        call_count[0] += 1
        
        # Build model with all previous phases
        model_params = baseline_config.copy()
        model_params.update(phase1_params)
        model_params.update(phase2_params)
        model_params.update({
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
        })
        # Evaluate via inner CV on training set (PR-AUC)
        neg_mean_pr, mean_pr = evaluate_params_via_cv(model_params, n_splits=5)

        # Train on full training set for diagnostics and evaluate on outer val
        model = xgb.XGBClassifier(
            **model_params,
            tree_method='hist',
            eval_metric='aucpr',
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(X_train.to_numpy(), y_train.to_numpy())

        y_val_proba = model.predict_proba(X_val.to_numpy())[:, 1]
        _, val_pr = compute_pr_auc(y_val, y_val_proba)

        y_train_proba = model.predict_proba(X_train.to_numpy())[:, 1]
        _, train_pr = compute_pr_auc(y_train, y_train_proba)

        print(f"  [{call_count[0]}/{max_evals}] learning_rate={learning_rate:.4f}, n_estimators={n_estimators} "
              f"| CV mean PR: {mean_pr:.4f} | Val PR: {val_pr:.4f} | Train PR: {train_pr:.4f}")

        if neg_mean_pr < best_result['loss']:
            best_result['loss'] = neg_mean_pr
            best_result['params'] = {
                'learning_rate': learning_rate,
                'n_estimators': n_estimators,
            }
            best_result['val_pr_auc'] = val_pr
            best_result['train_pr_auc'] = train_pr

        return neg_mean_pr
    
    # Run Bayesian Optimization
    result = gp_minimize(
        objective,
        search_space,
        n_calls=max_evals,
        random_state=42,
        n_initial_points=5,
        verbose=0,
    )
    
    best_params = best_result['params']
    
    print(f"\nPhase 3 Best Parameters: {best_params}")
    print(f"  Validation PR AUC: {best_result['val_pr_auc']:.4f}")
    print(f"  Train PR AUC: {best_result['train_pr_auc']:.4f}")
    
    return best_params

def train_final_model(best_params):
    """Train final model on train+val combined and evaluate on test set.
    
    Uses all non-test data for training to maximize signal, then evaluates on test.
    Note: final hyperparameters were tuned via inner CV, so retraining on train+val
    does not constitute tuning on the test set.
    
    Returns: (model, results_dict, val_proba, y_val_labels)
    where val_proba and y_val_labels can be used for threshold selection.
    """
    print("\n" + "="*60)
    print("FINAL MODEL: Training on Train+Val with Best Parameters")
    print("="*60)
    
    # Build final model
    model_params = baseline_config.copy()
    model_params.update(best_params)
    # Ensure consistent scale_pos_weight
    model_params['scale_pos_weight'] = GLOBAL_SPW
    
    # Combine train and val for final training (all non-test data)
    X_final_train = pd.concat([X_train, X_val], axis=0)
    y_final_train = pd.concat([y_train, y_val], axis=0)
    
    X_final_train_np = X_final_train.to_numpy()
    X_val_np = X_val.to_numpy()
    X_test_np = X_test.to_numpy()
    y_final_train_np = y_final_train.to_numpy()
    y_val_np = y_val.to_numpy()
    y_test_np = y_test.to_numpy()
    
    model = xgb.XGBClassifier(
        **model_params,
        tree_method='hist',
        eval_metric='aucpr',
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    
    # Fit final model on combined train+val (no eval_set; early stopping not used for final training)
    model.fit(X_final_train_np, y_final_train_np, verbose=False)
    
    # Evaluate on test set (threshold-independent metric only)
    y_test_pred_proba = model.predict_proba(X_test_np)[:, 1]
    pr_auc = average_precision_score(y_test_np, y_test_pred_proba)
    
    print(f"\nTEST SET PR-AUC (threshold-independent): {pr_auc:.4f}")
    print("Note: Threshold-dependent metrics (recall, FNR, confusion matrix) will be computed separately")
    print("      using find_threshold.py after clinically optimal threshold is selected.")
    
    # Generate validation predictions for threshold selection
    y_val_pred_proba = model.predict_proba(X_val_np)[:, 1]
    
    results = {
        'test_pr_auc': float(pr_auc),
        'test_samples': int(len(y_test_np)),
        'val_samples': int(len(y_val_np)),
    }
    
    return model, results, y_val_pred_proba, y_val_np

def main():
    """Main orchestrator for 3-phase hyperparameter tuning."""
    start_time = time.time()
    
    print("\n" + "#"*60)
    print("# XGBoost Hyperparameter Tuning: 3-Phase Bayesian Optimization")
    print("# Objective: Maximize PR AUC (inner CV) — no validation leakage")
    print("#"*60)
    
    # Load and split data
    load_split_data()
    
    # Phase 1: Tree Structure (10 evals, ~3-4 min)
    print(f"\nStarting Phase 1 at {time.time() - start_time:.1f}s...")
    phase1_params = phase1_tune_tree_structure(max_evals=10)
    print(f"Phase 1 completed in {time.time() - start_time:.1f}s")
    
    # Phase 2: Regularization (10 evals, ~3-4 min)
    print(f"\nStarting Phase 2 at {time.time() - start_time:.1f}s...")
    phase2_params = phase2_tune_regularization(phase1_params, max_evals=10)
    print(f"Phase 2 completed in {time.time() - start_time:.1f}s")
    
    # Phase 3: Learning Rate (8 evals, ~2-3 min)
    print(f"\nStarting Phase 3 at {time.time() - start_time:.1f}s...")
    phase3_params = phase3_tune_learning(phase1_params, phase2_params, max_evals=8)
    
    # Combine all best parameters
    best_params = baseline_config.copy()
    best_params.update(phase1_params)
    best_params.update(phase2_params)
    best_params.update(phase3_params)
    
    # Train final model
    final_model, results, y_val_pred_proba, y_val_labels = train_final_model(best_params)
    
    # Save results and model
    output_dir = Path('tuning_results')
    output_dir.mkdir(exist_ok=True)
    
    # Save the trained model
    model_path = output_dir / 'final_model.pkl'
    joblib_dump(final_model, model_path)
    print(f"\nModel saved to {model_path}")
    
    # Save validation predictions for threshold selection
    val_pred_df = pd.DataFrame({
        'true_label': y_val_labels,
        'pred_proba': y_val_pred_proba,
    })
    val_pred_path = output_dir / 'val_predictions.csv'
    val_pred_df.to_csv(val_pred_path, index=False)
    print(f"Validation predictions saved to {val_pred_path}")
    
    # Convert numpy types to Python types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(v) for v in obj]
        elif hasattr(obj, 'item'):  # numpy types
            return obj.item()
        else:
            return obj
    
    results_to_save = {
        'best_parameters': convert_to_native(best_params),
        'performance_metrics': convert_to_native(results),
        'total_time_seconds': time.time() - start_time,
    }
    
    with open(output_dir / 'tuning_results.json', 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    print("\n" + "#"*60)
    print(f"# Tuning Complete! Total time: {time.time() - start_time:.1f}s")
    print(f"# Results saved to tuning_results/tuning_results.json")
    print(f"# Final model trained on train+val combined ({X_train.shape[0] + X_val.shape[0]} samples)")
    print(f"# Evaluated on test set ({X_test.shape[0]} samples)")
    print("#"*60)
    
    return final_model, best_params, results

if __name__ == '__main__':
    final_model, best_params, results = main()
