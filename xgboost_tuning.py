import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, confusion_matrix
from skopt import gp_minimize, space
from skopt.utils import use_named_args
import json
import time
from pathlib import Path
from missing_values import prepare_for_xgboost
from train_model import load_training_data, split_data, fit_xgboost

# Global variables to store data across phases
X_train, X_val, X_test, y_train, y_val, y_test = None, None, None, None, None, None
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
}

def load_split_data():
    """Load and split data once for all phases."""
    global X_train, X_val, X_test, y_train, y_val, y_test
    features, labels = load_training_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(features, labels)
    print(f"Data loaded: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")

def compute_score_with_recall(y_true, y_pred_proba, threshold=0.5):
    """
    Compute combined score: PR AUC + Recall.
    Returns: -(0.5 * PR_AUC + 0.5 * Recall)
    This objective minimizes false negatives by heavily weighting recall.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    pr_auc = average_precision_score(y_true, y_pred_proba)
    recall = recall_score(y_true, y_pred)
    
    # Combine metrics: balance PR AUC with recall (minimize FN by high recall)
    combined_score = 0.5 * pr_auc + 0.5 * recall
    
    # Return negative because hyperopt minimizes
    return -combined_score, pr_auc, recall

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
        
        # Convert to numpy for XGBoost
        X_train_np = X_train.to_numpy()
        X_val_np = X_val.to_numpy()
        y_train_np = y_train.to_numpy()
        y_val_np = y_val.to_numpy()
        
        # Train model
        model = xgb.XGBClassifier(
            **model_params,
            tree_method='hist',
            eval_metric='aucpr',
            random_state=42,
            n_jobs=-1,
            verbose=0,
        )
        
        model.fit(
            X_train_np, y_train_np,
            eval_set=[(X_val_np, y_val_np)],
            verbose=False
        )
        
        # Evaluate on validation
        y_val_pred_proba = model.predict_proba(X_val_np)[:, 1]
        combined_score, pr_auc, recall = compute_score_with_recall(y_val_np, y_val_pred_proba)
        
        # Also compute train to detect overfitting
        y_train_pred_proba = model.predict_proba(X_train_np)[:, 1]
        _, train_pr_auc, train_recall = compute_score_with_recall(y_train_np, y_train_pred_proba)
        
        print(f"  [{call_count[0]}/{max_evals}] max_depth={max_depth}, min_child_weight={min_child_weight}, gamma={gamma:.2f} "
              f"| Val PR AUC: {pr_auc:.4f}, Recall: {recall:.4f} | Train PR AUC: {train_pr_auc:.4f}")
        
        # Track best result
        if combined_score < best_result['loss']:
            best_result['loss'] = combined_score
            best_result['params'] = {
                'max_depth': max_depth,
                'min_child_weight': min_child_weight,
                'gamma': gamma,
            }
            best_result['val_pr_auc'] = pr_auc
            best_result['val_recall'] = recall
            best_result['train_pr_auc'] = train_pr_auc
            best_result['train_recall'] = train_recall
        
        return combined_score
    
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
    print(f"  Validation PR AUC: {best_result['val_pr_auc']:.4f}, Recall: {best_result['val_recall']:.4f}")
    print(f"  Train PR AUC: {best_result['train_pr_auc']:.4f}, Recall: {best_result['train_recall']:.4f}")
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
        
        X_train_np = X_train.to_numpy()
        X_val_np = X_val.to_numpy()
        y_train_np = y_train.to_numpy()
        y_val_np = y_val.to_numpy()
        
        model = xgb.XGBClassifier(
            **model_params,
            tree_method='hist',
            eval_metric='aucpr',
            random_state=42,
            n_jobs=-1,
            verbose=0,
        )
        
        model.fit(
            X_train_np, y_train_np,
            eval_set=[(X_val_np, y_val_np)],
            verbose=False
        )
        
        # Evaluate
        y_val_pred_proba = model.predict_proba(X_val_np)[:, 1]
        combined_score, pr_auc, recall = compute_score_with_recall(y_val_np, y_val_pred_proba)
        
        y_train_pred_proba = model.predict_proba(X_train_np)[:, 1]
        _, train_pr_auc, train_recall = compute_score_with_recall(y_train_np, y_train_pred_proba)
        
        print(f"  [{call_count[0]}/{max_evals}] subsample={subsample:.2f}, colsample={colsample_bytree:.2f}, "
              f"lambda={reg_lambda:.2f}, alpha={reg_alpha:.2f} "
              f"| Val PR AUC: {pr_auc:.4f}, Recall: {recall:.4f}")
        
        if combined_score < best_result['loss']:
            best_result['loss'] = combined_score
            best_result['params'] = {
                'subsample': subsample,
                'colsample_bytree': colsample_bytree,
                'reg_lambda': reg_lambda,
                'reg_alpha': reg_alpha,
            }
            best_result['val_pr_auc'] = pr_auc
            best_result['val_recall'] = recall
            best_result['train_pr_auc'] = train_pr_auc
            best_result['train_recall'] = train_recall
        
        return combined_score
    
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
    print(f"  Validation PR AUC: {best_result['val_pr_auc']:.4f}, Recall: {best_result['val_recall']:.4f}")
    print(f"  Train PR AUC: {best_result['train_pr_auc']:.4f}, Recall: {best_result['train_recall']:.4f}")
    
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
    search_space = [
        space.Real(0.01, 0.3, prior='log-uniform', name='learning_rate'),
        space.Integer(100, 500, name='n_estimators'),
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
        
        X_train_np = X_train.to_numpy()
        X_val_np = X_val.to_numpy()
        y_train_np = y_train.to_numpy()
        y_val_np = y_val.to_numpy()
        
        model = xgb.XGBClassifier(
            **model_params,
            tree_method='hist',
            eval_metric='aucpr',
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=20,
            verbose=0,
        )
        
        model.fit(
            X_train_np, y_train_np,
            eval_set=[(X_val_np, y_val_np)],
            verbose=False
        )
        
        # Evaluate
        y_val_pred_proba = model.predict_proba(X_val_np)[:, 1]
        combined_score, pr_auc, recall = compute_score_with_recall(y_val_np, y_val_pred_proba)
        
        y_train_pred_proba = model.predict_proba(X_train_np)[:, 1]
        _, train_pr_auc, train_recall = compute_score_with_recall(y_train_np, y_train_pred_proba)
        
        print(f"  [{call_count[0]}/{max_evals}] learning_rate={learning_rate:.4f}, n_estimators={n_estimators} "
              f"| Val PR AUC: {pr_auc:.4f}, Recall: {recall:.4f} | Stopped at: {model.best_iteration}")
        
        if combined_score < best_result['loss']:
            best_result['loss'] = combined_score
            best_result['params'] = {
                'learning_rate': learning_rate,
                'n_estimators': n_estimators,
            }
            best_result['val_pr_auc'] = pr_auc
            best_result['val_recall'] = recall
            best_result['train_pr_auc'] = train_pr_auc
            best_result['train_recall'] = train_recall
            best_result['best_iteration'] = model.best_iteration
        
        return combined_score
    
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
    print(f"  Validation PR AUC: {best_result['val_pr_auc']:.4f}, Recall: {best_result['val_recall']:.4f}")
    print(f"  Train PR AUC: {best_result['train_pr_auc']:.4f}, Recall: {best_result['train_recall']:.4f}")
    print(f"  Best iteration (early stop): {best_result['best_iteration']}")
    
    return best_params

def train_final_model(best_params):
    """Train final model with best parameters and evaluate on test set."""
    print("\n" + "="*60)
    print("FINAL MODEL: Training with Best Parameters")
    print("="*60)
    
    # Build final model
    model_params = baseline_config.copy()
    model_params.update(best_params)
    
    X_train_np = X_train.to_numpy()
    X_val_np = X_val.to_numpy()
    X_test_np = X_test.to_numpy()
    y_train_np = y_train.to_numpy()
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
    
    model.fit(
        X_train_np, y_train_np,
        eval_set=[(X_val_np, y_val_np)],
        verbose=False
    )
    
    # Evaluate on all splits
    results = {}
    
    for split_name, X, y, X_np, y_np in [
        ('train', X_train, y_train, X_train_np, y_train_np),
        ('val', X_val, y_val, X_val_np, y_val_np),
        ('test', X_test, y_test, X_test_np, y_test_np),
    ]:
        y_pred_proba = model.predict_proba(X_np)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        pr_auc = average_precision_score(y_np, y_pred_proba)
        recall = recall_score(y_np, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_np, y_pred).ravel()
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        results[split_name] = {
            'PR_AUC': pr_auc,
            'Recall': recall,
            'False_Negative_Rate': fnr,
            'TP': int(tp),
            'FP': int(fp),
            'FN': int(fn),
            'TN': int(tn),
        }
        
        print(f"\n{split_name.upper()}:")
        print(f"  PR AUC: {pr_auc:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  False Negative Rate: {fnr:.4f}")
        print(f"  Confusion Matrix - TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    
    return model, results

def main():
    """Main orchestrator for 3-phase hyperparameter tuning."""
    start_time = time.time()
    
    print("\n" + "#"*60)
    print("# XGBoost Hyperparameter Tuning: 3-Phase Bayesian Optimization")
    print("# Objective: Maximize PR AUC + Recall (Minimize False Negatives)")
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
    final_model, results = train_final_model(best_params)
    
    # Save results
    output_dir = Path('tuning_results')
    output_dir.mkdir(exist_ok=True)
    
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
    print("#"*60)
    
    return final_model, best_params, results

if __name__ == '__main__':
    final_model, best_params, results = main()
