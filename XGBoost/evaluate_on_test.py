"""
Final Test Set Evaluation with Chosen Threshold

This script loads the tuned model and applies a clinically-determined threshold
to evaluate performance on the held-out test set.

Workflow:
1. Run XGBoost/xgboost_tuning.py to tune hyperparameters
2. Run XGBoost/find_threshold.py to sweep thresholds and identify optimal one
3. Update final_threshold below with the chosen threshold
4. Run this script to get final test metrics
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from joblib import load as joblib_load
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    recall_score,
    precision_score,
    f1_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from missing_values import prepare_for_xgboost
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# UPDATE THIS THRESHOLD BASED ON find_threshold.py RESULTS
# ============================================================================
final_threshold = 0.2  # Change this to your chosen threshold (e.g., 0.25, 0.30, etc.)
# ============================================================================


def load_test_split(test_size=0.2, random_state=42):
    """Load and extract test set using the same stratification as xgboost_tuning.py.
    
    Returns: (X_test, y_test) after preprocessing
    """
    path = Path('artifacts') / 'train_features.csv'
    df = pd.read_csv(path)

    if 'person_id' not in df.columns or 'SepsisLabel' not in df.columns:
        raise RuntimeError('person_id and SepsisLabel columns required')

    # Person-level stratification (same as tuning script)
    person_label = df.groupby('person_id')['SepsisLabel'].max()
    persons = person_label.index.to_numpy()
    person_labels = person_label.values

    # Extract test split (using same random seed as tuning script)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(sss.split(persons, person_labels))
    test_persons = persons[test_idx]

    test_df = df[df['person_id'].isin(test_persons)].copy()

    # Prepare features (drop person_id, measurement_datetime, SepsisLabel)
    y_test = test_df['SepsisLabel'].astype(int)
    X_test = test_df.drop(columns=['SepsisLabel', 'measurement_datetime', 'person_id'])
    
    # Apply preprocessing (same as tuning script)
    X_test = prepare_for_xgboost(X_test, label_column=None, add_indicators=True, add_summary=True)

    return X_test, y_test


def evaluate_test_set(model_path, X_test, y_test, threshold):
    """Load model and compute final metrics on test set."""
    
    # Load model
    model = joblib_load(model_path)
    
    print(f"Test set shape: X {X_test.shape}, y {y_test.shape}")
    print(f"Test positive class prevalence: {y_test.mean():.4f} ({y_test.sum()} cases)\n")
    
    # Generate predictions
    y_pred_proba = model.predict_proba(X_test.to_numpy())[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Compute metrics
    print(f"\n{'='*70}")
    print(f"FINAL TEST SET EVALUATION (Threshold = {threshold:.2f})")
    print(f"{'='*70}")
    
    # Threshold-independent metrics
    pr_auc = average_precision_score(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\nThreshold-Independent Metrics:")
    print(f"  PR-AUC: {pr_auc:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    
    # Threshold-dependent metrics
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(y_test, y_pred)
    
    print(f"\nThreshold-Dependent Metrics (Threshold = {threshold:.2f}):")
    print(f"  Sensitivity (Recall): {sensitivity:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  Precision: {ppv:.4f}")
    print(f"  NPV: {npv:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  False Negative Rate (FNR): {fnr:.4f}")
    print(f"  False Positive Rate (FPR): {fpr:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {tn:>6}    False Positives: {fp:>6}")
    print(f"  False Negatives: {fn:>6}    True Positives:  {tp:>6}")
    
    print(f"\nDetailed Counts:")
    print(f"  Total positive cases: {tp + fn}")
    print(f"  Correctly identified (TP): {tp}")
    print(f"  Missed cases (FN): {fn}")
    print(f"  False alarms (FP): {fp}")
    
    # Save results
    results = {
        'threshold': threshold,
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': ppv,
        'npv': npv,
        'f1_score': f1,
        'fnr': fnr,
        'fpr': fpr,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn),
        'total_positive': int(tp + fn),
        'total_samples': len(y_test),
    }
    
    results_path = Path('tuning_results') / f'test_results_threshold_{threshold:.2f}.json'
    import json
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("CLINICAL INTERPRETATION")
    print(f"{'='*70}")
    print(f"""
Out of {tp + fn} actual sepsis cases in the test set:
  - {tp} cases were correctly identified (sensitivity: {sensitivity:.1%})
  - {fn} cases were missed (FNR: {fnr:.1%})

Out of {fp + tn} actual non-sepsis samples:
  - {tn} were correctly identified as negative (specificity: {specificity:.1%})
  - {fp} were incorrectly flagged as positive (FPR: {fpr:.1%})

For {tp + fp} predicted positive cases:
  - {tp} were correct (precision: {ppv:.1%})
  - {fp} were false alarms
    """)


def main():
    """Main entry point."""
    
    model_path = Path('tuning_results') / 'final_model.pkl'
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please run XGBoost/xgboost_tuning.py first.")
        return
    
    try:
        X_test, y_test = load_test_split()
    except Exception as e:
        print(f"Error loading test data: {e}")
        print("\nMake sure artifacts/train_features.csv exists and has 'person_id' and 'SepsisLabel' columns.")
        return
    
    # Run evaluation
    evaluate_test_set(model_path, X_test, y_test, final_threshold)


if __name__ == '__main__':
    main()
