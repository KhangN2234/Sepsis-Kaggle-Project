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

import pandas as pd
import numpy as np
from pathlib import Path
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
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# UPDATE THIS THRESHOLD BASED ON find_threshold.py RESULTS
# ============================================================================
final_threshold = 0.5  # Change this to your chosen threshold (e.g., 0.25, 0.30, etc.)
# ============================================================================


def evaluate_test_set(model_path, test_features_path, test_labels_path, threshold):
    """Load model, test data, and compute final metrics."""
    
    # Load model
    print(f"\nLoading tuned model from {model_path}...")
    model = joblib_load(model_path)
    
    # Load test data
    print(f"Loading test features from {test_features_path}...")
    X_test = pd.read_csv(test_features_path)
    
    print(f"Loading test labels from {test_labels_path}...")
    y_test = pd.read_csv(test_labels_path).iloc[:, 0].astype(int).values
    
    print(f"Test set shape: X {X_test.shape}, y {y_test.shape}")
    print(f"Test positive class prevalence: {y_test.mean():.4f} ({y_test.sum()} cases)\n")
    
    # Generate predictions
    print(f"Generating predictions...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
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
    print(f"\nResults saved to {results_path}")
    
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
    
    # Attempt to find test data paths
    # These are inferred from the typical project structure
    test_features_candidates = [
        Path('artifacts') / 'test_features.csv',
        Path('artifacts') / 'X_test.csv',
    ]
    test_labels_candidates = [
        Path('artifacts') / 'test_labels.csv',
        Path('artifacts') / 'y_test.csv',
    ]
    
    test_features_path = None
    for path in test_features_candidates:
        if path.exists():
            test_features_path = path
            break
    
    test_labels_path = None
    for path in test_labels_candidates:
        if path.exists():
            test_labels_path = path
            break
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please run XGBoost/xgboost_tuning.py first.")
        return
    
    if test_features_path is None or test_labels_path is None:
        print("\nWarning: Could not find test data files in artifacts/")
        print("Please provide test data paths in the code.")
        print("\nTo fix this, update test_features_path and test_labels_path in this script.")
        print("Or place test data at:")
        print("  - artifacts/test_features.csv")
        print("  - artifacts/test_labels.csv")
        return
    
    # Run evaluation
    evaluate_test_set(model_path, test_features_path, test_labels_path, final_threshold)


if __name__ == '__main__':
    main()
