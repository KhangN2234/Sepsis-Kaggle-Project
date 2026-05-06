"""
Threshold Selection Utility for Sepsis Prediction Model

After hyperparameter tuning, this script sweeps decision thresholds on the validation set
and identifies the optimal threshold based on clinical criteria.

Threshold selection is a clinical decision that should be made separately from tuning,
as it depends on the acceptable trade-off between sensitivity (recall) and specificity
for the deployment context.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    f1_score,
    recall_score,
    precision_score,
)


def compute_metrics_at_threshold(y_true, y_pred_proba, threshold):
    """Compute all relevant metrics for a given threshold."""
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0  # False Negative Rate (critical for sepsis!)
    fpr = fp / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(y_true, y_pred)
    
    return {
        'threshold': threshold,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': ppv,
        'npv': npv,
        'fnr': fnr,
        'fpr': fpr,
        'f1': f1,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn),
    }


def main():
    """Load validation predictions and sweep thresholds."""
    
    # Load validation predictions
    val_pred_path = Path('tuning_results') / 'val_predictions.csv'
    if not val_pred_path.exists():
        print(f"Error: {val_pred_path} not found.")
        print("Please run XGBoost/xgboost_tuning.py first to generate validation predictions.")
        return
    
    df = pd.read_csv(val_pred_path)
    y_true = df['true_label'].values
    y_pred_proba = df['pred_proba'].values
    
    print(f"\nLoaded {len(y_true)} validation samples")
    print(f"Positive class prevalence: {y_true.mean():.4f} ({y_true.sum()} cases)")
    
    # Sweep thresholds
    thresholds = np.arange(0.01, 1.0, 0.01)
    metrics_list = []
    
    for threshold in thresholds:
        metrics = compute_metrics_at_threshold(y_true, y_pred_proba, threshold)
        metrics_list.append(metrics)
    
    metrics_df = pd.DataFrame(metrics_list)
    
    # Print summary table
    print("\n" + "="*100)
    print("THRESHOLD SWEEP RESULTS (Validation Set)")
    print("="*100)
    print(f"\n{'Threshold':<12} {'Sensitivity':<15} {'Specificity':<15} {'FNR':<12} {'FPR':<12} {'F1':<10}")
    print("-" * 100)
    
    for _, row in metrics_df.iterrows():
        if row['threshold'] in [0.1, 0.25, 0.5, 0.75, 0.9]:  # Highlight common thresholds
            print(
                f"{row['threshold']:<12.2f} {row['sensitivity']:<15.4f} {row['specificity']:<15.4f} "
                f"{row['fnr']:<12.4f} {row['fpr']:<12.4f} {row['f1']:<10.4f}"
            )
    
    # Identify key thresholds
    print("\n" + "="*100)
    print("RECOMMENDED THRESHOLDS BY CLINICAL CRITERIA")
    print("="*100)
    
    # Max F1 (general balance)
    max_f1_idx = metrics_df['f1'].idxmax()
    max_f1_row = metrics_df.loc[max_f1_idx]
    print(f"\n1. Max F1 Score (balanced sensitivity/specificity):")
    print(f"   Threshold: {max_f1_row['threshold']:.2f}")
    print(f"   Sensitivity: {max_f1_row['sensitivity']:.4f}, Specificity: {max_f1_row['specificity']:.4f}")
    print(f"   FNR: {max_f1_row['fnr']:.4f} (False Negatives: {int(max_f1_row['fn'])})")
    
    # Min FNR (minimize false negatives - critical for sepsis!)
    min_fnr_idx = metrics_df['fnr'].idxmin()
    min_fnr_row = metrics_df.loc[min_fnr_idx]
    print(f"\n2. Min False Negative Rate (minimize missed sepsis cases):")
    print(f"   Threshold: {min_fnr_row['threshold']:.2f}")
    print(f"   Sensitivity: {min_fnr_row['sensitivity']:.4f}, Specificity: {min_fnr_row['specificity']:.4f}")
    print(f"   FNR: {min_fnr_row['fnr']:.4f} (False Negatives: {int(min_fnr_row['fn'])})")
    
    # High sensitivity (>= 90%)
    high_sens = metrics_df[metrics_df['sensitivity'] >= 0.90].iloc[0]
    print(f"\n3. High Sensitivity (>= 90% - catch most cases):")
    print(f"   Threshold: {high_sens['threshold']:.2f}")
    print(f"   Sensitivity: {high_sens['sensitivity']:.4f}, Specificity: {high_sens['specificity']:.4f}")
    print(f"   FNR: {high_sens['fnr']:.4f} (False Negatives: {int(high_sens['fn'])})")
    
    # Save results
    metrics_df.to_csv(Path('tuning_results') / 'threshold_sweep_results.csv', index=False)
    print(f"\nDetailed results saved to tuning_results/threshold_sweep_results.csv")
    
    # Plot precision-recall curve
    try:
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Precision-Recall curve
        axes[0].plot(recall, precision, marker='o', linewidth=2, markersize=4)
        axes[0].axvline(max_f1_row['sensitivity'], color='r', linestyle='--', label=f"Max F1 (t={max_f1_row['threshold']:.2f})")
        axes[0].axvline(min_fnr_row['sensitivity'], color='g', linestyle='--', label=f"Min FNR (t={min_fnr_row['threshold']:.2f})")
        axes[0].set_xlabel('Recall (Sensitivity)')
        axes[0].set_ylabel('Precision')
        axes[0].set_title('Precision-Recall Curve')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Plot 2: Sensitivity, Specificity, FNR by threshold
        axes[1].plot(metrics_df['threshold'], metrics_df['sensitivity'], label='Sensitivity', marker='o', markersize=3)
        axes[1].plot(metrics_df['threshold'], metrics_df['specificity'], label='Specificity', marker='s', markersize=3)
        axes[1].plot(metrics_df['threshold'], metrics_df['fnr'], label='FNR (False Neg Rate)', marker='^', markersize=3, color='red')
        axes[1].axvline(max_f1_row['threshold'], color='r', linestyle='--', alpha=0.5)
        axes[1].axvline(min_fnr_row['threshold'], color='g', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('Decision Threshold')
        axes[1].set_ylabel('Metric Value')
        axes[1].set_title('Metrics vs Decision Threshold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plot_path = Path('tuning_results') / 'threshold_analysis.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {plot_path}")
        plt.close()
    except Exception as e:
        print(f"Warning: Could not create plot ({e})")
    
    print("\n" + "="*100)
    print("NEXT STEPS")
    print("="*100)
    print("""
1. Review the threshold sweep results and plots.
2. Consult with clinical team on acceptable trade-offs:
   - Lower threshold: higher sensitivity (catch more sepsis), but more false alarms
   - Higher threshold: higher specificity (fewer false alarms), but miss some cases
3. Update final_threshold in evaluate_on_test.py with your chosen threshold
4. Run evaluate_on_test.py to compute final test set metrics using the chosen threshold
    """)


if __name__ == '__main__':
    main()
