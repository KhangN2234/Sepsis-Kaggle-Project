"""
Generate submission CSV using the tuned model.

Outputs: `tuning_results/submission.csv` with columns:
  person_id_datetime,SepsisLabel

Usage:
  .\.venv\Scripts\python.exe submit.py
"""
import sys
from pathlib import Path

# Ensure project root imports work
project_root = Path(__file__).resolve().parents[0]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
from joblib import load as joblib_load
from missing_values import prepare_for_xgboost


def main():
    model_path = Path('tuning_results') / 'final_model.pkl'
    test_path = Path('artifacts') / 'test_features.csv'
    out_path = Path('tuning_results') / 'submission.csv'

    if not model_path.exists():
        print(f"Model not found at {model_path}. Run xgboost_tuning.py first.")
        return
    if not test_path.exists():
        print(f"Test features not found at {test_path}.")
        return

    print(f"Loading test features from {test_path}...")
    df = pd.read_csv(test_path)
    if 'person_id' not in df.columns or 'measurement_datetime' not in df.columns:
        print("Expected 'person_id' and 'measurement_datetime' columns in test features.")
        return

    ids = df['person_id'].astype(str) + '_' + df['measurement_datetime'].astype(str)

    # Drop metadata columns and preprocess
    X = df.drop(columns=['person_id', 'measurement_datetime'], errors='ignore')
    X_proc = prepare_for_xgboost(X, label_column=None, add_indicators=True, add_summary=True)

    # Ensure test features match training feature set used by the model.
    # Recreate training-processed columns by preprocessing the training artifact
    train_path = Path('artifacts') / 'train_features.csv'
    if train_path.exists():
        train_df = pd.read_csv(train_path)
        train_X = train_df.drop(columns=['SepsisLabel', 'person_id', 'measurement_datetime'], errors='ignore')
        train_X_proc = prepare_for_xgboost(train_X, label_column=None, add_indicators=True, add_summary=True)
        train_cols = list(train_X_proc.columns)
        # Align test to train columns: add missing cols as NaN, drop extras
        X_proc = X_proc.reindex(columns=train_cols)
    else:
        print(f"Warning: {train_path} not found — proceeding without column alignment.")

    print("Loading model...")
    model = joblib_load(model_path)

    print("Generating probabilities...")
    proba = model.predict_proba(X_proc.to_numpy())[:, 1]

    # Build submission DataFrame with same header as sample
    sub = pd.DataFrame({'person_id_datetime': ids, 'SepsisLabel': proba})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    print(f"Wrote submission to {out_path}")


if __name__ == '__main__':
    main()
