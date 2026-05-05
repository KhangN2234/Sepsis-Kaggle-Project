from __future__ import annotations

import os
import sys

import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from xgboost import XGBClassifier
from pathlib import Path

# Ensure project root imports work when running this file via XGBoost/train_XGBoost.py
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

import data_prep
from missing_values import prepare_for_xgboost


# Combine all csv files into one big table. Uncomment if running for the first time.
# data_prep.main()


def load_training_data() -> tuple[pd.DataFrame, pd.Series]:
	"""Load the engineered training table and separate features from the label."""
	train_path = Path("artifacts") / "train_features.csv"
	train_df = pd.read_csv(train_path)

	labels = train_df["SepsisLabel"].astype(int)
	features = train_df.drop(columns=["SepsisLabel", "person_id", "measurement_datetime"])
	features = prepare_for_xgboost(features, label_column=None, add_indicators=True, add_summary=True)

	return train_df, features, labels


def split_data_person_aware(
	df: pd.DataFrame,
	features: pd.DataFrame,
	labels: pd.Series,
	test_size: float = 0.2,
	val_size: float = 0.2,
	random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
	"""Split data using person-level stratification (no data leakage).
	
	Ensures each person appears in only one split (train, val, or test).
	This matches the methodology used in xgboost_tuning.py.
	"""
	if 'person_id' not in df.columns:
		raise ValueError("DataFrame must contain 'person_id' column for person-aware splitting")
	
	# Person-level labels for stratification
	person_label = df.groupby('person_id')['SepsisLabel'].max()
	persons = person_label.index.to_numpy()
	person_labels = person_label.values
	
	# Stratified split: persons into train_val and test
	sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
	train_val_idx, test_idx = next(sss.split(persons, person_labels))
	train_val_persons = persons[train_val_idx]
	test_persons = persons[test_idx]
	
	# Further split train_val into train and val (person-level stratification)
	remaining = train_val_persons
	remaining_labels = person_label.loc[remaining].values
	sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size / (1.0 - test_size), random_state=random_state)
	train_idx, val_idx = next(sss2.split(remaining, remaining_labels))
	train_persons = remaining[train_idx]
	val_persons = remaining[val_idx]
	
	# Extract feature and label indices for each split
	train_indices = df[df['person_id'].isin(train_persons)].index
	val_indices = df[df['person_id'].isin(val_persons)].index
	test_indices = df[df['person_id'].isin(test_persons)].index
	
	X_train = features.loc[train_indices]
	X_val = features.loc[val_indices]
	X_test = features.loc[test_indices]
	
	y_train = labels.loc[train_indices]
	y_val = labels.loc[val_indices]
	y_test = labels.loc[test_indices]
	
	return X_train, X_val, X_test, y_train, y_val, y_test


def split_data(
	features: pd.DataFrame,
	labels: pd.Series,
	test_size: float = 0.2,
	val_size: float = 0.2,
	random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
	"""Split data into train, validation, and test sets using stratification.
	
	WARNING: This uses sample-level stratification and can cause data leakage
	(same person in train and test). Use split_data_person_aware() instead.
	"""
	X_train_val, X_test, y_train_val, y_test = train_test_split(
		features,
		labels,
		test_size=test_size,
		stratify=labels,
		random_state=random_state,
	)

	val_fraction_of_train_val = val_size / (1.0 - test_size)
	X_train, X_val, y_train, y_val = train_test_split(
		X_train_val,
		y_train_val,
		test_size=val_fraction_of_train_val,
		stratify=y_train_val,
		random_state=random_state,
	)

	return X_train, X_val, X_test, y_train, y_val, y_test


def fit_xgboost(
	X_train: pd.DataFrame,
	y_train: pd.Series,
	X_val: pd.DataFrame,
	y_val: pd.Series,
) -> XGBClassifier:
	"""Fit an XGBoost classifier and use validation data for early stopping."""
	positive_count = float(y_train.sum())
	negative_count = float(len(y_train) - y_train.sum())
	scale_pos_weight = negative_count / positive_count if positive_count > 0 else 1.0

	model = XGBClassifier()

	model.fit(
		X_train.to_numpy(),
		y_train,
		eval_set=[(X_val.to_numpy(), y_val)],
		verbose=False,
	)
	return model


def fit_tuned_xgboost(
	X_train: pd.DataFrame,
	y_train: pd.Series,
	X_val: pd.DataFrame,
	y_val: pd.Series,
) -> XGBClassifier:
	"""Fit a tuned XGBoost classifier with hyperparameters from Bayesian Optimization."""
	positive_count = float(y_train.sum())
	negative_count = float(len(y_train) - y_train.sum())
	scale_pos_weight = negative_count / positive_count if positive_count > 0 else 1.0

	# Tuned hyperparameters from Phase 1-3 optimization
	# Note: tree_method, eval_metric are fixed configuration choices (not tuned)
	# early_stopping_rounds is not used since final model trains on combined train+val
	model = XGBClassifier(
		n_estimators=1372,
		learning_rate=0.016256670876862687,
		max_depth=3,
		subsample=0.5,
		colsample_bytree=1,
		min_child_weight=10,
		reg_lambda=3.4968153791141336,
		reg_alpha=0,
		gamma=5.0,
		scale_pos_weight=scale_pos_weight,
		random_state=42,
		objective="binary:logistic",
		tree_method="hist",
		eval_metric="aucpr",
	)

	model.fit(
		X_train.to_numpy(),
		y_train,
		eval_set=[(X_val.to_numpy(), y_val)],
		verbose=False,
	)
	return model


def score_model(
	model: XGBClassifier,
	X: pd.DataFrame,
	y: pd.Series,
	split_name: str,
	threshold: float = 0.2,
) -> None:
	"""Print ranking metrics and thresholded classification metrics for a split.

	Accuracy is usually less informative for sepsis because the classes are imbalanced,
	but it is still reported here for completeness. Recall and precision are more useful
	for understanding false negatives and false positives at the chosen threshold.
	"""
	probas = model.predict_proba(X.to_numpy())[:, 1]
	predictions = (probas >= threshold).astype(int)

	roc_auc = roc_auc_score(y, probas)
	pr_auc = average_precision_score(y, probas)
	accuracy = accuracy_score(y, predictions)
	recall = recall_score(y, predictions, zero_division=0)
	precision = precision_score(y, predictions, zero_division=0)

	print(f"{split_name} ROC AUC:   {roc_auc:.4f}")
	print(f"{split_name} PR AUC:    {pr_auc:.4f}")
	print(f"{split_name} Accuracy:  {accuracy:.4f} @ threshold={threshold:.2f}")
	print(f"{split_name} Recall:    {recall:.4f} @ threshold={threshold:.2f}")
	print(f"{split_name} Precision: {precision:.4f} @ threshold={threshold:.2f}")


def main() -> None:
	df, features, labels = load_training_data()
	X_train, X_val, X_test, y_train, y_val, y_test = split_data_person_aware(df, features, labels)

	print("Split sizes (person-level stratification):")
	print(f"- Train: {X_train.shape}")
	print(f"- Val:   {X_val.shape}")
	print(f"- Test:  {X_test.shape}")

	# Baseline model
	# print("\n" + "="*60)
	# print("BASELINE MODEL (n_estimators=500, lr=0.05, max_depth=5)")
	# print("="*60)
	# baseline_model = fit_xgboost(X_train, y_train, X_val, y_val)
	# print("\nBaseline Scores:")
	# score_model(baseline_model, X_train, y_train, "Train")
	# score_model(baseline_model, X_val, y_val, "Validation")
	# score_model(baseline_model, X_test, y_test, "Test")

	# Tuned model
	print("\n" + "="*60)
	print("TUNED MODEL (Bayesian Optimization - Phase 1-3)")
	print("="*60)
	tuned_model = fit_tuned_xgboost(X_train, y_train, X_val, y_val)
	print("\nTuned Scores:")
	score_model(tuned_model, X_train, y_train, "Train")
	score_model(tuned_model, X_val, y_val, "Validation")
	score_model(tuned_model, X_test, y_test, "Test")


if __name__ == "__main__":
	main()