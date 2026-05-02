from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def coerce_infinities_to_nan(df: pd.DataFrame) -> pd.DataFrame:
	"""Replace positive/negative infinity with NaN.

	XGBoost can handle missing values natively, but it does not expect infinite values.
	This helper makes the table safe for model input without imputing values.
	"""
	return df.replace([np.inf, -np.inf], np.nan)


def add_missing_indicators(
	df: pd.DataFrame,
	columns: Iterable[str] | None = None,
	prefix: str = "missing_",
) -> pd.DataFrame:
	"""Add binary missingness flags for selected columns.

	A missingness indicator helps the model learn whether the absence of a value
	is itself informative, which is often true in clinical data.
	"""
	result = df.copy()
	selected_columns = list(columns) if columns is not None else list(result.columns)

	# Build indicator series first (no repeated DataFrame mutation)
	indicators: dict[str, pd.Series] = {}
	for column in selected_columns:
		if column in result.columns:
			indicators[f"{prefix}{column}"] = result[column].isna().astype("int8")

	# Concatenate all indicators at once for performance
	if indicators:
		ind_df = pd.DataFrame(indicators, index=result.index)
		result = pd.concat([result, ind_df], axis=1)

	return result


def add_missing_summary_features(
	df: pd.DataFrame,
	numeric_columns: Iterable[str] | None = None,
	prefix: str = "missing_",
) -> pd.DataFrame:
	"""Add summary features that count missing values per row.

	These features give the model a compact view of how sparse a patient's record
	is at a given hour, which can be a strong signal in ICU data.
	"""
	result = df.copy()
	columns_to_check = list(numeric_columns) if numeric_columns is not None else list(result.columns)
	columns_to_check = [column for column in columns_to_check if column in result.columns]

	result[f"{prefix}count"] = result[columns_to_check].isna().sum(axis=1).astype("int16")
	result[f"{prefix}fraction"] = (
		result[columns_to_check].isna().mean(axis=1).astype("float32")
		if columns_to_check
		else 0.0
	)
	return result


def prepare_for_xgboost(
	df: pd.DataFrame,
	label_column: str | None = None,
	add_indicators: bool = True,
	add_summary: bool = True,
) -> pd.DataFrame:
	"""Prepare a feature table for XGBoost without imputing missing values.

	This function is intentionally conservative: it keeps NaNs in place so XGBoost
	can route them natively. It only removes infinities, optionally adds missingness
	indicators, and adds row-level missingness summaries.
	"""
	result = coerce_infinities_to_nan(df)

	if label_column is not None and label_column in result.columns:
		feature_columns = [column for column in result.columns if column != label_column]
	else:
		feature_columns = list(result.columns)

	if add_indicators:
		result = add_missing_indicators(result, columns=feature_columns)

	if add_summary:
		result = add_missing_summary_features(result, numeric_columns=feature_columns)

	return result


def main() -> None:
	# Small sanity check when the module is run directly.
	example = pd.DataFrame(
		{
			"a": [1.0, np.nan, 3.0],
			"b": [np.inf, 2.0, -np.inf],
			"SepsisLabel": [0, 1, 0],
		}
	)
	prepared = prepare_for_xgboost(example, label_column="SepsisLabel")
	print(prepared)

if __name__ == "__main__":
	main()