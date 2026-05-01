from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


DATA_ROOT = Path("phems-hackathon-early-sepsis-prediction")
TRAIN_DIR = DATA_ROOT / "training_data"
TEST_DIR = DATA_ROOT / "testing_data"


def to_hour(series: pd.Series) -> pd.Series:
	return pd.to_datetime(series, errors="coerce").dt.floor("h")


def safe_numeric_columns(df: pd.DataFrame, exclude: Iterable[str]) -> list[str]:
	"""Identify all numeric columns in a DataFrame, excluding specified column names.
	"""
	excluded = set(exclude)
	return [
		c
		for c in df.columns
		if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
	]


def build_master_train() -> pd.DataFrame:
	"""Build the master training index from SepsisLabel_train.csv.
	
	This creates the backbone of the training dataset: one row per person per hour,
	with the target label (SepsisLabel). All other features will be joined onto this.
	
	Returns:
		A DataFrame with columns [person_id, measurement_datetime, SepsisLabel].
	"""
	y = pd.read_csv(TRAIN_DIR / "SepsisLabel_train.csv")
	y["measurement_datetime"] = to_hour(y["measurement_datetime"])
	return y[["person_id", "measurement_datetime", "SepsisLabel"]].copy()


def build_master_test() -> pd.DataFrame:
	"""Build the master test index from SepsisLabel_test.csv.
	
	This creates the backbone of the test dataset: one row per person per hour.
	Note: SepsisLabel is omitted (not provided in test set). Features will be joined
	onto this index, and predictions will be made for each row.
	
	Returns:
		A DataFrame with columns [person_id, measurement_datetime].
	"""
	x = pd.read_csv(TEST_DIR / "SepsisLabel_test.csv")
	x["measurement_datetime"] = to_hour(x["measurement_datetime"])
	return x[["person_id", "measurement_datetime"]].copy()


def aggregate_wide_measurements(df: pd.DataFrame) -> pd.DataFrame:
	"""Aggregate measurement tables (labs, vitals, observations) to hourly granularity.
	
	If multiple measurements exist for the same person in the same hour, this takes the mean.
	This handles cases where lab tests or vital signs are recorded multiple times per hour.
	
	Args:
		df: A measurement table with columns like [person_id, measurement_datetime, ...lab_values].
	
	Returns:
		A DataFrame grouped by person_id and measurement_datetime, with numeric columns averaged.
	"""
	keys = ["person_id", "measurement_datetime"]
	value_cols = [c for c in df.columns if c not in keys and c != "visit_occurrence_id"]
	# Convert all value columns to numeric, coercing errors to NaN
	for col in value_cols:
		df[col] = pd.to_numeric(df[col], errors="coerce")
	return (
		df.groupby(keys, as_index=False)[value_cols]
		.mean(numeric_only=True)
		.sort_values(keys)
	)


def load_measurement_table(path: Path) -> pd.DataFrame:
	"""Load and process a measurement CSV (labs, vitals, observations) into hourly features.
	
	This reads one of the wide measurement tables (measurement_lab, measurement_meds, etc.),
	floors timestamps to hourly, and aggregates multiple measurements per hour by taking means.
	
	Args:
		path: Path to the CSV file (e.g., measurement_lab_train.csv).
	
	Returns:
		A DataFrame with shape (n_person_hours, n_measurement_columns).
	"""
	df = pd.read_csv(path)
	df["measurement_datetime"] = to_hour(df["measurement_datetime"])
	return aggregate_wide_measurements(df)


def load_event_counts_table(
	path: Path,
	datetime_col: str,
	category_col: str,
	prefix: str,
) -> pd.DataFrame:
	"""Load and pivot an event-based table (devices, procedures) into hourly categorical counts.
	
	Event tables have one row per event (e.g., one row per device applied). This function
	transforms them into wide format: one row per person-hour, with columns for each
	category (e.g., 'Endotracheal tube', 'Urinary catheter'), and values as event counts.
	
	Args:
		path: Path to the CSV file (e.g., devices_train.csv).
		datetime_col: Name of the datetime column in the CSV (e.g., 'device_datetime_hourly').
		category_col: Name of the column to pivot by (e.g., 'device', 'procedure').
		prefix: Prefix to add to resulting column names (e.g., 'device_', 'procedure_').
	
	Returns:
		A DataFrame with shape (n_person_hours, n_categories+2) where columns are
		[person_id, measurement_datetime, prefix_category1, prefix_category2, ...].
	"""
	df = pd.read_csv(path)
	# Convert datetime column to hourly granularity
	df["measurement_datetime"] = to_hour(df[datetime_col])
	df = df[["person_id", "measurement_datetime", category_col]].copy()
	# Convert category values to strings and handle missing values
	df[category_col] = df[category_col].astype("string").fillna("UNKNOWN")

	# Pivot: reshape from long (one row per event) to wide (one row per person-hour)
	counts = (
		df.assign(event_count=1)  # Add a counter column
		.pivot_table(
			index=["person_id", "measurement_datetime"],  # Rows
			columns=category_col,  # Columns become each device/procedure type
			values="event_count",  # Values to aggregate
			aggfunc="sum",  # Sum multiple events in the same hour
			fill_value=0,  # If no events, fill with 0 (not NaN)
		)
		.reset_index()
	)

	# Ensure all column names are strings (pivot can create non-string indices)
	counts.columns = [
		col if isinstance(col, str) else str(col) for col in counts.columns
	]
	# Add prefix to distinguish feature sources (e.g., device_Endotracheal tube)
	renamed = {
		c: f"{prefix}_{c}" for c in counts.columns if c not in {"person_id", "measurement_datetime"}
	}
	return counts.rename(columns=renamed)


def load_drug_table(path: Path) -> pd.DataFrame:
	"""Load and pivot drug exposure data into hourly counts by drug and route.
	
	This function transforms drug administration events into two pivot tables:
	1. Counts of each drug administered per person-hour (e.g., 'drug_vancomycin')
	2. Counts of each route used per person-hour (e.g., 'route_Intravenous')
	
	Both pivots are then merged to create one feature table per person-hour.
	
	Args:
		path: Path to the drug exposure CSV (e.g., drugsexposure_train.csv).
	
	Returns:
		A DataFrame with shape (n_person_hours, n_drugs+n_routes+2) containing
		[person_id, measurement_datetime, drug_amikacin, drug_vancomycin, ...,
		route_Intravenous, route_Oral, ...].
	"""
	df = pd.read_csv(path)
	# Convert drug administration time to hourly granularity
	df["measurement_datetime"] = to_hour(df["drug_datetime_hourly"])
	df = df[
		[
			"person_id",
			"measurement_datetime",
			"drug_concept_id",
			"route_concept_id",
		]
	].copy()
	# Convert drug and route to strings, fill missing with UNKNOWN
	df["drug_concept_id"] = df["drug_concept_id"].astype("string").fillna("UNKNOWN")
	df["route_concept_id"] = df["route_concept_id"].astype("string").fillna("UNKNOWN")

	# Pivot drugs: rows=person-hours, columns=drug types, values=count of administrations
	drug_counts = (
		df.assign(event_count=1)
		.pivot_table(
			index=["person_id", "measurement_datetime"],
			columns="drug_concept_id",
			values="event_count",
			aggfunc="sum",
			fill_value=0,
		)
		.reset_index()
	)
	# Pivot routes: rows=person-hours, columns=route types, values=count of administrations
	route_counts = (
		df.assign(event_count=1)
		.pivot_table(
			index=["person_id", "measurement_datetime"],
			columns="route_concept_id",
			values="event_count",
			aggfunc="sum",
			fill_value=0,
		)
		.reset_index()
	)

	# Ensure column names are strings
	drug_counts.columns = [
		col if isinstance(col, str) else str(col) for col in drug_counts.columns
	]
	route_counts.columns = [
		col if isinstance(col, str) else str(col) for col in route_counts.columns
	]

	# Add prefixes to distinguish drug and route features
	drug_counts = drug_counts.rename(
		columns={
			c: f"drug_{c}"
			for c in drug_counts.columns
			if c not in {"person_id", "measurement_datetime"}
		}
	)
	route_counts = route_counts.rename(
		columns={
			c: f"route_{c}"
			for c in route_counts.columns
			if c not in {"person_id", "measurement_datetime"}
		}
	)

	# Merge drug and route tables on person-hour (outer join to keep all hours from both)
	return drug_counts.merge(
		route_counts,
		on=["person_id", "measurement_datetime"],
		how="outer",
	)


def load_observation_table(path: Path, top_k: int = 50) -> pd.DataFrame:
	"""Load and pivot observation data into hourly counts, keeping only top-K observation types.
	
	Observation tables can have many unique observation types. To keep the feature space
	manageable, this function filters to only the top_k most frequent observations before
	pivoting into hourly counts.
	
	Args:
		path: Path to the observation CSV (e.g., observation_train.csv).
		top_k: Maximum number of unique observation types to retain (default 50).
	
	Returns:
		A DataFrame with shape (n_person_hours, top_k+2) containing
		[person_id, measurement_datetime, obs_type1, obs_type2, ...].
	"""
	df = pd.read_csv(path)
	# Convert observation datetime to hourly granularity
	df["measurement_datetime"] = to_hour(df["observation_datetime"])
	# Convert observation names to strings and handle missing
	df["observation_concept_name"] = (
		df["observation_concept_name"].astype("string").fillna("UNKNOWN")
	)

	# Keep only the top-k most frequent observations (reduces dimensionality)
	top_names = (
		df["observation_concept_name"].value_counts().head(top_k).index.tolist()
	)
	df = df[df["observation_concept_name"].isin(top_names)].copy()

	# Pivot: rows=person-hours, columns=observation types, values=count per hour
	obs_counts = (
		df.assign(event_count=1)
		.pivot_table(
			index=["person_id", "measurement_datetime"],
			columns="observation_concept_name",
			values="event_count",
			aggfunc="sum",
			fill_value=0,
		)
		.reset_index()
	)
	# Ensure column names are strings
	obs_counts.columns = [
		col if isinstance(col, str) else str(col) for col in obs_counts.columns
	]
	# Add 'obs_' prefix to observation feature names
	return obs_counts.rename(
		columns={
			c: f"obs_{c}"
			for c in obs_counts.columns
			if c not in {"person_id", "measurement_datetime"}
		}
	)


def load_demographics(path: Path) -> pd.DataFrame:
	"""Load and aggregate demographic information to one row per patient.
	
	Demographics (age, gender) are static per patient, so this aggregates them
	across all episodes/visits by taking the median age and most common gender.
	Result is one row per person_id to be merged onto the hourly feature table.
	
	Args:
		path: Path to the demographics CSV (e.g., person_demographics_episode_train.csv).
	
	Returns:
		A DataFrame with shape (n_unique_persons, 3) containing
		[person_id, age_in_months, gender].
	"""
	df = pd.read_csv(path)
	# Convert date columns if present (for potential future use)
	if "birth_datetime" in df.columns:
		df["birth_datetime"] = pd.to_datetime(df["birth_datetime"], errors="coerce")
	if "visit_start_date" in df.columns:
		df["visit_start_date"] = pd.to_datetime(df["visit_start_date"], errors="coerce")

	# Select only demographic columns of interest
	keep_cols = [c for c in ["person_id", "age_in_months", "gender"] if c in df.columns]
	d = df[keep_cols].copy()

	# Ensure age is numeric
	if "age_in_months" in d.columns:
		d["age_in_months"] = pd.to_numeric(d["age_in_months"], errors="coerce")

	# Aggregate demographics by person: median age, most common gender
	if "gender" in d.columns:
		d["gender"] = d["gender"].astype("string").fillna("UNKNOWN")
		d = (
			d.sort_values(["person_id"]) 
			.groupby("person_id", as_index=False)
			.agg(
				{
					"age_in_months": "median" if "age_in_months" in d.columns else "first",
					"gender": lambda s: s.mode().iat[0] if not s.mode().empty else "UNKNOWN",
				}
			)
		)
	else:
		# If gender not present, just take median of numeric columns
		d = d.groupby("person_id", as_index=False).agg("median")

	return d


def merge_feature_tables(master: pd.DataFrame, feature_tables: list[pd.DataFrame]) -> pd.DataFrame:
	"""Left-join multiple feature tables onto the master index.
	
	This is the core feature assembly step. Each feature table is merged onto the
	master timeline (person_id + measurement_datetime) using left join, ensuring
	we keep exactly the prediction rows we need without data leakage.
	
	Args:
		master: The master DataFrame (train or test) with [person_id, measurement_datetime, ...].
		feature_tables: List of feature DataFrames to join, each with [person_id, measurement_datetime, feature_cols].
	
	Returns:
		A single DataFrame with all features merged onto the master index.
	"""
	out = master.copy()
	for feat in feature_tables:
		# Left join preserves all rows from master and adds columns from feat
		out = out.merge(feat, on=["person_id", "measurement_datetime"], how="left")
	return out


def build_features(split: str) -> pd.DataFrame:
	"""Build the complete feature table for train or test split.
	
	This is the main feature engineering orchestration function. It:
	1. Loads the appropriate master index (train or test).
	2. Loads and processes all measurement tables (labs, vitals, observations).
	3. Loads and processes all event-based tables (devices, procedures, drugs, observations).
	4. Loads and aggregates demographics.
	5. Joins all features onto the master index using left joins.
	6. One-hot encodes categorical columns (gender).
	
	Args:
		split: Either 'train' or 'test' to specify which data split to process.
	
	Returns:
		A complete feature DataFrame ready for model training or prediction.
	"""
	if split not in {"train", "test"}:
		raise ValueError("split must be 'train' or 'test'")

	# Select correct directory and master table based on split
	base_dir = TRAIN_DIR if split == "train" else TEST_DIR
	master = build_master_train() if split == "train" else build_master_test()

	# Load and process measurement tables (wide numeric features)
	measurement_tables = [
		load_measurement_table(base_dir / f"measurement_lab_{split}.csv"),
		load_measurement_table(base_dir / f"measurement_meds_{split}.csv"),
		load_measurement_table(base_dir / f"measurement_observation_{split}.csv"),
	]

	# Load and process event-based tables (count features per category)
	event_tables = [
		load_event_counts_table(
			base_dir / f"devices_{split}.csv",
			datetime_col="device_datetime_hourly",
			category_col="device",
			prefix="device",
		),
		load_event_counts_table(
			base_dir / f"proceduresoccurrences_{split}.csv",
			datetime_col="procedure_datetime_hourly",
			category_col="procedure",
			prefix="procedure",
		),
		load_drug_table(base_dir / f"drugsexposure_{split}.csv"),
		load_observation_table(base_dir / f"observation_{split}.csv", top_k=50),
	]

	# Load demographic information (one row per patient)
	demographics = load_demographics(base_dir / f"person_demographics_episode_{split}.csv")

	# Join all feature tables onto the master index
	features = merge_feature_tables(master, measurement_tables + event_tables)
	features = features.merge(demographics, on="person_id", how="left")

	# One-hot encode gender categorical variable
	if "gender" in features.columns:
		features = pd.get_dummies(features, columns=["gender"], prefix="gender")

	# Sort by person and time, reset index for clean output
	return features.sort_values(["person_id", "measurement_datetime"]).reset_index(drop=True)


def main() -> None:
	# Build feature tables
	train_features = build_features("train")
	test_features = build_features("test")

	# Create output directory if it doesn't exist
	output_dir = Path("artifacts")
	output_dir.mkdir(parents=True, exist_ok=True)

	# Save to CSV
	train_path = output_dir / "train_features.csv"
	test_path = output_dir / "test_features.csv"
	train_features.to_csv(train_path, index=False)
	test_features.to_csv(test_path, index=False)

	# Print summary
	print("Built feature tables:")
	print(f"- Train shape: {train_features.shape}")
	print(f"- Test shape:  {test_features.shape}")
	print(f"- Saved: {train_path}")
	print(f"- Saved: {test_path}")


if __name__ == "__main__":
	main()
