import pandas as pd
import data_prep
from missing_values import prepare_for_xgboost

# Combine all csv file into one big one. Uncomment if running for the first time
#data_prep.main()

# Load the train feature table
train_path = "artifacts/train_features.csv"
train_df = pd.read_csv(train_path)

# Handle missing values. Uncomment if running for the first time.
# Note: Certained missing values are turned into actual features because it makes
#       the model learns split directions better according to ChatGPT 
#       so that's why we have so many columns.

#train_df = prepare_for_xgboost(train_df, label_column="SepsisLabel", add_indicators=True, add_summary=True)


print("First 5 instances of train_features.csv:")
print(train_df)
print(f"\nShape: {train_df.shape}")
print(f"\nColumns: {train_df.columns.tolist()}")