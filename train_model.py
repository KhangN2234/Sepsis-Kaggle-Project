import pandas as pd

# Load first 5 instances from train_features.csv
train_data = pd.read_csv("artifacts/train_features.csv", nrows=5)

print("First 5 instances of train_features.csv:")
print(train_data)
print(f"\nShape: {train_data.shape}")
print(f"\nColumns: {train_data.columns.tolist()}")
