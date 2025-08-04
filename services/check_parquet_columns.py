import pandas as pd

df = pd.read_parquet("data/processed/stock_data.parquet")
print("Columns in the Parquet file:")
print(df.columns)
print("\nFirst few rows:")
print(df.head())
