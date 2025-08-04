# run_preprocessing.py

import os
import pandas as pd
from services.preprocess_data import preprocess_stock_csv

RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)

print("🔄 Starting preprocessing...")

if not os.path.exists(RAW_DIR):
    print(f"❌ Raw data folder not found: {RAW_DIR}")
    exit()

csv_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".csv")]

if not csv_files:
    print(f"⚠️ No CSV files found in {RAW_DIR}")
    exit()

for file in csv_files:
    try:
        file_path = os.path.join(RAW_DIR, file)
        df = preprocess_stock_csv(file_path)

        if not df.empty:
            ticker = file.split("_")[0]
            output_path = os.path.join(PROCESSED_DIR, f"{ticker}.parquet")
            df.to_parquet(output_path)
            print(f"📦 Saved processed {ticker} to {output_path}")
        else:
            print(f"⚠️ Skipped {file} (empty after preprocessing)")
    except Exception as e:
        print(f"❌ Error processing {file}: {e}")

print("✅ All preprocessing done.")
