# fix_scores_file.py

import pandas as pd

input_path = "data/results/scored_stocks.csv"
df = pd.read_csv(input_path)

# Forcefully rename the first column to 'ticker' if needed
first_column = df.columns[0]
if first_column != "ticker":
    print(f"🔁 Renaming column '{first_column}' to 'ticker'")
    df.rename(columns={first_column: "ticker"}, inplace=True)

df.to_csv(input_path, index=False)
print("✅ ticker column restored and confirmed.")
