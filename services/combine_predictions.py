import pandas as pd
import os
import glob

# Set paths
predictions_dir = "data/predictions"
output_path = "data/results/signals.csv"

# Create results directory if it doesn't exist
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Find all prediction CSV files
prediction_files = glob.glob(os.path.join(predictions_dir, "*_predictions.csv"))

# Collect all signal data
all_signals = []

print(f"🔍 Found {len(prediction_files)} prediction files...")

for file_path in prediction_files:
    try:
        # Extract ticker from filename (e.g., AAPL from aapl_predictions.csv)
        filename = os.path.basename(file_path)
        ticker = filename.split("_")[0].upper()

        df = pd.read_csv(file_path)

        # Ensure required columns exist
        if not {'date', 'close', 'predicted_close'}.issubset(df.columns):
            print(f"⚠️ Skipping {filename} — missing required columns.")
            continue

        # Drop rows with missing values
        df = df.dropna(subset=['date', 'close', 'predicted_close'])

        # Generate BUY/HOLD/SELL signal
        df["signal"] = df.apply(
            lambda row: "BUY" if row["predicted_close"] > row["close"] else "SELL",
            axis=1
        )

        df["ticker"] = ticker

        # Reorder columns
        df = df[["date", "ticker", "close", "predicted_close", "signal"]]

        all_signals.append(df)
        print(f"✅ Processed: {ticker}")

    except Exception as e:
        print(f"❌ Error processing {file_path}: {str(e)}")

# Combine all
if all_signals:
    combined_df = pd.concat(all_signals)
    combined_df.to_csv(output_path, index=False)
    print(f"\n✅ Combined signals saved to: {output_path}")
else:
    print("🚫 No valid prediction files found.")
