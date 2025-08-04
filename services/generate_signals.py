import os
import pandas as pd
import glob
from datetime import datetime, timedelta

print("⚙️ Generating signals...")

# Define input/output
input_dir = "data/predictions"
output_file = os.path.join(input_dir, "signals.csv")
signal_data = []

# Signal thresholds
BUY_THRESHOLD = 1.002
SELL_THRESHOLD = 0.998

# Loop through all prediction files
for file in glob.glob(f"{input_dir}/*_predictions.csv"):
    try:
        df = pd.read_csv(file)

        if not {"close", "predicted_close"}.issubset(df.columns):
            print(f"⚠️ Skipping file (missing required columns): {file}")
            continue

        # Add synthetic dates (replace with actual dates when available)
        num_rows = len(df)
        today = datetime.today()
        df["date"] = [today - timedelta(days=num_rows - i) for i in range(num_rows)]

        # Extract ticker from filename
        ticker = os.path.basename(file).split("_")[0].upper()

        # Define signal logic
        def determine_signal(row):
            if row["predicted_close"] > row["close"] * BUY_THRESHOLD:
                return "BUY"
            elif row["predicted_close"] < row["close"] * SELL_THRESHOLD:
                return "SELL"
            else:
                return "HOLD"

        # Add signals and metadata
        df["signal"] = df.apply(determine_signal, axis=1)
        df["ticker"] = ticker
        df["confidence_score"] = ((df["predicted_close"] - df["close"]) / df["close"]).round(4)

        signal_data.append(df[["date", "ticker", "close", "predicted_close", "signal", "confidence_score"]])

    except Exception as e:
        print(f"🔥 Error processing {file}: {e}")

# Combine all signals
if signal_data:
    all_signals = pd.concat(signal_data)
    all_signals.sort_values("date", inplace=True)
    all_signals.to_csv(output_file, index=False)

    print(f"\n✅ Signals saved to: {output_file}")
    print("\n📊 Signal Breakdown (total):")
    print(all_signals["signal"].value_counts())

    print("\n📈 Signal Breakdown by Ticker:")
    print(all_signals.groupby("ticker")["signal"].value_counts().unstack(fill_value=0))

else:
    print("🚫 No signals generated.")
