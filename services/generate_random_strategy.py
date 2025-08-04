import pandas as pd
import numpy as np
import os
import random

SIGNAL_DIR = "data/predictions"
OUTPUT_FILE = os.path.join(SIGNAL_DIR, "random_signals.csv")

# Load real signals to match date and ticker structure
real_signals_path = os.path.join(SIGNAL_DIR, "signals.csv")
signals_df = pd.read_csv(real_signals_path)

# Generate random signals with same structure
np.random.seed(42)
possible_signals = ["BUY", "SELL", "HOLD"]
signals_df["signal"] = np.random.choice(possible_signals, size=len(signals_df))

# Save to new file
signals_df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Random signals saved to: {OUTPUT_FILE}")
