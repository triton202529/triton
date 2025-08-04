# services/signal_engine.py

import pandas as pd
import os

def generate_signals(prediction_file):
    try:
        if not os.path.exists(prediction_file):
            raise FileNotFoundError(f"❌ File not found: {prediction_file}")

        df = pd.read_csv(prediction_file)

        # Drop completely blank rows
        df.dropna(how='all', inplace=True)

        # Ensure required columns exist
        required_cols = {'ticker', 'prediction', 'actual_close'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Missing columns in prediction file. Required: {required_cols}")

        signals = []

        for _, row in df.iterrows():
            ticker = row['ticker']
            predicted = row['prediction']
            actual = row['actual_close']
            diff = predicted - actual

            # Basic rule for generating signals
            if diff > 2.0:
                signal = "BUY"
            elif diff < -2.0:
                signal = "SELL"
            else:
                signal = "HOLD"

            signals.append({
                "ticker": ticker,
                "actual_price": round(actual, 2),
                "predicted_price": round(predicted, 2),
                "difference": round(diff, 2),
                "signal": signal
            })

        df_signals = pd.DataFrame(signals)
        output_path = "../predictions/signals.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_signals.to_csv(output_path, index=False)

        print(f"✅ Signals saved to: {output_path}")
        return df_signals

    except Exception as e:
        print(f"❌ Error generating signals: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    print("🔁 Running signal engine...")
    signals = generate_signals("../predictions/latest_predictions.csv")
    print("\n📊 Trading Signals:\n")
    print(signals.to_string(index=False))
