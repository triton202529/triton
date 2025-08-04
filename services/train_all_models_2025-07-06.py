import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime

print("🧠 Training all models...")

data_dir = "data"
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

for filename in os.listdir(data_dir):
    if filename.endswith(".csv"):
        file_path = os.path.join(data_dir, filename)
        print(f"\n📄 Processing {filename}")

        try:
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)

            # 🔍 Try to detect correct close column
            close_column = None
            for col in df.columns:
                if col.strip().lower() in ['close', 'adj close']:
                    close_column = col
                    break

            if not close_column:
                print(f"⚠️ Skipping {filename} — 'close' column missing")
                continue

            # Normalize column and ensure it's numeric
            df = df[[close_column]].copy()
            df.rename(columns={close_column: 'close'}, inplace=True)
            df['close'] = pd.to_numeric(df['close'], errors='coerce')

            # Drop rows with non-numeric close values
            df.dropna(inplace=True)

            # Compute return and target
            df['return'] = df['close'].pct_change()
            df['target'] = df['return'].shift(-1)
            df.dropna(inplace=True)

            X = df[['return']]
            y = df['target']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

            model = LinearRegression()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)

            result_file = os.path.join(model_dir, filename.replace(".csv", "_mse.txt"))
            with open(result_file, 'w') as f:
                f.write(f"Mean Squared Error: {mse:.6f}\n")
                f.write(f"Trained on: {filename}\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            print(f"✅ Trained and saved MSE for {filename}")

        except Exception as e:
            print(f"❌ Error training model for {filename}: {e}")
