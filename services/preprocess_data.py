# services/preprocess_data.py

import pandas as pd

def preprocess_stock_csv(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)

        # Normalize column names
        df.columns = [col.lower().strip() for col in df.columns]

        # Ensure required columns exist
        required_cols = {'date', 'close'}
        if not required_cols.issubset(df.columns):
            print(f"⚠️ Skipping {file_path}: Missing required columns.")
            return pd.DataFrame()

        # Convert date and close to correct types
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['close'] = pd.to_numeric(df['close'], errors='coerce')

        # Drop rows with missing critical values
        df = df.dropna(subset=['date', 'close'])

        # ✅ Convert other numeric columns if they exist
        for col in ['open', 'high', 'low', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Feature engineering
        df['ma7'] = df['close'].rolling(window=7).mean()
        df['ma21'] = df['close'].rolling(window=21).mean()
        df['returns'] = df['close'].pct_change()

        df = df.dropna().reset_index(drop=True)
        return df

    except Exception as e:
        print(f"❌ Failed to preprocess {file_path}: {e}")
        return pd.DataFrame()
