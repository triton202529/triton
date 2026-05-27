from scripts.failed_ticker_utils import load_failed_tickers

failed_path = "data/logs/failed_tickers.txt"
unique_path = "data/logs/failed_tickers_unique.txt"

def clean_failed_tickers(source_path=failed_path, output_path=unique_path):
    unique_tickers = load_failed_tickers(source_path)
    with open(output_path, "w") as f:
        for ticker in unique_tickers:
            f.write(f"{ticker}\n")
    return unique_tickers

if __name__ == "__main__":
    unique_tickers = clean_failed_tickers()
    print(f"✅ Cleaned list saved to {unique_path}")
    print(f"Found {len(unique_tickers)} unique tickers.")
