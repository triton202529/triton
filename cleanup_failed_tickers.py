# cleanup_failed_tickers.py
failed_path = "data/logs/failed_tickers.txt"
unique_path = "data/logs/failed_tickers_unique.txt"

with open(failed_path, "r") as f:
    tickers = [line.strip() for line in f if line.strip()]

# Keep unique while preserving order
seen = set()
unique_tickers = []
for t in tickers:
    if t not in seen:
        seen.add(t)
        unique_tickers.append(t)

with open(unique_path, "w") as f:
    for t in unique_tickers:
        f.write(f"{t}\n")

print(f"✅ Cleaned list saved to {unique_path}")
print(f"Found {len(unique_tickers)} unique tickers.")
