def parse_failed_ticker_line(line):
    """Return just the ticker symbol from a failed-ticker log line."""
    ticker = line.strip().split("(", 1)[0].strip()
    return ticker or None

def unique_failed_tickers(lines):
    seen = set()
    tickers = []
    for line in lines:
        ticker = parse_failed_ticker_line(line)
        if ticker and ticker not in seen:
            seen.add(ticker)
            tickers.append(ticker)
    return tickers

def load_failed_tickers(path):
    with open(path, "r") as file:
        return unique_failed_tickers(file)
