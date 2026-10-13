import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_news_script_reads_key_from_env():
    source = (ROOT / "fetch_news_sentiment.py").read_text()
    assert "NEWS_API_KEY" in source
    assert not re.search(r'API_KEY\s*=\s*"[0-9a-fA-F]{32}"', source)


def test_env_file_is_gitignored():
    ignored = {
        line.strip()
        for line in (ROOT / ".gitignore").read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    assert ".env" in ignored


def test_env_example_has_no_secret_values():
    for line in (ROOT / ".env.example").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.endswith("URL"):
            assert value.startswith("https://")
            continue
        assert value == ""
