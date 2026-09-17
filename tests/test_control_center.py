import sys
import types

import pandas as pd


def _install_fake_streamlit():
    fake_st = types.ModuleType("streamlit")

    def cache_data(fn=None, **_kwargs):
        if callable(fn):
            return fn
        return lambda f: f

    class FakeColumn:
        def slider(self, *args, **kwargs):
            return 0.0

        def text_input(self, *args, **kwargs):
            return ""

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    fake_st.cache_data = cache_data
    fake_st.set_page_config = lambda **kwargs: None
    fake_st.title = lambda *args, **kwargs: None
    fake_st.warning = lambda *args, **kwargs: None
    fake_st.info = lambda *args, **kwargs: None
    fake_st.subheader = lambda *args, **kwargs: None
    fake_st.columns = lambda n: [FakeColumn() for _ in range(n if isinstance(n, int) else 2)]
    fake_st.write = lambda *args, **kwargs: None
    fake_st.slider = lambda *args, **kwargs: 0.0
    fake_st.text_input = lambda *args, **kwargs: ""
    fake_st.multiselect = lambda *args, **kwargs: []
    fake_st.dataframe = lambda *args, **kwargs: None
    fake_st.button = lambda *args, **kwargs: False
    fake_st.success = lambda *args, **kwargs: None
    fake_st.markdown = lambda *args, **kwargs: None
    fake_st.caption = lambda *args, **kwargs: None
    sys.modules["streamlit"] = fake_st
    sys.modules.pop("control_center", None)


def test_prepare_signals_accepts_committed_schema_without_confidence():
    _install_fake_streamlit()
    from control_center import prepare_signals
    df = pd.DataFrame(
        {
            "date": ["2020-08-05", "2020-08-05", "2020-08-06"],
            "ticker": ["AAPL", "AAPL", "MSFT"],
            "close": [100.0, 100.0, 200.0],
            "predicted_close": [110.0, 110.0, 180.0],
            "signal": ["BUY", "BUY", "SELL"],
            "total_score": [48.5, 48.5, 40.0],
            "rationale": ["Pred 110 vs 100", "Pred 110 vs 100", "Pred 180 vs 200"],
        }
    )

    out = prepare_signals(df)

    assert "confidence" in out.columns
    assert len(out[out["ticker"] == "AAPL"]) == 1
    assert out.loc[out["ticker"] == "AAPL", "confidence"].iloc[0] == 0.1
    assert out.loc[out["ticker"] == "MSFT", "confidence"].iloc[0] == 0.1


def test_prepare_signals_keeps_existing_confidence():
    _install_fake_streamlit()
    from control_center import prepare_signals
    df = pd.DataFrame(
        {
            "date": ["2024-01-02"],
            "ticker": ["NVDA"],
            "close": [100.0],
            "predicted_close": [150.0],
            "signal": ["BUY"],
            "confidence": [0.85],
        }
    )

    out = prepare_signals(df)

    assert out["confidence"].iloc[0] == 0.85
