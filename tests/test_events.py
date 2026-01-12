import pandas as pd

from state_engine.events import EventExtractor, EventType


def _make_m5_df() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=6, freq="5min")
    return pd.DataFrame(
        {
            "open": [100, 100, 100, 100, 100, 100],
            "high": [101, 102, 101, 101, 101, 101],
            "low": [99, 98, 99, 99, 99, 99],
            "close": [100, 101, 100, 100, 100, 100],
            "vwap": [100, 100, 100, 100, 100, 100],
            "volume": [10, 12, 11, 13, 10, 9],
        },
        index=idx,
    )


def test_events_schema_has_required_columns() -> None:
    df_m5 = _make_m5_df()
    extractor = EventExtractor()

    events = extractor.extract(df_m5, symbol="TEST")

    required = {
        "symbol",
        "ts",
        "family_id",
        "event_id",
        "dist_to_vwap",
        "abs_dist_to_vwap",
        "vwap_slope_5",
        "range_1",
        "body_ratio",
        "upper_wick_ratio",
        "lower_wick_ratio",
        "atr_14",
        "dist_to_vwap_atr",
        "volume_rel",
        "hour",
        "minute",
    }
    assert required.issubset(events.columns)


def test_events_no_leakage_features_ignore_future_changes() -> None:
    df_m5 = _make_m5_df()
    extractor = EventExtractor()

    events = extractor.extract(df_m5, symbol="TEST")
    event_ts = events.loc[0, "ts"]
    baseline = events.loc[events["ts"] == event_ts, "dist_to_vwap"].iloc[0]

    df_future = df_m5.copy()
    df_future.loc[df_future.index[-1], "close"] = 999
    df_future.loc[df_future.index[-1], "vwap"] = 999

    events_future = extractor.extract(df_future, symbol="TEST")
    updated = events_future.loc[events_future["ts"] == event_ts, "dist_to_vwap"].iloc[0]

    assert baseline == updated


def test_detects_touch_vwap_event() -> None:
    df_m5 = _make_m5_df()
    extractor = EventExtractor()

    events = extractor.extract(df_m5, symbol="TEST")

    assert (events["family_id"] == EventType.TOUCH_VWAP.value).any()
