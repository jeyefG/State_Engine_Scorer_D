import numpy as np
import pandas as pd

from state_engine.context_features import (
    _compute_standard_vwap,
    compute_dist_vwap_atr,
    compute_session_bucket,
    compute_state_age,
)


def test_compute_session_bucket_uses_hour_only() -> None:
    index = pd.date_range("2024-01-01 00:00", periods=4, freq="6h")
    buckets = compute_session_bucket(index)

    assert list(buckets) == ["ASIA", "ASIA", "LONDON", "NY_PM"]


def test_compute_state_age_resets_on_state_change() -> None:
    state_hat = pd.Series([1, 1, 2, 2, 2, 1, 1], index=pd.RangeIndex(7))
    ages = compute_state_age(state_hat, max_age=12)

    assert list(ages) == [1, 2, 1, 2, 3, 1, 2]


def test_compute_dist_vwap_atr_is_absolute() -> None:
    df = pd.DataFrame(
        {
            "high": [11.0, 11.0],
            "low": [9.0, 9.0],
            "close": [11.0, 9.0],
            "vwap": [10.0, 10.0],
            "atr_h2": [2.0, 2.0],
        }
    )
    dist = compute_dist_vwap_atr(df, atr_window=14, eps=1e-9)

    assert dist.iloc[0] == dist.iloc[1] == 0.5


def test_fallback_vwap_handles_zero_tick_volume() -> None:
    index = pd.date_range("2024-01-01 00:00", periods=30, freq="h")
    df = pd.DataFrame(
        {
            "high": np.linspace(100, 129, len(index)),
            "low": np.linspace(99, 128, len(index)),
            "close": np.linspace(100.5, 129.5, len(index)),
            "tick_volume": np.zeros(len(index)),
        },
        index=index,
    )

    vwap = _compute_standard_vwap(df, day_anchor="server_midnight", vwap_window=5)

    assert vwap is not None
    assert float(vwap.isna().mean() * 100.0) < 1.0
    assert np.isfinite(vwap.to_numpy()).all()

    day_two_idx = index[24]
    expected_pivot = (
        df.loc[day_two_idx, "high"]
        + df.loc[day_two_idx, "low"]
        + df.loc[day_two_idx, "close"]
    ) / 3.0
    assert np.isclose(vwap.loc[day_two_idx], expected_pivot)


def test_fallback_vwap_prefers_real_volume_over_tick_volume() -> None:
    index = pd.date_range("2024-01-01 00:00", periods=6, freq="h")
    df = pd.DataFrame(
        {
            "high": [10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
            "low": [9.0, 11.0, 13.0, 15.0, 17.0, 19.0],
            "close": [9.5, 11.5, 13.5, 15.5, 17.5, 19.5],
            "tick_volume": [0, 0, 0, 0, 0, 0],
            "real_volume": [10, 20, 5, 30, 15, 25],
        },
        index=index,
    )

    vwap_real = _compute_standard_vwap(df, day_anchor="server_midnight", vwap_window=3)
    vwap_tick_only = _compute_standard_vwap(
        df.drop(columns=["real_volume"]),
        day_anchor="server_midnight",
        vwap_window=3,
    )

    assert vwap_real is not None
    assert vwap_tick_only is not None
    assert float(vwap_real.isna().mean() * 100.0) < 1.0
    assert vwap_real.iloc[-1] != vwap_tick_only.iloc[-1]
