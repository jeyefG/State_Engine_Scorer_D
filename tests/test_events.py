import pandas as pd

from state_engine.events import label_events


def _make_m5_df() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=6, freq="5min")
    return pd.DataFrame(
        {
            "open": [100, 100, 100, 100, 100, 100],
            "high": [101, 102, 101, 101, 101, 101],
            "low": [99, 98, 99, 99, 99, 99],
            "close": [100, 101, 100, 100, 100, 100],
        },
        index=idx,
    )


def test_label_events_uses_next_open_for_entry() -> None:
    df_m5 = _make_m5_df()
    events = pd.DataFrame(
        {
            "side": ["long"],
        },
        index=[df_m5.index[1]],
    )

    labeled = label_events(events, df_m5, k_bars=2, reward_r=1.0, sl_mult=1.0, atr_window=1)

    assert labeled["entry_price"].iloc[0] == df_m5["open"].iloc[2]


def test_label_events_outcome_is_clipped() -> None:
    df_m5 = _make_m5_df()
    events = pd.DataFrame(
        {
            "side": ["long"],
        },
        index=[df_m5.index[1]],
    )

    labeled = label_events(events, df_m5, k_bars=2, reward_r=2.0, sl_mult=1.0, atr_window=1)
    outcome = labeled["r_outcome"].iloc[0]

    assert outcome <= 2.0
    assert outcome >= -1.0


def test_label_events_tie_break_distance_prefers_sl() -> None:
    df_m5 = pd.DataFrame(
        {
            "open": [100, 100, 100],
            "high": [101, 102, 100],
            "low": [99, 98, 100],
            "close": [100, 100, 100],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="5min"),
    )
    events = pd.DataFrame(
        {
            "side": ["long"],
        },
        index=[df_m5.index[0]],
    )

    labeled = label_events(events, df_m5, k_bars=1, reward_r=1.0, sl_mult=1.0, atr_window=1)

    assert labeled["r_outcome"].iloc[0] == -1.0
