"""Event detection and labeling for the M5 Event Scorer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class EventFamily(str, Enum):
    BALANCE_FADE = "E_BALANCE_FADE"
    BALANCE_REVERT = "E_BALANCE_REVERT"
    TRANSITION_TEST = "E_TRANSITION_TEST"
    TRANSITION_FAILURE = "E_TRANSITION_FAILURE"
    TREND_PULLBACK = "E_TREND_PULLBACK"
    TREND_CONTINUATION = "E_TREND_CONTINUATION"


@dataclass(frozen=True)
class EventDetectionConfig:
    balance_window: int = 12
    balance_edge: float = 0.20
    transition_window: int = 12
    trend_window: int = 6
    trend_pullback_edge: float = 0.30
    trend_continuation_comp_window: int = 6
    trend_continuation_break_window: int = 6
    trend_continuation_comp_thr: float = 1.2
    trend_continuation_momentum_window: int = 3


def detect_events(df_m5_ctx: pd.DataFrame, config: EventDetectionConfig | None = None) -> pd.DataFrame:
    """Detect candidate events on M5 given H1 context (shifted)."""
    cfg = config or EventDetectionConfig()
    required = {"open", "high", "low", "close"}
    missing = required - set(df_m5_ctx.columns)
    if missing:
        raise ValueError(f"Missing required columns for event detection: {sorted(missing)}")

    allow_cols = {
        "ALLOW_trend_pullback",
        "ALLOW_trend_continuation",
        "ALLOW_balance_fade",
        "ALLOW_transition_failure",
    }
    missing_allow = allow_cols - set(df_m5_ctx.columns)
    if missing_allow:
        raise ValueError(f"Missing required ALLOW_* columns: {sorted(missing_allow)}")

    df = df_m5_ctx.copy()
    high = df["high"]
    low = df["low"]
    close = df["close"]

    events: list[pd.DataFrame] = []

    # BALANCE FADE: extreme location within recent range.
    range_high = high.rolling(cfg.balance_window).max()
    range_low = low.rolling(cfg.balance_window).min()
    range_width = (range_high - range_low).replace(0, np.nan)
    location = (close - range_low) / range_width
    balance_mask = df["ALLOW_balance_fade"] == 1
    balance_short = balance_mask & (location >= (1.0 - cfg.balance_edge))
    balance_long = balance_mask & (location <= cfg.balance_edge)
    if balance_short.any():
        subset = df.loc[balance_short].copy()
        subset["family_id"] = EventFamily.BALANCE_FADE.value
        subset["side"] = "short"
        events.append(subset)
    if balance_long.any():
        subset = df.loc[balance_long].copy()
        subset["family_id"] = EventFamily.BALANCE_FADE.value
        subset["side"] = "long"
        events.append(subset)

    # TRANSITION FAILURE: re-entry after a brief break.
    prev_close = close.shift(1)
    range_high_prev = high.shift(1).rolling(cfg.transition_window).max()
    range_low_prev = low.shift(1).rolling(cfg.transition_window).min()
    inside_now = (close >= range_low_prev) & (close <= range_high_prev)
    outside_prev = (prev_close > range_high_prev) | (prev_close < range_low_prev)
    transition_mask = df["ALLOW_transition_failure"] == 1
    transition_event = transition_mask & inside_now & outside_prev
    if transition_event.any():
        subset = df.loc[transition_event].copy()
        subset["family_id"] = EventFamily.TRANSITION_FAILURE.value
        subset["side"] = np.where(prev_close.loc[transition_event] > range_high_prev.loc[transition_event], "short", "long")
        events.append(subset)

    # TREND PULLBACK: direction inferred from recent momentum + pullback location.
    trend_mask = df["ALLOW_trend_pullback"] == 1
    momentum = close - close.shift(cfg.trend_window)
    trend_range_high = high.rolling(cfg.trend_window).max()
    trend_range_low = low.rolling(cfg.trend_window).min()
    trend_range_width = (trend_range_high - trend_range_low).replace(0, np.nan)
    pullback_edge = cfg.trend_pullback_edge

    uptrend = trend_mask & (momentum > 0)
    downtrend = trend_mask & (momentum < 0)

    up_pullback = uptrend & (close <= (trend_range_low + pullback_edge * trend_range_width)) & (close < close.shift(1))
    down_pullback = downtrend & (close >= (trend_range_high - pullback_edge * trend_range_width)) & (close > close.shift(1))

    if up_pullback.any():
        subset = df.loc[up_pullback].copy()
        subset["family_id"] = EventFamily.TREND_PULLBACK.value
        subset["side"] = "long"
        events.append(subset)
    if down_pullback.any():
        subset = df.loc[down_pullback].copy()
        subset["family_id"] = EventFamily.TREND_PULLBACK.value
        subset["side"] = "short"
        events.append(subset)

    # TREND CONTINUATION: compression + breakout aligned with momentum.
    continuation_mask = df["ALLOW_trend_continuation"] == 1
    momentum_tc = close - close.shift(cfg.trend_continuation_momentum_window)
    momentum_sign = np.sign(momentum_tc)
    side_long = continuation_mask & (momentum_sign > 0)
    side_short = continuation_mask & (momentum_sign < 0)

    range_last_n = high.rolling(cfg.trend_continuation_comp_window).max() - low.rolling(cfg.trend_continuation_comp_window).min()
    if "atr_short" in df.columns:
        atr_short = df["atr_short"]
    else:
        atr_short = _atr(high, low, close, 14)
    compression = (range_last_n / atr_short) < cfg.trend_continuation_comp_thr

    breakout_high = high.shift(1).rolling(cfg.trend_continuation_break_window).max()
    breakout_low = low.shift(1).rolling(cfg.trend_continuation_break_window).min()
    breakout_long = close > breakout_high
    breakout_short = close < breakout_low

    trend_cont_long = side_long & compression & breakout_long
    trend_cont_short = side_short & compression & breakout_short
    if trend_cont_long.any():
        subset = df.loc[trend_cont_long].copy()
        subset["family_id"] = EventFamily.TREND_CONTINUATION.value
        subset["side"] = "long"
        subset["tc_n_comp"] = cfg.trend_continuation_comp_window
        subset["tc_n_brk"] = cfg.trend_continuation_break_window
        subset["tc_comp_thr"] = cfg.trend_continuation_comp_thr
        subset["tc_momentum_window"] = cfg.trend_continuation_momentum_window
        events.append(subset)
    if trend_cont_short.any():
        subset = df.loc[trend_cont_short].copy()
        subset["family_id"] = EventFamily.TREND_CONTINUATION.value
        subset["side"] = "short"
        subset["tc_n_comp"] = cfg.trend_continuation_comp_window
        subset["tc_n_brk"] = cfg.trend_continuation_break_window
        subset["tc_comp_thr"] = cfg.trend_continuation_comp_thr
        subset["tc_momentum_window"] = cfg.trend_continuation_momentum_window
        events.append(subset)

    if not events:
        return pd.DataFrame(columns=[*df.columns, "family_id", "side"])

    events_df = pd.concat(events).sort_index()
    events_df["event_price"] = events_df["close"]
    return events_df


def label_events(
    events_df: pd.DataFrame,
    df_m5: pd.DataFrame,
    k_bars: int,
    reward_r: float,
    sl_mult: float,
    atr_window: int = 14,
) -> pd.DataFrame:
    """Label events using a mechanical TP/SL proxy."""
    if events_df.empty:
        return events_df.copy()

    required = {"open", "high", "low", "close"}
    missing = required - set(df_m5.columns)
    if missing:
        raise ValueError(f"Missing required OHLC columns in df_m5: {sorted(missing)}")

    df = events_df.copy()
    atr_short = _atr(df_m5["high"], df_m5["low"], df_m5["close"], atr_window)

    entry_prices: list[float | None] = []
    sl_prices: list[float | None] = []
    tp_prices: list[float | None] = []
    labels: list[int | None] = []

    for ts, row in df.iterrows():
        if ts not in df_m5.index:
            entry_prices.append(None)
            sl_prices.append(None)
            tp_prices.append(None)
            labels.append(None)
            continue

        idx = df_m5.index.get_loc(ts)
        entry_idx = idx + 1
        if entry_idx >= len(df_m5.index):
            entry_prices.append(None)
            sl_prices.append(None)
            tp_prices.append(None)
            labels.append(None)
            continue

        entry_price = float(df_m5["open"].iloc[entry_idx])
        atr_value = atr_short.iloc[idx]
        if pd.isna(atr_value):
            entry_prices.append(entry_price)
            sl_prices.append(None)
            tp_prices.append(None)
            labels.append(None)
            continue

        sl_proxy = float(atr_value * sl_mult)
        if row["side"] == "long":
            sl = entry_price - sl_proxy
            tp = entry_price + reward_r * sl_proxy
        else:
            sl = entry_price + sl_proxy
            tp = entry_price - reward_r * sl_proxy

        outcome = 0
        for future_idx in range(entry_idx, min(entry_idx + k_bars, len(df_m5.index))):
            high = df_m5["high"].iloc[future_idx]
            low = df_m5["low"].iloc[future_idx]
            if row["side"] == "long":
                hit_sl = low <= sl
                hit_tp = high >= tp
            else:
                hit_sl = high >= sl
                hit_tp = low <= tp

            if hit_sl and hit_tp:
                outcome = 0
                break
            if hit_sl:
                outcome = 0
                break
            if hit_tp:
                outcome = 1
                break

        entry_prices.append(entry_price)
        sl_prices.append(sl)
        tp_prices.append(tp)
        labels.append(outcome)

    df["entry_price"] = entry_prices
    df["sl_price"] = sl_prices
    df["tp_price"] = tp_prices
    df["label"] = labels
    return df


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window).mean()


__all__ = ["EventFamily", "EventDetectionConfig", "detect_events", "label_events"]
