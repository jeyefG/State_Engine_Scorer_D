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
    balance_min_range_atr: float = 0.8
    balance_soft_edge: float = 0.35
    balance_soft_range_mult: float = 0.5
    transition_window: int = 12
    transition_near_edge_atr: float = 0.25
    trend_window: int = 6
    trend_pullback_edge: float = 0.30
    trend_momentum_atr_thr: float = 0.3
    trend_pullback_momentum_mult: float = 0.5
    trend_pullback_edge_mult: float = 0.6
    trend_continuation_comp_window: int = 6
    trend_continuation_break_window: int = 6
    trend_continuation_comp_thr: float = 1.2
    trend_continuation_momentum_window: int = 3
    trend_continuation_comp_mult: float = 1.5
    trend_continuation_break_atr: float = 0.25


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

    if "atr_short" in df.columns:
        atr_short = df["atr_short"]
    else:
        atr_short = _atr(high, low, close, 14)

    # Shared context features to enrich candidates (proposal engine).
    range_high = high.rolling(cfg.balance_window).max()
    range_low = low.rolling(cfg.balance_window).min()
    range_width = (range_high - range_low).replace(0, np.nan)
    range_width_atr = range_width / atr_short.replace(0, np.nan)
    location_0_1 = (close - range_low) / range_width

    trend_momentum = close - close.shift(cfg.trend_window)
    momentum_norm = trend_momentum / atr_short.replace(0, np.nan)

    trend_range_high = high.rolling(cfg.trend_window).max()
    trend_range_low = low.rolling(cfg.trend_window).min()
    trend_range_width = (trend_range_high - trend_range_low).replace(0, np.nan)
    trend_location = (close - trend_range_low) / trend_range_width

    range_last_n = high.rolling(cfg.trend_continuation_comp_window).max() - low.rolling(
        cfg.trend_continuation_comp_window
    ).min()
    compression_ratio = range_last_n / atr_short.replace(0, np.nan)
    breakout_high = high.shift(1).rolling(cfg.trend_continuation_break_window).max()
    breakout_low = low.shift(1).rolling(cfg.trend_continuation_break_window).min()
    breakout_dist_long = (breakout_high - close) / atr_short.replace(0, np.nan)
    breakout_dist_short = (close - breakout_low) / atr_short.replace(0, np.nan)

    df["range_width_atr"] = range_width_atr
    df["location_0_1"] = location_0_1
    df["momentum_norm"] = momentum_norm
    df["compression_ratio"] = compression_ratio
    df["breakout_dist_atr"] = np.where(momentum_norm >= 0, breakout_dist_long, breakout_dist_short)

    # BALANCE FADE: softer extreme location within recent range.
    range_filter = range_width_atr > (cfg.balance_min_range_atr * cfg.balance_soft_range_mult)
    balance_mask = df["ALLOW_balance_fade"] == 1
    balance_edge = max(cfg.balance_edge, cfg.balance_soft_edge)
    balance_short = balance_mask & range_filter & (location_0_1 >= (1.0 - balance_edge))
    balance_long = balance_mask & range_filter & (location_0_1 <= balance_edge)
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

    # TRANSITION FAILURE: re-entry after a brief break or near-edge reclaim.
    prev_close = close.shift(1)
    range_high_prev = high.shift(1).rolling(cfg.transition_window).max()
    range_low_prev = low.shift(1).rolling(cfg.transition_window).min()
    inside_now = (close >= range_low_prev) & (close <= range_high_prev)
    outside_prev = (prev_close > range_high_prev) | (prev_close < range_low_prev)
    dist_to_high = (range_high_prev - close) / atr_short.replace(0, np.nan)
    dist_to_low = (close - range_low_prev) / atr_short.replace(0, np.nan)
    near_edge = (dist_to_high <= cfg.transition_near_edge_atr) | (dist_to_low <= cfg.transition_near_edge_atr)
    transition_mask = df["ALLOW_transition_failure"] == 1
    transition_event = transition_mask & inside_now & (outside_prev | near_edge)
    if transition_event.any():
        subset = df.loc[transition_event].copy()
        subset["family_id"] = EventFamily.TRANSITION_FAILURE.value
        side_series = np.where(
            outside_prev.loc[transition_event],
            np.where(prev_close.loc[transition_event] > range_high_prev.loc[transition_event], "short", "long"),
            np.where(dist_to_high.loc[transition_event] <= dist_to_low.loc[transition_event], "short", "long"),
        )
        subset["side"] = side_series
        subset["breakout_dist_atr"] = np.where(
            subset["side"] == "short",
            dist_to_high.loc[transition_event],
            dist_to_low.loc[transition_event],
        )
        events.append(subset)

    # TREND PULLBACK: direction inferred from recent momentum + pullback location.
    trend_mask = df["ALLOW_trend_pullback"] == 1
    pullback_edge = cfg.trend_pullback_edge * cfg.trend_pullback_edge_mult
    momentum_thr = cfg.trend_momentum_atr_thr * cfg.trend_pullback_momentum_mult

    uptrend = trend_mask & (momentum_norm >= momentum_thr)
    downtrend = trend_mask & (momentum_norm <= -momentum_thr)

    pullback_depth = np.where(momentum_norm >= 0, 1.0 - trend_location, trend_location)
    df["pullback_depth_0_1"] = pullback_depth

    up_pullback = uptrend & (pullback_depth >= pullback_edge)
    down_pullback = downtrend & (pullback_depth >= pullback_edge)

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

    # TREND CONTINUATION: compression + breakout aligned with momentum (soft pre-breakout allowed).
    continuation_mask = df["ALLOW_trend_continuation"] == 1
    momentum_tc = close - close.shift(cfg.trend_continuation_momentum_window)
    momentum_sign = np.sign(momentum_tc)
    side_long = continuation_mask & (momentum_sign > 0)
    side_short = continuation_mask & (momentum_sign < 0)

    compression = compression_ratio < (cfg.trend_continuation_comp_thr * cfg.trend_continuation_comp_mult)

    breakout_long = close > breakout_high
    breakout_short = close < breakout_low
    prebreak_long = breakout_dist_long <= cfg.trend_continuation_break_atr
    prebreak_short = breakout_dist_short <= cfg.trend_continuation_break_atr

    trend_cont_long = side_long & compression & (breakout_long | prebreak_long)
    trend_cont_short = side_short & compression & (breakout_short | prebreak_short)
    if trend_cont_long.any():
        subset = df.loc[trend_cont_long].copy()
        subset["family_id"] = EventFamily.TREND_CONTINUATION.value
        subset["side"] = "long"
        subset["breakout_dist_atr"] = breakout_dist_long.loc[trend_cont_long]
        subset["tc_n_comp"] = cfg.trend_continuation_comp_window
        subset["tc_n_brk"] = cfg.trend_continuation_break_window
        subset["tc_comp_thr"] = cfg.trend_continuation_comp_thr
        subset["tc_momentum_window"] = cfg.trend_continuation_momentum_window
        events.append(subset)
    if trend_cont_short.any():
        subset = df.loc[trend_cont_short].copy()
        subset["family_id"] = EventFamily.TREND_CONTINUATION.value
        subset["side"] = "short"
        subset["breakout_dist_atr"] = breakout_dist_short.loc[trend_cont_short]
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
    r_thr: float = 0.0,
    tie_break: str = "distance",
    clip_mtm: bool = True,
) -> pd.DataFrame:
    """Label events using a triple-barrier proxy with continuous outcome."""
    if events_df.empty:
        return events_df.copy()

    required = {"open", "high", "low", "close"}
    missing = required - set(df_m5.columns)
    if missing:
        raise ValueError(f"Missing required OHLC columns in df_m5: {sorted(missing)}")

    df = events_df.copy().sort_index()
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")]

    df_m5 = df_m5.copy().sort_index()
    if df_m5.index.has_duplicates:
        df_m5 = df_m5[~df_m5.index.duplicated(keep="last")]

    atr_short = _atr(df_m5["high"], df_m5["low"], df_m5["close"], atr_window)

    if tie_break not in {"worst", "distance"}:
        raise ValueError("tie_break must be 'worst' or 'distance'")

    entry_prices: list[float | None] = []
    sl_prices: list[float | None] = []
    tp_prices: list[float | None] = []
    labels: list[int | None] = []
    r_outcomes: list[float | None] = []
    r_mean_k: list[float | None] = []

    indexer = df_m5.index.get_indexer(df.index)

    for i, (ts, row) in enumerate(df.iterrows()):
        idx = indexer[i]
        if idx == -1:
            entry_prices.append(None)
            sl_prices.append(None)
            tp_prices.append(None)
            labels.append(None)
            r_outcomes.append(None)
            r_mean_k.append(None)
            continue

        entry_idx = idx + 1
        if entry_idx >= len(df_m5.index):
            entry_prices.append(None)
            sl_prices.append(None)
            tp_prices.append(None)
            labels.append(None)
            r_outcomes.append(None)
            r_mean_k.append(None)
            continue

        entry_price = float(df_m5["open"].iloc[entry_idx])
        atr_value = atr_short.iloc[idx]
        if pd.isna(atr_value):
            entry_prices.append(entry_price)
            sl_prices.append(None)
            tp_prices.append(None)
            labels.append(None)
            r_outcomes.append(None)
            r_mean_k.append(None)
            continue

        sl_proxy = float(atr_value * sl_mult)
        if row["side"] == "long":
            sl = entry_price - sl_proxy
            tp = entry_price + reward_r * sl_proxy
            direction = 1.0
        else:
            sl = entry_price + sl_proxy
            tp = entry_price - reward_r * sl_proxy
            direction = -1.0

        outcome: float | None = None
        end_idx = min(entry_idx + k_bars, len(df_m5.index))
        future_slice = df_m5.iloc[entry_idx:end_idx]
        for future_idx, (high_val, low_val) in enumerate(
            zip(future_slice["high"].to_numpy(), future_slice["low"].to_numpy()),
            start=entry_idx,
        ):
            if row["side"] == "long":
                hit_sl = low_val <= sl
                hit_tp = high_val >= tp
            else:
                hit_sl = high_val >= sl
                hit_tp = low_val <= tp

            if hit_sl and hit_tp:
                if tie_break == "worst":
                    outcome = -1.0
                else:
                    open_price = float(df_m5["open"].iloc[future_idx])
                    sl_dist = abs(open_price - sl)
                    tp_dist = abs(tp - open_price)
                    outcome = -1.0 if sl_dist <= tp_dist else reward_r
                break
            if hit_sl:
                outcome = -1.0
                break
            if hit_tp:
                outcome = reward_r
                break

        if outcome is None:
            end_idx_mtm = min(entry_idx + k_bars - 1, len(df_m5.index) - 1)
            close_end = float(df_m5["close"].iloc[end_idx_mtm])
            sl_dist = sl_proxy if sl_proxy != 0 else np.nan
            outcome = direction * (close_end - entry_price) / sl_dist
            if clip_mtm and not pd.isna(outcome):
                outcome = float(np.clip(outcome, -1.0, reward_r))

        if sl_proxy != 0 and not future_slice.empty:
            mtm_series = direction * (future_slice["close"] - entry_price) / sl_proxy
            r_mean_k.append(float(mtm_series.mean()))
        else:
            r_mean_k.append(None)

        entry_prices.append(entry_price)
        sl_prices.append(sl)
        tp_prices.append(tp)
        if outcome is None or pd.isna(outcome):
            r_outcomes.append(None)
            labels.append(None)
        else:
            r_outcomes.append(float(outcome))
            labels.append(int(outcome > r_thr))

    df["entry_price"] = entry_prices
    df["sl_price"] = sl_prices
    df["tp_price"] = tp_prices
    df["label"] = labels
    df["r_outcome"] = r_outcomes
    df["r_mean_k"] = r_mean_k
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
