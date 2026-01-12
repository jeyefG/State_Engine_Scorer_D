"""Event detection and feature extraction for the M5 Event Scorer (Layer A)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

EPS = 1e-9


class EventType(str, Enum):
    TOUCH_VWAP = "E_TOUCH_VWAP"
    REJECTION_VWAP = "E_REJECTION_VWAP"


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


class EventExtractor:
    """Layer A extractor: detect events and compute event features without leakage."""

    def __init__(self, *, eps: float = EPS) -> None:
        self.eps = eps

    def extract(self, df_m5: pd.DataFrame, symbol: str, *, vwap_col: str = "vwap") -> pd.DataFrame:
        if vwap_col not in df_m5.columns:
            raise ValueError(f"Missing VWAP column '{vwap_col}' required for event detection")

        required = {"open", "high", "low", "close"}
        missing = required - set(df_m5.columns)
        if missing:
            raise ValueError(f"Missing required OHLC columns: {sorted(missing)}")

        df = df_m5.copy()
        ts = _extract_timestamp(df)

        open_ = df["open"]
        high = df["high"]
        low = df["low"]
        close = df["close"]
        vwap = df[vwap_col]

        dist_to_vwap = close - vwap
        abs_dist_to_vwap = dist_to_vwap.abs()
        vwap_slope_5 = (vwap - vwap.shift(5)) / 5.0
        range_1 = (high - low)
        body_ratio = (close - open_).abs() / (range_1 + self.eps)
        upper_wick = high - np.maximum(open_, close)
        lower_wick = np.minimum(open_, close) - low
        upper_wick_ratio = upper_wick / (range_1 + self.eps)
        lower_wick_ratio = lower_wick / (range_1 + self.eps)

        if "atr" in df.columns:
            atr_14 = df["atr"]
        else:
            atr_14 = _atr(high, low, close, 14)

        dist_to_vwap_atr = dist_to_vwap / (atr_14 + self.eps)

        if "volume" in df.columns:
            volume_rel = df["volume"] / df["volume"].rolling(20).mean()
        else:
            volume_rel = pd.Series(np.nan, index=df.index)

        features = pd.DataFrame(
            {
                "dist_to_vwap": dist_to_vwap,
                "abs_dist_to_vwap": abs_dist_to_vwap,
                "vwap_slope_5": vwap_slope_5,
                "range_1": range_1,
                "body_ratio": body_ratio,
                "upper_wick_ratio": upper_wick_ratio,
                "lower_wick_ratio": lower_wick_ratio,
                "atr_14": atr_14,
                "dist_to_vwap_atr": dist_to_vwap_atr,
                "volume_rel": volume_rel,
                "hour": ts.dt.hour,
                "minute": ts.dt.minute,
            },
            index=df.index,
        )

        touch_mask = (low <= vwap) & (high >= vwap)
        rejection_mask = touch_mask & (
            ((close > vwap) & (open_ <= vwap)) | ((close < vwap) & (open_ >= vwap))
        )

        events = [
            _build_events(features, ts, symbol, touch_mask, EventType.TOUCH_VWAP),
            _build_events(features, ts, symbol, rejection_mask, EventType.REJECTION_VWAP),
        ]

        events_df = pd.concat([frame for frame in events if not frame.empty], axis=0)
        if events_df.empty:
            return pd.DataFrame(
                columns=[
                    "symbol",
                    "ts",
                    "family_id",
                    "event_id",
                    *features.columns,
                    "event_features",
                ]
            )

        events_df = events_df.sort_values("ts")
        events_df["event_id"] = range(1, len(events_df) + 1)
        return events_df.reset_index(drop=True)


def detect_events(df_m5_ctx: pd.DataFrame, config: EventDetectionConfig | None = None) -> pd.DataFrame:
    """Backward-compatible wrapper around EventExtractor.extract."""
    _ = config
    symbol = "UNKNOWN"
    if "symbol" in df_m5_ctx.columns and not df_m5_ctx["symbol"].empty:
        symbol = str(df_m5_ctx["symbol"].iloc[0])
    extractor = EventExtractor()
    return extractor.extract(df_m5_ctx, symbol=symbol)


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


def _extract_timestamp(df: pd.DataFrame) -> pd.Series:
    if isinstance(df.index, pd.DatetimeIndex):
        return pd.Series(df.index, index=df.index)
    if "ts" in df.columns:
        return pd.to_datetime(df["ts"])
    if "time" in df.columns:
        return pd.to_datetime(df["time"])
    raise ValueError("Missing datetime index or 'ts'/'time' column for timestamps")


def _build_events(
    features: pd.DataFrame,
    ts: pd.Series,
    symbol: str,
    mask: pd.Series,
    event_type: EventType,
) -> pd.DataFrame:
    if not mask.any():
        return pd.DataFrame()

    subset = features.loc[mask].copy()
    subset.insert(0, "symbol", symbol)
    subset.insert(1, "ts", ts.loc[mask])
    subset.insert(2, "family_id", event_type.value)
    subset["event_id"] = np.nan
    subset["event_features"] = subset[features.columns].to_dict(orient="records")
    return subset


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


__all__ = [
    "EventType",
    "EventExtractor",
    "EventFamily",
    "EventDetectionConfig",
    "detect_events",
    "label_events",
]
