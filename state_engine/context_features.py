"""Contextual pseudo-features for H2 gating filters."""

from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd


_SESSION_BUCKETS = ("ASIA", "LONDON", "NY", "NY_PM")
_XAU_SYMBOLS = {"XAUUSD"}
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContextFeatureConfig:
    """Configuration for contextual pseudo-features."""

    max_state_age: int = 12
    atr_window: int = 14
    vwap_window: int = 50
    vwap_reset_mode: str | None = None
    eps: float = 1e-9


def _normalize_symbol(symbol: str | None) -> str:
    if symbol is None:
        return ""
    return "".join(ch for ch in str(symbol).upper() if ch.isalnum())


def _normalize_timeframe(timeframe: str | None) -> str:
    if timeframe is None:
        return ""
    return str(timeframe).upper()


def is_context_enabled(symbol: str | None, timeframe: str | None) -> bool:
    """Return True when XAUUSD H2 context features are enabled."""
    normalized_symbol = _normalize_symbol(symbol)
    normalized_tf = _normalize_timeframe(timeframe)
    return any(normalized_symbol.startswith(sym) for sym in _XAU_SYMBOLS) and normalized_tf == "H2"


def _hour_bucket(hour: int) -> str:
    if 0 <= hour <= 6:
        return "ASIA"
    if 7 <= hour <= 12:
        return "LONDON"
    if 13 <= hour <= 17:
        return "NY"
    return "NY_PM"


def compute_session_bucket(index: pd.DatetimeIndex) -> pd.Series:
    """Map timestamps to ASIA/LONDON/NY/NY_PM buckets using index hour.

    The hour is taken directly from the provided index (no timezone conversion).
    """
    hours = pd.Series(index.hour, index=index)
    return hours.map(_hour_bucket).rename("ctx_session_bucket")


def compute_state_age(state_hat: pd.Series, *, max_age: int) -> pd.Series:
    """Count consecutive bars in the same base state (1-based, clipped)."""
    state_series = state_hat.copy()
    change = state_series.ne(state_series.shift(1)) | state_series.isna()
    groups = change.cumsum()
    age = state_series.groupby(groups).cumcount() + 1
    age = age.where(state_series.notna()).clip(upper=max_age)
    return age.rename("ctx_state_age")


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


def _resolve_vwap_column(ohlcv: pd.DataFrame) -> str | None:
    for col in ("vwap", "VWAP"):
        if col in ohlcv.columns:
            return col
    return None


def _resolve_vwap_series(
    ohlcv: pd.DataFrame,
    *,
    vwap_window: int,
    reset_mode: str | None,
) -> pd.Series | None:
    from .events import _compute_vwap, _resolve_vwap_reset_mode

    vwap_col = _resolve_vwap_column(ohlcv)
    if vwap_col is not None:
        return pd.to_numeric(ohlcv[vwap_col], errors="coerce")

    effective_mode = _resolve_vwap_reset_mode(ohlcv, reset_mode)
    try:
        vwap = _compute_vwap(
            ohlcv,
            vwap_col="vwap",
            reset_mode=effective_mode,
            vwap_window=vwap_window,
        )
    except ValueError as exc:
        _LOGGER.warning("VWAP missing for ctx_dist_vwap_atr; skipping (%s).", exc)
        return None
    _LOGGER.warning(
        "VWAP missing for ctx_dist_vwap_atr; computed fallback (mode=%s, window=%s).",
        effective_mode,
        vwap_window,
    )
    return pd.to_numeric(vwap, errors="coerce")


def _resolve_atr_series(ohlcv: pd.DataFrame, *, window: int) -> pd.Series:
    for col in ohlcv.columns:
        if col.lower() == "atr_h2":
            return pd.to_numeric(ohlcv[col], errors="coerce")
    return _atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], window)


def compute_dist_vwap_atr(
    ohlcv: pd.DataFrame,
    *,
    atr_window: int,
    vwap_window: int,
    vwap_reset_mode: str | None,
    eps: float,
) -> pd.Series:
    """Compute absolute distance to VWAP normalized by ATR."""
    vwap = _resolve_vwap_series(
        ohlcv,
        vwap_window=vwap_window,
        reset_mode=vwap_reset_mode,
    )
    if vwap is None:
        return pd.Series(np.nan, index=ohlcv.index, name="ctx_dist_vwap_atr")
    atr = _resolve_atr_series(ohlcv, window=atr_window)
    close = pd.to_numeric(ohlcv["close"], errors="coerce")
    denom = np.maximum(atr.to_numpy(), eps)
    dist = (close - vwap).abs() / denom
    return pd.Series(dist, index=ohlcv.index, name="ctx_dist_vwap_atr")


def build_context_features(
    ohlcv: pd.DataFrame,
    outputs: pd.DataFrame,
    *,
    symbol: str | None,
    timeframe: str | None,
    config: ContextFeatureConfig | None = None,
) -> pd.DataFrame:
    """Build ctx_* features after state_hat/margin are available."""
    cfg = config or ContextFeatureConfig()
    if not is_context_enabled(symbol, timeframe):
        return pd.DataFrame(index=outputs.index)

    session_bucket = compute_session_bucket(outputs.index)
    state_age = compute_state_age(outputs["state_hat"], max_age=cfg.max_state_age)
    dist_vwap_atr = compute_dist_vwap_atr(
        ohlcv,
        atr_window=cfg.atr_window,
        vwap_window=cfg.vwap_window,
        vwap_reset_mode=cfg.vwap_reset_mode,
        eps=cfg.eps,
    )
    dist_vwap_atr = dist_vwap_atr.reindex(outputs.index)

    return pd.concat([session_bucket, state_age, dist_vwap_atr], axis=1)


__all__ = [
    "ContextFeatureConfig",
    "is_context_enabled",
    "compute_session_bucket",
    "compute_state_age",
    "compute_dist_vwap_atr",
    "build_context_features",
]
