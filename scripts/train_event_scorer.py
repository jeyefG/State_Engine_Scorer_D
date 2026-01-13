"""Train the Event Scorer model from MT5 data.

The scorer uses a triple-barrier continuous outcome (r_outcome) and reports
ranking metrics like lift@K to gauge whether top-ranked events outperform the
base rate. lift@K = precision@K / base_rate.
"""

from __future__ import annotations

import argparse
import json
import logging
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
import io
from pathlib import Path
import sys
import textwrap

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sklearn.metrics import roc_auc_score

from state_engine.events import EventDetectionConfig, detect_events, label_events
from state_engine.features import FeatureConfig, FeatureEngineer
from state_engine.gating import GatingPolicy
from state_engine.model import StateEngineModel
from state_engine.mt5_connector import MT5Connector
from state_engine.labels import StateLabels
from state_engine.scoring import EventScorer, EventScorerBundle, EventScorerConfig, FeatureBuilder
from state_engine.session import SESSION_BUCKETS, get_session_bucket
from state_engine.config_loader import deep_merge, load_config


def parse_args() -> argparse.Namespace:
    default_end = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    default_start = (datetime.today() - timedelta(days=700)).strftime("%Y-%m-%d")
    parser = argparse.ArgumentParser(description="Train Event Scorer model.")
    parser.add_argument("--symbol", default="EURUSD", help="Símbolo MT5 (ej. EURUSD)")
    parser.add_argument("--start", default=default_start, help="Fecha inicio (YYYY-MM-DD) para descarga M5/H1")
    parser.add_argument("--end", default=default_end, help="Fecha fin (YYYY-MM-DD) para descarga M5/H1")
    parser.add_argument("--state-model", type=Path, default=None, help="Ruta del modelo State Engine (pkl)")
    parser.add_argument("--model-out", type=Path, default=None, help="Ruta de salida para Event Scorer")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(PROJECT_ROOT / "state_engine" / "models"),
        help="Directorio base para modelos",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Ratio train/calibración")
    parser.add_argument("--k-bars", type=int, default=24, help="Ventana futura K para etiquetas")
    parser.add_argument("--reward-r", type=float, default=1.0, help="R múltiplo para TP proxy")
    parser.add_argument("--sl-mult", type=float, default=1.0, help="Multiplicador de ATR para SL proxy")
    parser.add_argument("--r-thr", type=float, default=0.0, help="Umbral para label binario basado en r_outcome")
    parser.add_argument("--tie-break", default="distance", choices=["distance", "worst"], help="Tie-break TP/SL")
    parser.add_argument(
        "--meta-policy",
        default="on",
        choices=["on", "off"],
        help="Activar meta policy (gating superior) para regímenes operables",
    )
    parser.add_argument(
        "--include-transition",
        default="on",
        choices=["on", "off"],
        help="Incluir eventos en régimen TRANSITION (default on)",
    )
    parser.add_argument("--meta-margin-min", type=float, default=0.10, help="Margen mínimo H1 para meta policy")
    parser.add_argument("--meta-margin-max", type=float, default=0.95, help="Margen máximo H1 para meta policy")
    parser.add_argument("--decision-n-min", type=int, default=200, help="N mínimo para decision tradeable")
    parser.add_argument("--decision-r-mean-min", type=float, default=0.02, help="r_mean mínimo para decision tradeable")
    parser.add_argument("--decision-winrate-min", type=float, default=0.52, help="Winrate mínimo para decision tradeable")
    parser.add_argument("--decision-p10-min", type=float, default=-0.2, help="p10 mínimo para decision tradeable")
    parser.add_argument("--fallback-min-samples", type=int, default=300, help="Mínimo de muestras post-meta para metrics")
    parser.add_argument("--config", type=Path, default=None, help="Ruta config YAML/JSON con overrides por símbolo")
    parser.add_argument(
        "--mode",
        default="production",
        choices=["research", "production"],
        help="Modo de thresholds para reportes diagnósticos",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level (INFO, DEBUG, WARNING)")
    return parser.parse_args()


def _default_symbol_config(args: argparse.Namespace) -> dict:
    return {
        "symbol": args.symbol,
        "event_scorer": {
            "train_ratio": args.train_ratio,
            "k_bars": args.k_bars,
            "reward_r": args.reward_r,
            "sl_mult": args.sl_mult,
            "r_thr": args.r_thr,
            "tie_break": args.tie_break,
            "include_transition": args.include_transition == "on",
            "meta_policy": {
                "enabled": args.meta_policy == "on",
                "meta_margin_min": args.meta_margin_min,
                "meta_margin_max": args.meta_margin_max,
            },
            "family_training": {
                "min_samples_train": 200,
            },
            "decision_thresholds": {
                "n_min": args.decision_n_min,
                "winrate_min": args.decision_winrate_min,
                "r_mean_min": args.decision_r_mean_min,
                "p10_min": args.decision_p10_min,
            },
            "research": {
                "enabled": False,
                "d1_anchor_hour": 0,
                "features": {
                    "session_bucket": False,
                    "hour_bucket": False,
                    "trend_context_D1": False,
                    "vol_context": False,
                },
                "k_bars_grid": [args.k_bars],
                "diagnostics": {
                    "max_family_concentration": 0.6,
                    "min_temporal_dispersion": 0.3,
                },
            },
        },
    }


def _resolve_decision_thresholds(config: dict, mode: str) -> dict:
    event_cfg = config.get("event_scorer", {}) if isinstance(config.get("event_scorer"), dict) else {}
    base_thresholds = event_cfg.get("decision_thresholds", {}) or {}
    mode_thresholds = (
        event_cfg.get("modes", {}).get(mode, {}).get("decision_thresholds", {}) if isinstance(event_cfg.get("modes"), dict) else {}
    )
    return deep_merge(base_thresholds, mode_thresholds or {})


def _resolve_research_config(config: dict, mode: str) -> dict:
    event_cfg = config.get("event_scorer", {}) if isinstance(config.get("event_scorer"), dict) else {}
    research_cfg = event_cfg.get("research", {}) if isinstance(event_cfg.get("research"), dict) else {}
    enabled = bool(research_cfg.get("enabled", False)) and mode == "research"
    features = research_cfg.get("features", {}) if isinstance(research_cfg.get("features"), dict) else {}
    diagnostics = research_cfg.get("diagnostics", {}) if isinstance(research_cfg.get("diagnostics"), dict) else {}
    k_bars_grid = research_cfg.get("k_bars_grid")
    return {
        "enabled": enabled,
        "features": features,
        "diagnostics": diagnostics,
        "k_bars_grid": k_bars_grid,
        "d1_anchor_hour": research_cfg.get("d1_anchor_hour", 0),
    }


def setup_logging(level: str) -> logging.Logger:
    logger = logging.getLogger("event_scorer")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)-8s %(asctime)s | %(message)s")
    handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def build_h1_context(
    ohlcv_h1: pd.DataFrame,
    state_model: StateEngineModel,
    feature_engineer: FeatureEngineer,
    gating: GatingPolicy,
) -> pd.DataFrame:
    full_features = feature_engineer.compute_features(ohlcv_h1)
    features = feature_engineer.training_features(full_features)
    outputs = state_model.predict_outputs(features)
    allows = gating.apply(outputs, features=full_features)
    ctx = pd.concat([outputs[["state_hat", "margin"]], allows], axis=1)
    ctx = ctx.shift(1)
    return ctx


def merge_h1_m5(ctx_h1: pd.DataFrame, ohlcv_m5: pd.DataFrame) -> pd.DataFrame:
    logger = logging.getLogger("event_scorer")
    h1 = ctx_h1.copy().sort_index()
    m5 = ohlcv_m5.copy().sort_index()
    if getattr(h1.index, "tz", None) is not None:
        h1.index = h1.index.tz_localize(None)
    if getattr(m5.index, "tz", None) is not None:
        m5.index = m5.index.tz_localize(None)
    h1 = h1.reset_index().rename(columns={h1.index.name or "index": "time"})
    m5 = m5.reset_index().rename(columns={m5.index.name or "index": "time"})
    merged = pd.merge_asof(m5, h1, on="time", direction="backward")
    merged = merged.set_index("time")
    merged = merged.rename(columns={"state_hat": "state_hat_H1", "margin": "margin_H1"})
    missing_ctx = merged[["state_hat_H1", "margin_H1"]].isna().mean()
    if (missing_ctx > 0.25).any():
        logger.warning("High missing context after merge: %s", missing_ctx.to_dict())
    return merged


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


def _ensure_atr_14(df: pd.DataFrame) -> pd.Series:
    for col in ("atr_14", "atr", "ATR", "atr14"):
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce").astype(float)
            return series.rename("atr_14")
    return _atr(df["high"], df["low"], df["close"], 14).rename("atr_14")


def _reset_index_for_export(df: pd.DataFrame) -> pd.DataFrame:
    index_name = df.index.name or "index"
    if index_name in df.columns:
        return df.reset_index(drop=True)
    return df.reset_index()


def _with_suffix(path: Path, suffix: str) -> Path:
    if not suffix:
        return path
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def _format_interval(interval: pd.Interval | str) -> str:
    if isinstance(interval, pd.Interval):
        left = f"{interval.left:.2f}"
        right = f"{interval.right:.2f}"
        return f"m({left}-{right}]"
    if isinstance(interval, str):
        return f"m{interval}"
    return "mNA"


def _margin_bins(series: pd.Series, q: int = 3) -> pd.Series:
    bins = pd.Series(index=series.index, dtype="object")
    non_na = series.dropna()
    if non_na.empty:
        return bins
    try:
        bins.loc[non_na.index] = pd.qcut(non_na, q=q, duplicates="drop")
    except ValueError:
        rank_pct = non_na.rank(pct=True)
        bins.loc[non_na.index] = pd.cut(rank_pct, bins=q, include_lowest=True)
    return bins


def _make_margin_bin_rank(series: pd.Series) -> pd.Series:
    bins = pd.Series(index=series.index, dtype="object")
    non_na = series.dropna()
    if non_na.empty:
        return bins
    try:
        cuts = pd.qcut(non_na, q=3, duplicates="drop")
    except ValueError:
        cuts = pd.Series(index=non_na.index, dtype="object")
    if not isinstance(cuts, pd.Series) or cuts.isna().all():
        p33, p66 = non_na.quantile([1 / 3, 2 / 3])
        cuts = pd.cut(
            non_na,
            bins=[-np.inf, p33, p66, np.inf],
            include_lowest=True,
        )
    categories = list(pd.Series(cuts).cat.categories)
    mapping = {cat: f"m{idx}" for idx, cat in enumerate(categories)}
    bins.loc[non_na.index] = pd.Series(cuts).map(mapping)
    return bins


def _state_label(value: int | float) -> str:
    try:
        return StateLabels(int(value)).name
    except Exception:
        return "UNKNOWN"


def _vwap_zone(dist_to_vwap_atr: pd.Series) -> pd.Series:
    dist_abs = dist_to_vwap_atr.abs()
    zones = np.select(
        [
            dist_abs < 0.35,
            dist_abs < 0.80,
            dist_abs < 1.60,
        ],
        ["CENTER", "INNER", "OUTER"],
        default="EXTREME",
    )
    zones = pd.Series(zones, index=dist_to_vwap_atr.index)
    zones = zones.where(dist_to_vwap_atr.notna(), other="UNKNOWN")
    return zones


def _resolve_vwap_report_mode(
    events_df: pd.DataFrame,
    df_m5_ctx: pd.DataFrame,
    event_config: EventDetectionConfig,
) -> str:
    attrs_mode = events_df.attrs.get("vwap_reset_mode_effective")
    if attrs_mode is None and "vwap_reset_mode_effective" in events_df.columns and not events_df.empty:
        attrs_mode = events_df["vwap_reset_mode_effective"].iloc[0]
    if attrs_mode is not None:
        return str(attrs_mode)
    if "vwap" in df_m5_ctx.columns:
        return "provided (no metadata)"
    config_mode = (
        event_config.vwap_reset_mode
        if event_config.vwap_reset_mode is not None
        else ("session" if any(col in df_m5_ctx.columns for col in ("session_id", "session")) else "daily")
    )
    return f"config_{config_mode} (no metadata)"

def _value_state(
    overlap_ratio: pd.Series,
    compression_ratio: pd.Series,
    range_width_atr: pd.Series,
) -> pd.Series:
    valid = overlap_ratio.notna() & compression_ratio.notna() & range_width_atr.notna()

    at_value = (overlap_ratio >= 0.55) & (compression_ratio >= 0.55)
    far_value = (overlap_ratio <= 0.25) & (range_width_atr >= 1.20)

    states = np.select(
        [at_value, far_value],
        ["AT_VALUE", "FAR_FROM_VALUE"],
        default="NEAR_VALUE",
    )
    states = pd.Series(states, index=overlap_ratio.index)
    states = states.where(valid, other="UNKNOWN")
    return states

def _expansion_state(
    range_atr: pd.Series,
    atr_ratio: pd.Series,
    breakout_dist_atr: pd.Series,
) -> pd.Series:
    valid = range_atr.notna() & atr_ratio.notna() & breakout_dist_atr.notna()
    strong = (range_atr >= 1.6) & (breakout_dist_atr >= 0.6) & (atr_ratio >= 1.1)
    none = (range_atr < 1.0) & (breakout_dist_atr < 0.25) & (atr_ratio <= 1.0)
    states = np.select(
        [strong, none],
        ["STRONG", "NONE"],
        default="MILD",
    )
    states = pd.Series(states, index=range_atr.index)
    states = states.where(valid, other="UNKNOWN")
    return states


PSEUDO_FEATURES = {
    "vwap_zone": ["CENTER", "INNER", "OUTER", "EXTREME", "UNKNOWN"],
    "value_state": ["AT_VALUE", "NEAR_VALUE", "FAR_FROM_VALUE", "UNKNOWN"],
    "expansion_state": ["NONE", "MILD", "STRONG", "UNKNOWN"],
}


def _encode_pseudo_features(pseudo_features: pd.DataFrame) -> pd.DataFrame:
    encoded_parts: list[pd.DataFrame] = []
    for feature_name, categories in PSEUDO_FEATURES.items():
        series = pseudo_features[feature_name].astype("category")
        series = series.cat.set_categories(categories)
        dummies = pd.get_dummies(series, prefix=f"pf_{feature_name}")
        for category in categories:
            col = f"pf_{feature_name}_{category}"
            if col not in dummies.columns:
                dummies[col] = 0
        encoded_parts.append(dummies)
    return pd.concat(encoded_parts, axis=1)


def _build_pseudo_features(
    events_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    atr_ratio: pd.Series,
) -> pd.DataFrame:
    if "dist_to_vwap_atr" in events_df.columns:
        dist_to_vwap_atr = pd.to_numeric(events_df["dist_to_vwap_atr"], errors="coerce")
    else:
        dist_to_vwap_atr = pd.Series(np.nan, index=events_df.index)
    overlap_ratio = pd.to_numeric(feature_df.get("overlap_ratio", pd.Series(np.nan, index=events_df.index)), errors="coerce")
    compression_ratio = pd.to_numeric(
        feature_df.get("compression_ratio", pd.Series(np.nan, index=events_df.index)),
        errors="coerce",
    )
    range_width_atr = pd.to_numeric(
        feature_df.get("range_width_atr", pd.Series(np.nan, index=events_df.index)),
        errors="coerce",
    )
    range_atr = pd.to_numeric(feature_df.get("range_atr", pd.Series(np.nan, index=events_df.index)), errors="coerce")
    breakout_dist_atr = pd.to_numeric(
        feature_df.get("breakout_dist_atr", pd.Series(np.nan, index=events_df.index)),
        errors="coerce",
    )
    atr_ratio_event = atr_ratio.reindex(feature_df.index)
    pseudo = pd.DataFrame(
        {
            "vwap_zone": _vwap_zone(dist_to_vwap_atr),
            "value_state": _value_state(overlap_ratio, compression_ratio, range_width_atr),
            "expansion_state": _expansion_state(range_atr, atr_ratio_event, breakout_dist_atr),
        },
        index=events_df.index,
    )
    return pseudo


def _format_state_mix(counts: pd.Series, state_order: list[str]) -> list[str]:
    total = counts.sum()
    lines = []
    for state in state_order:
        count = int(counts.get(state, 0))
        pct = (count / total * 100.0) if total else 0.0
        lines.append(f"{state}={count} ({pct:.2f}%)")
    return lines


def _format_metric(value: float) -> str:
    if pd.isna(value):
        return "nan"
    return f"{value:.4f}"

def _conditional_edge_regime_table(
    events_diag: pd.DataFrame,
    pseudo_col: str,
    min_n: int,
) -> pd.DataFrame:
    columns = [
        "state_hat_H1",
        "margin_bin",
        "allow_id",
        "family_id",
        pseudo_col,
        "n",
        "winrate",
        "r_mean",
        "r_median",
        "p10",
        "p90",
    ]
    if events_diag.empty or pseudo_col not in events_diag.columns:
        return pd.DataFrame(columns=columns)

    grouped = events_diag.groupby(
        ["state_hat_H1", "margin_bin", "allow_id", "family_id", pseudo_col],
        observed=True,
    )

    summary = grouped.agg(
        n=("win", "size"),
        winrate=("win", "mean"),
        r_mean=("r", "mean"),
        r_median=("r", "median"),
        p10=("r", lambda s: s.quantile(0.1)),
        p90=("r", lambda s: s.quantile(0.9)),
    ).reset_index()

    summary = summary[summary["n"] >= min_n]
    summary = summary.sort_values(
        ["r_mean", "winrate", "n"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return summary[columns]

def _meta_policy_mask(
    events_df: pd.DataFrame,
    allow_cols: list[str],
    margin_min: float,
    margin_max: float,
) -> pd.Series:
    allow_active = (
        events_df[allow_cols].fillna(0).sum(axis=1) > 0
        if allow_cols
        else pd.Series(False, index=events_df.index)
    )
    margin_ok = events_df["margin_H1"].between(margin_min, margin_max, inclusive="both")
    return allow_active & margin_ok


def _coverage_table(events_diag: pd.DataFrame, df_m5_ctx: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns_global = [
        "events_total_post_meta",
        "events_per_day",
        "unique_days",
        "state_balance_pct",
        "state_trend_pct",
        "state_transition_pct",
        "margin_bin_m0_pct",
        "margin_bin_m1_pct",
        "margin_bin_m2_pct",
        "events_index_match_pct",
    ]
    columns_by = ["state_hat_H1", "margin_bin", "events_count", "events_pct"]
    if events_diag.empty:
        empty_global = pd.DataFrame([{col: 0.0 for col in columns_global}])
        empty_global["events_index_match_pct"] = float("nan") if df_m5_ctx is None else float("nan")
        empty_by = pd.DataFrame(columns=columns_by)
        return empty_global, empty_by
    total = len(events_diag)
    unique_days = events_diag.index.normalize().nunique()
    events_per_day = total / unique_days if unique_days else 0.0
    state_counts = events_diag["state_hat_H1"].value_counts()
    margin_counts = events_diag["margin_bin"].value_counts()
    index_match = (
        float(events_diag.index.isin(df_m5_ctx.index).mean()) if df_m5_ctx is not None else float("nan")
    )
    global_row = {
        "events_total_post_meta": total,
        "events_per_day": events_per_day,
        "unique_days": unique_days,
        "state_balance_pct": (state_counts.get("BALANCE", 0) / total * 100.0) if total else 0.0,
        "state_trend_pct": (state_counts.get("TREND", 0) / total * 100.0) if total else 0.0,
        "state_transition_pct": (state_counts.get("TRANSITION", 0) / total * 100.0) if total else 0.0,
        "margin_bin_m0_pct": (margin_counts.get("m0", 0) / total * 100.0) if total else 0.0,
        "margin_bin_m1_pct": (margin_counts.get("m1", 0) / total * 100.0) if total else 0.0,
        "margin_bin_m2_pct": (margin_counts.get("m2", 0) / total * 100.0) if total else 0.0,
        "events_index_match_pct": index_match,
    }
    coverage_global = pd.DataFrame([global_row])[columns_global]
    by_state = (
        events_diag.groupby(["state_hat_H1", "margin_bin"], observed=True)
        .size()
        .reset_index(name="events_count")
    )
    by_state["events_pct"] = (by_state["events_count"] / total * 100.0) if total else 0.0
    by_state = by_state.sort_values(["state_hat_H1", "margin_bin"]).reset_index(drop=True)
    return coverage_global, by_state[columns_by]


def _regime_edge_table(events_diag: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "state_hat_H1",
        "margin_bin",
        "allow_id",
        "n",
        "winrate",
        "r_mean",
        "r_median",
        "p10",
        "p90",
    ]
    if events_diag.empty:
        return pd.DataFrame(columns=columns), pd.DataFrame(columns=columns)
    grouped = events_diag.groupby(["state_hat_H1", "margin_bin", "allow_id"], observed=True)
    summary = grouped.agg(
        n=("win", "size"),
        winrate=("win", "mean"),
        r_mean=("r", "mean"),
        r_median=("r", "median"),
        p10=("r", lambda s: s.quantile(0.1)),
        p90=("r", lambda s: s.quantile(0.9)),
    ).reset_index()
    summary = summary.sort_values(["state_hat_H1", "margin_bin", "allow_id"]).reset_index(drop=True)
    ranked_desc = summary.sort_values(
        ["r_mean", "n", "winrate", "state_hat_H1", "margin_bin", "allow_id"],
        ascending=[False, False, False, True, True, True],
    )
    ranked_asc = summary.sort_values(
        ["r_mean", "n", "winrate", "state_hat_H1", "margin_bin", "allow_id"],
        ascending=[True, False, False, True, True, True],
    )
    top10 = ranked_desc.head(10)
    bottom10 = ranked_asc.head(10)
    ranked = pd.concat([top10, bottom10], ignore_index=True)
    return summary[columns], ranked[columns]


def _temporal_splits() -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    return [
        ("2024H1", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-06-30 23:59:59")),
        ("2024H2", pd.Timestamp("2024-07-01"), pd.Timestamp("2024-12-31 23:59:59")),
        ("2025H1", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-06-30 23:59:59")),
        ("2025H2", pd.Timestamp("2025-07-01"), pd.Timestamp("2025-12-31 23:59:59")),
        ("2026YTD", pd.Timestamp("2026-01-01"), pd.Timestamp.max),
    ]


def _stability_table(events_diag: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "split",
        "state_hat_H1",
        "margin_bin",
        "allow_id",
        "n",
        "winrate",
        "r_mean",
        "r_median",
        "p10",
        "p90",
    ]
    if events_diag.empty:
        return pd.DataFrame(columns=columns)
    rows: list[pd.DataFrame] = []
    for split_name, start_ts, end_ts in _temporal_splits():
        mask = (events_diag.index >= start_ts) & (events_diag.index <= end_ts)
        split_events = events_diag.loc[mask]
        if split_events.empty:
            continue
        grouped = split_events.groupby(["state_hat_H1", "margin_bin", "allow_id"], observed=True)
        summary = grouped.agg(
            n=("win", "size"),
            winrate=("win", "mean"),
            r_mean=("r", "mean"),
            r_median=("r", "median"),
            p10=("r", lambda s: s.quantile(0.1)),
            p90=("r", lambda s: s.quantile(0.9)),
        ).reset_index()
        summary.insert(0, "split", split_name)
        rows.append(summary)
    if not rows:
        return pd.DataFrame(columns=columns)
    table = pd.concat(rows, ignore_index=True)
    table = table.sort_values(["split", "state_hat_H1", "margin_bin", "allow_id"]).reset_index(drop=True)
    return table[columns]


def _decision_table(events_diag: pd.DataFrame, thresholds: argparse.Namespace) -> pd.DataFrame:
    columns = [
        "state_hat_H1",
        "margin_bin",
        "allow_id",
        "n",
        "winrate",
        "r_mean",
        "r_median",
        "p10",
        "p90",
        "decision",
        "decision_reason",
    ]
    if events_diag.empty:
        return pd.DataFrame(columns=columns)
    grouped = events_diag.groupby(["state_hat_H1", "margin_bin", "allow_id"], observed=True)
    summary = grouped.agg(
        n=("win", "size"),
        winrate=("win", "mean"),
        r_mean=("r", "mean"),
        r_median=("r", "median"),
        p10=("r", lambda s: s.quantile(0.1)),
        p90=("r", lambda s: s.quantile(0.9)),
    ).reset_index()
    decisions = []
    for _, row in summary.iterrows():
        reasons = []
        if row["n"] < thresholds.decision_n_min:
            reasons.append(f"n<{thresholds.decision_n_min}")
        if row["r_mean"] < thresholds.decision_r_mean_min:
            reasons.append(f"r_mean<{thresholds.decision_r_mean_min}")
        if row["winrate"] < thresholds.decision_winrate_min:
            reasons.append(f"winrate<{thresholds.decision_winrate_min}")
        if thresholds.decision_p10_min is not None and row["p10"] < thresholds.decision_p10_min:
            reasons.append(f"p10<{thresholds.decision_p10_min}")
        decision = "TRADEABLE" if not reasons else "NO_TRADEABLE"
        decisions.append(
            {
                "decision": decision,
                "decision_reason": "|".join(reasons),
            }
        )
    decision_df = pd.DataFrame(decisions)
    summary = pd.concat([summary, decision_df], axis=1)
    summary = summary.sort_values(["state_hat_H1", "margin_bin", "allow_id"]).reset_index(drop=True)
    return summary[columns]


def _session_bucket_series(
    events_index: pd.DatetimeIndex,
    symbol: str,
    df_m5_ctx: pd.DataFrame | None,
) -> pd.Series:
    if df_m5_ctx is not None and "pf_session_bucket" in df_m5_ctx.columns:
        return df_m5_ctx["pf_session_bucket"].reindex(events_index)
    return pd.Series(
        [get_session_bucket(ts, symbol) for ts in events_index],
        index=events_index,
        name="pf_session_bucket",
    )


def _session_conditional_edge_table(
    events_diag: pd.DataFrame,
    thresholds: argparse.Namespace,
) -> pd.DataFrame:
    columns = [
        "family_id",
        "state_hat_H1",
        "margin_bin",
        "pf_session_bucket",
        "n",
        "winrate",
        "r_mean",
        "p10",
        "decision_reason",
    ]
    if events_diag.empty or "pf_session_bucket" not in events_diag.columns:
        return pd.DataFrame(columns=columns)
    grouped = events_diag.groupby(
        ["family_id", "state_hat_H1", "margin_bin", "pf_session_bucket"],
        observed=True,
    )
    summary = grouped.agg(
        n=("win", "size"),
        winrate=("win", "mean"),
        r_mean=("r", "mean"),
        p10=("r", lambda s: s.quantile(0.1)),
    ).reset_index()
    decision_reasons = []
    for _, row in summary.iterrows():
        reasons = []
        if row["n"] < thresholds.decision_n_min:
            reasons.append(f"n<{thresholds.decision_n_min}")
        if row["r_mean"] < thresholds.decision_r_mean_min:
            reasons.append(f"r_mean<{thresholds.decision_r_mean_min}")
        if row["winrate"] < thresholds.decision_winrate_min:
            reasons.append(f"winrate<{thresholds.decision_winrate_min}")
        if thresholds.decision_p10_min is not None and row["p10"] < thresholds.decision_p10_min:
            reasons.append(f"p10<{thresholds.decision_p10_min}")
        decision_reasons.append("|".join(reasons))
    summary["decision_reason"] = decision_reasons
    summary = summary.sort_values(
        ["family_id", "state_hat_H1", "margin_bin", "pf_session_bucket"],
    ).reset_index(drop=True)
    return summary[columns]


def _format_table_block(title: str, df_or_dict: pd.DataFrame | dict) -> str:
    if isinstance(df_or_dict, dict):
        df = pd.DataFrame([df_or_dict])
    else:
        df = df_or_dict
    if df.empty:
        body = "(no rows)"
    else:
        body = df.to_string(index=False)
    return f"{title}\n{body}"


def _apply_fallback_guardrails(events_total_post_meta: int, fallback_min_samples: int) -> bool:
    return events_total_post_meta < fallback_min_samples


def _print_research_summary_block(session_edge: pd.DataFrame) -> None:
    min_n = 150
    min_r_mean = 0.05
    min_winrate = 0.55
    if session_edge.empty:
        qualifying = pd.DataFrame()
    else:
        qualifying = session_edge[
            (session_edge["n"] >= min_n)
            & (session_edge["r_mean"] >= min_r_mean)
            & (session_edge["winrate"] >= min_winrate)
        ]
    qualifying_count = int(len(qualifying))
    if session_edge.empty:
        top_rows = pd.DataFrame()
    else:
        top_rows = (
            session_edge.sort_values(
                ["r_mean", "winrate", "n"],
                ascending=[False, False, False],
            )
            .head(5)
            .reset_index(drop=True)
        )
    print("\n=== SYMBOL SPECIALIZATION RESEARCH SUMMARY ===")
    print(
        "qualified_session_buckets={count} (n>={n_min}, r_mean>={r_min}, winrate>={w_min})".format(
            count=qualifying_count,
            n_min=min_n,
            r_min=min_r_mean,
            w_min=min_winrate,
        )
    )
    print("top_5_candidate_cells_by_r_mean:")
    if top_rows.empty:
        print("(no candidates)")
    else:
        for _, row in top_rows.iterrows():
            cell = (
                f"{row['family_id']} | {row['state_hat_H1']} | {row['margin_bin']} | {row['pf_session_bucket']}"
            )
            print(cell)
    verdict = (
        "LOCAL EDGE DETECTED — CANDIDATE FOR PROD SPECIALIZATION"
        if qualifying_count > 0
        else "NO LOCAL EDGE DETECTED"
    )
    print(f"verdict={verdict}")


def _save_model_if_ready(
    is_fallback: bool,
    scorer: EventScorerBundle,
    path: Path,
    metadata: dict[str, object],
) -> bool:
    if is_fallback:
        return False
    scorer.save(path, metadata=metadata)
    return True


def _build_allow_id(events_df: pd.DataFrame, allow_cols: list[str]) -> pd.Series:
    if not allow_cols:
        return pd.Series("ALLOW_none", index=events_df.index)
    allow_active = events_df[allow_cols].fillna(0).astype(int)
    allow_id = allow_active.apply(
        lambda row: ",".join(sorted([col for col, val in row.items() if val == 1])) or "ALLOW_none",
        axis=1,
    )
    return allow_id


def _redistribution_table(events_diag: pd.DataFrame, feature_col: str) -> pd.DataFrame:
    columns = ["family_id", feature_col, "n", "pct_of_family"]
    if events_diag.empty or feature_col not in events_diag.columns:
        return pd.DataFrame(columns=columns)
    grouped = (
        events_diag.groupby(["family_id", feature_col], observed=True)
        .size()
        .reset_index(name="n")
    )
    grouped["pct_of_family"] = (
        grouped["n"] / grouped.groupby("family_id")["n"].transform("sum") * 100.0
    ).round(4)
    grouped = grouped.sort_values(["family_id", feature_col]).reset_index(drop=True)
    return grouped[columns]


def _conditional_edge_table(events_diag: pd.DataFrame, feature_col: str) -> pd.DataFrame:
    columns = ["family_id", feature_col, "n", "winrate", "r_mean", "p10", "p90"]
    if events_diag.empty or feature_col not in events_diag.columns:
        return pd.DataFrame(columns=columns)
    grouped = events_diag.groupby(["family_id", feature_col], observed=True)
    summary = grouped.agg(
        n=("win", "size"),
        winrate=("win", "mean"),
        r_mean=("r", "mean"),
        p10=("r", lambda s: s.quantile(0.1)),
        p90=("r", lambda s: s.quantile(0.9)),
    ).reset_index()
    summary = summary.sort_values(["family_id", feature_col]).reset_index(drop=True)
    return summary[columns]


def _pseudo_temporal_splits() -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    return [
        ("2025H1", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-06-30 23:59:59")),
        ("2025H2", pd.Timestamp("2025-07-01"), pd.Timestamp("2025-12-31 23:59:59")),
        ("2026YTD", pd.Timestamp("2026-01-01"), pd.Timestamp.max),
    ]


def _pseudo_stability_table(events_diag: pd.DataFrame, feature_col: str) -> pd.DataFrame:
    columns = ["split", "family_id", feature_col, "n", "winrate", "r_mean", "p10", "p90"]
    if events_diag.empty or feature_col not in events_diag.columns:
        return pd.DataFrame(columns=columns)
    rows: list[pd.DataFrame] = []
    for split_name, start_ts, end_ts in _pseudo_temporal_splits():
        mask = (events_diag.index >= start_ts) & (events_diag.index <= end_ts)
        split_events = events_diag.loc[mask]
        if split_events.empty:
            continue
        grouped = split_events.groupby(["family_id", feature_col], observed=True)
        summary = grouped.agg(
            n=("win", "size"),
            winrate=("win", "mean"),
            r_mean=("r", "mean"),
            p10=("r", lambda s: s.quantile(0.1)),
            p90=("r", lambda s: s.quantile(0.9)),
        ).reset_index()
        summary.insert(0, "split", split_name)
        rows.append(summary)
    if not rows:
        return pd.DataFrame(columns=columns)
    table = pd.concat(rows, ignore_index=True)
    table = table.sort_values(["split", "family_id", feature_col]).reset_index(drop=True)
    return table[columns]


def _session_bucket_distribution(df_m5_ctx: pd.DataFrame) -> pd.DataFrame:
    if df_m5_ctx.empty:
        return pd.DataFrame(columns=["symbol", "pf_session_bucket", "n", "pct"])
    df = df_m5_ctx.copy()
    if "pf_session_bucket" not in df.columns:
        symbol_series = df.get("symbol", pd.Series("UNKNOWN", index=df.index))
        df["pf_session_bucket"] = [
            get_session_bucket(ts, symbol)
            for ts, symbol in zip(df.index, symbol_series)
        ]
    if "symbol" not in df.columns:
        df["symbol"] = "UNKNOWN"
    counts = df.groupby(["symbol", "pf_session_bucket"]).size().reset_index(name="n")
    counts["pct"] = (
        counts["n"] / counts.groupby("symbol")["n"].transform("sum") * 100.0
    ).round(2)
    counts = counts.sort_values(["symbol", "n"], ascending=[True, False]).reset_index(drop=True)
    return counts[["symbol", "pf_session_bucket", "n", "pct"]]


def _merge_asof_features(m5_index: pd.DatetimeIndex, features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame(index=m5_index)
    if getattr(m5_index, "tz", None) is not None:
        m5_index = m5_index.tz_localize(None)
    m5_df = pd.DataFrame({"time": m5_index})
    feats = features.copy()
    if getattr(feats.index, "tz", None) is not None:
        feats.index = feats.index.tz_localize(None)
    feats = feats.reset_index().rename(columns={feats.index.name or "index": "time"})
    merged = pd.merge_asof(m5_df, feats.sort_values("time"), on="time", direction="backward")
    return merged.set_index("time")


def _build_research_context_features(
    df_m5_ctx: pd.DataFrame,
    ohlcv_h1: pd.DataFrame,
    symbol: str,
    research_cfg: dict,
) -> pd.DataFrame:
    feature_flags = research_cfg.get("features", {}) if isinstance(research_cfg.get("features"), dict) else {}
    logger = logging.getLogger("event_scorer")
    raw_anchor_hour = research_cfg.get("d1_anchor_hour", 0)
    try:
        parsed_anchor_hour = int(raw_anchor_hour)
    except (TypeError, ValueError):
        parsed_anchor_hour = 0
        logger.warning(
            "Invalid research.d1_anchor_hour=%r; defaulting to 0 and normalizing to [0, 23].",
            raw_anchor_hour,
        )
    else:
        if (
            isinstance(raw_anchor_hour, bool)
            or not isinstance(raw_anchor_hour, (int, np.integer))
            or not 0 <= raw_anchor_hour <= 23
        ):
            logger.warning(
                "Invalid research.d1_anchor_hour=%r; normalizing to [0, 23] via modulo.",
                raw_anchor_hour,
            )
    d1_anchor_hour = parsed_anchor_hour % 24
    parts: list[pd.DataFrame] = []
    index = df_m5_ctx.index

    if feature_flags.get("session_bucket"):
        session_series = pd.Series(
            [get_session_bucket(ts, symbol) for ts in index],
            index=index,
            name="pf_session_bucket_research",
        )
        session_one_hot = pd.get_dummies(session_series, prefix="pf_session_bucket_research")
        for bucket in SESSION_BUCKETS:
            col = f"pf_session_bucket_research_{bucket}"
            if col not in session_one_hot.columns:
                session_one_hot[col] = 0
        parts.append(session_one_hot)

    if feature_flags.get("hour_bucket"):
        hours = pd.Series(index=index, data=index.hour, name="pf_hour_bucket")
        hour_one_hot = pd.get_dummies(hours, prefix="pf_hour_bucket")
        for hour in range(24):
            col = f"pf_hour_bucket_{hour}"
            if col not in hour_one_hot.columns:
                hour_one_hot[col] = 0
        parts.append(hour_one_hot)

    needs_d1 = feature_flags.get("trend_context_D1") or feature_flags.get("vol_context")
    if needs_d1 and not ohlcv_h1.empty:
        h1 = ohlcv_h1.copy()
        if getattr(h1.index, "tz", None) is not None:
            h1.index = h1.index.tz_localize(None)
        if not h1.index.empty:
            daily_min = h1.index.to_series().groupby(h1.index.normalize()).min()
            aligned_ratio = float((daily_min.dt.hour == (d1_anchor_hour % 24)).mean())
            if aligned_ratio < 0.8:
                logger.warning(
                    "Research D1 resample anchor may be misaligned (anchor_hour=%s, aligned_days=%.1f%%). "
                    "Consider setting research.d1_anchor_hour or verifying server TZ.",
                    d1_anchor_hour,
                    aligned_ratio * 100.0,
                )
        if d1_anchor_hour:
            h1 = h1.copy()
            h1.index = h1.index - pd.Timedelta(hours=d1_anchor_hour)
        h1_atr = _atr(h1["high"], h1["low"], h1["close"], 14).rename("atr_h1")
        h1_features = pd.DataFrame(
            {
                "h1_close": h1["close"],
                "atr_h1": h1_atr,
            },
            index=h1.index,
        ).shift(1)
        h1_merged = _merge_asof_features(index, h1_features)

        d1 = (
            h1[["open", "high", "low", "close"]]
            .resample("1D")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
            .dropna()
        )
        if d1_anchor_hour:
            d1 = d1.copy()
            d1.index = d1.index + pd.Timedelta(hours=d1_anchor_hour)
        d1_ema200 = d1["close"].ewm(span=200, min_periods=200).mean().rename("ema200")
        d1_atr = _atr(d1["high"], d1["low"], d1["close"], 14).rename("atr_d1")
        d1_features = pd.DataFrame(
            {
                "d1_close": d1["close"],
                "d1_high": d1["high"],
                "d1_low": d1["low"],
                "d1_ema200": d1_ema200,
                "atr_d1": d1_atr,
            },
            index=d1.index,
        ).shift(1)
        d1_merged = _merge_asof_features(index, d1_features)

        if feature_flags.get("trend_context_D1"):
            ema_dist = (d1_merged["d1_close"] - d1_merged["d1_ema200"]) / d1_merged["atr_d1"].replace(0, np.nan)
            ema_side = np.where(ema_dist >= 0, "ABOVE", "BELOW")
            ema_side = pd.Series(ema_side, index=index).where(ema_dist.notna(), other="UNKNOWN")
            ema_side_one_hot = pd.get_dummies(ema_side, prefix="pf_d1_ema200_side")
            for side in ["ABOVE", "BELOW", "UNKNOWN"]:
                col = f"pf_d1_ema200_side_{side}"
                if col not in ema_side_one_hot.columns:
                    ema_side_one_hot[col] = 0
            dist_to_high = (d1_merged["d1_high"] - df_m5_ctx["close"]) / d1_merged["atr_d1"].replace(0, np.nan)
            dist_to_low = (df_m5_ctx["close"] - d1_merged["d1_low"]) / d1_merged["atr_d1"].replace(0, np.nan)
            trend_features = pd.DataFrame(
                {
                    "pf_d1_ema200_dist": ema_dist,
                    "pf_d1_dist_to_high_atr": dist_to_high,
                    "pf_d1_dist_to_low_atr": dist_to_low,
                },
                index=index,
            )
            parts.append(pd.concat([trend_features, ema_side_one_hot], axis=1))

        if feature_flags.get("vol_context"):
            atr_h1_norm = h1_merged["atr_h1"] / h1_merged["h1_close"].replace(0, np.nan)
            atr_h1_to_d1 = h1_merged["atr_h1"] / d1_merged["atr_d1"].replace(0, np.nan)
            vol_features = pd.DataFrame(
                {
                    "pf_atr_h1_norm": atr_h1_norm,
                    "pf_atr_h1_to_d1": atr_h1_to_d1,
                },
                index=index,
            )
            parts.append(vol_features)

    if not parts:
        return pd.DataFrame(index=index)
    return pd.concat(parts, axis=1)


def _score_shape_diagnostics(
    scores: pd.Series,
    families: pd.Series,
    timestamps: pd.DatetimeIndex,
    k_values: list[int],
) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame(
            columns=[
                "k",
                "edge_decay",
                "temporal_dispersion",
                "family_concentration",
                "score_tail_slope",
            ]
        )

    def _edge_decay(top_scores: pd.Series) -> float:
        if top_scores.empty:
            return float("nan")
        top_sorted = top_scores.sort_values(ascending=False)
        top_first = float(top_sorted.iloc[0])
        top_last = float(top_sorted.iloc[-1])
        if top_first == 0:
            return float("nan")
        return (top_first - top_last) / abs(top_first)

    unique_days_total = timestamps.normalize().nunique() if len(timestamps) else 0
    p90 = float(scores.quantile(0.9))
    p99 = float(scores.quantile(0.99))
    score_tail_slope = (p99 - p90) / 0.09 if not np.isnan(p90) and not np.isnan(p99) else float("nan")

    rows = []
    for k in k_values:
        k_eff = min(k, len(scores))
        top_idx = scores.nlargest(k_eff).index
        top_scores = scores.loc[top_idx]
        top_families = families.loc[top_idx]
        top_days = top_idx.normalize().nunique() if len(top_idx) else 0
        temporal_dispersion = (top_days / unique_days_total) if unique_days_total else float("nan")
        family_counts = top_families.value_counts()
        family_concentration = (
            float(family_counts.max()) / k_eff if k_eff and not family_counts.empty else float("nan")
        )
        rows.append(
            {
                "k": k,
                "edge_decay": _edge_decay(top_scores),
                "temporal_dispersion": temporal_dispersion,
                "family_concentration": family_concentration,
                "score_tail_slope": score_tail_slope,
            }
        )
    return pd.DataFrame(rows)


def _research_guardrails(
    score_shape: pd.DataFrame,
    train_metrics: dict[str, float] | None,
    calib_metrics: dict[str, float] | None,
    diagnostics_cfg: dict,
    calib_samples: int,
) -> pd.DataFrame:
    if score_shape.empty:
        return pd.DataFrame(columns=["k", "status", "reasons"])
    max_family_concentration = diagnostics_cfg.get("max_family_concentration", 0.6)
    min_temporal_dispersion = diagnostics_cfg.get("min_temporal_dispersion", 0.3)
    max_score_tail_slope = diagnostics_cfg.get("max_score_tail_slope", 2.5)
    min_calib_samples = diagnostics_cfg.get("min_calib_samples", 200)
    train_lift = train_metrics.get("lift@20") if train_metrics else None
    calib_lift = calib_metrics.get("lift@20") if calib_metrics else None
    rows = []

    def _has_reason(reasons_list: list[str], prefix: str) -> bool:
        return any(reason.startswith(prefix) for reason in reasons_list)

    for _, row in score_shape.iterrows():
        reasons = []
        if calib_samples < min_calib_samples:
            reasons.append(f"LOW_CALIB_SAMPLES<{min_calib_samples}")
        if pd.notna(row["family_concentration"]) and row["family_concentration"] > max_family_concentration:
            reasons.append("FAMILY_CONCENTRATION_HIGH")
        if pd.notna(row["temporal_dispersion"]) and row["temporal_dispersion"] < min_temporal_dispersion:
            reasons.append("TEMPORAL_DISPERSION_LOW")
        if pd.notna(row["score_tail_slope"]) and row["score_tail_slope"] > max_score_tail_slope:
            reasons.append("SCORE_TAIL_STEEP")
        if train_lift is not None and calib_lift is not None:
            if (train_lift - calib_lift) > 0.2 and calib_lift < 1.05:
                reasons.append("TRAIN_IMPROVEMENT_ONLY")
        status = "RESEARCH_OK"
        if any(reason in {"SCORE_TAIL_STEEP", "TRAIN_IMPROVEMENT_ONLY"} for reason in reasons):
            status = "RESEARCH_OVERFIT_SUSPECT"
        elif _has_reason(reasons, "LOW_CALIB_SAMPLES") or any(
            reason in {"FAMILY_CONCENTRATION_HIGH", "TEMPORAL_DISPERSION_LOW"} for reason in reasons
        ):
            status = "RESEARCH_UNSTABLE"
        rows.append({"k": int(row["k"]), "status": status, "reasons": "|".join(reasons)})
    return pd.DataFrame(rows)


def _build_training_diagnostic_report(
    events_diag: pd.DataFrame,
    df_m5_ctx: pd.DataFrame | None,
    thresholds: argparse.Namespace,
) -> dict[str, pd.DataFrame]:
    report = {}
    coverage_global, coverage_by = _coverage_table(events_diag, df_m5_ctx)
    regime_full, regime_ranked = _regime_edge_table(events_diag)
    stability = _stability_table(events_diag)
    decision = _decision_table(events_diag, thresholds)
    session_edge = _session_conditional_edge_table(events_diag, thresholds)
    base_frames = [coverage_global, coverage_by, regime_full, regime_ranked, stability, decision, session_edge]
    report["coverage_global"] = coverage_global
    report["coverage_by_state_margin"] = coverage_by
    report["regime_edge_full"] = regime_full
    report["regime_edge_ranked"] = regime_ranked
    report["stability"] = stability
    report["decision"] = decision
    report["session_conditional_edge"] = session_edge
    for pseudo_col in PSEUDO_FEATURES.keys():
        redistribution = _redistribution_table(events_diag, pseudo_col)
        conditional = _conditional_edge_table(events_diag, pseudo_col)
        stability_pseudo = _pseudo_stability_table(events_diag, pseudo_col)
        conditional_regime = _conditional_edge_regime_table(
            events_diag,
            pseudo_col,
            min_n=thresholds.decision_n_min,
        )
        report[f"conditional_edge_regime_{pseudo_col}"] = conditional_regime
        base_frames.append(conditional_regime)
        report[f"redistribution_{pseudo_col}"] = redistribution
        report[f"conditional_edge_{pseudo_col}"] = conditional
        report[f"stability_{pseudo_col}"] = stability_pseudo
        base_frames.extend([redistribution, conditional, stability_pseudo])
    for frame in base_frames:
        numeric_cols = frame.select_dtypes(include=[np.number]).columns
        frame[numeric_cols] = frame[numeric_cols].round(4)
    print("=== TRAINING DIAGNOSTIC REPORT ===")
    print(_format_table_block("Coverage (global)", coverage_global))
    print(_format_table_block("Coverage (by state x margin_bin)", coverage_by))
    print(_format_table_block("Regime Edge (full)", regime_full))
    print(_format_table_block("Regime Edge (top 10 / bottom 10 by r_mean)", regime_ranked))
    print(_format_table_block("Stability temporal (regime edge por split)", stability))
    print(_format_table_block("Decision", decision))
    print(_format_table_block("Session-Conditional Edge", session_edge))
    for pseudo_col in PSEUDO_FEATURES.keys():
        print(_format_table_block(f"Redistribution (family x {pseudo_col})", report[f"redistribution_{pseudo_col}"]))
        print(_format_table_block(f"Conditional Edge (family x {pseudo_col})", report[f"conditional_edge_{pseudo_col}"]))
        print(_format_table_block(f"Stability temporal (family x {pseudo_col})", report[f"stability_{pseudo_col}"]))
        print(_format_table_block(f"Conditional Edge REGIME (state x margin x allow x family x {pseudo_col})", report[f"conditional_edge_regime_{pseudo_col}"]))
    if df_m5_ctx is not None:
        session_table = _session_bucket_distribution(df_m5_ctx)
        print(_format_table_block("Session Bucket Distribution (M5)", session_table))
    return report


def _run_training_for_k(
    args: argparse.Namespace,
    k_bars: int,
    output_suffix: str,
    detected_events: pd.DataFrame,
    event_config: EventDetectionConfig,
    df_m5_ctx: pd.DataFrame,
    ohlcv_m5: pd.DataFrame,
    features_all: pd.DataFrame,
    atr_ratio: pd.Series,
    feature_builder: FeatureBuilder,
    m5_total: int,
    m5_ctx_merged: int,
    m5_ctx_dropna: int,
    h1_cutoff: pd.Timestamp,
    m5_cutoff: pd.Timestamp,
    min_samples_train: int,
    seed: int,
    scorer_out_base: Path,
    research_enabled: bool,
    research_cfg: dict,
) -> None:
    logger = logging.getLogger("event_scorer")
    print(
        "symbol={symbol} start={start} end={end} k_bars={k} reward_r={reward} sl_mult={sl} r_thr={thr} "
        "meta_policy={meta} include_transition={include_transition}".format(
            symbol=args.symbol,
            start=args.start,
            end=args.end,
            k=k_bars,
            reward=args.reward_r,
            sl=args.sl_mult,
            thr=args.r_thr,
            meta=args.meta_policy,
            include_transition=args.include_transition,
        )
    )

    events = label_events(
        detected_events.copy(),
        ohlcv_m5,
        k_bars,
        args.reward_r,
        args.sl_mult,
        r_thr=args.r_thr,
        tie_break=args.tie_break,
    )
    labeled_events = events.copy()
    labeled_total = len(labeled_events)
    dropna_mask = labeled_events[["label", "r_outcome"]].isna().any(axis=1)
    dropna_count = int(dropna_mask.sum())
    events = labeled_events.dropna(subset=["label", "r_outcome"])
    events = events.sort_index()

    if events.empty:
        logger.warning("No labeled events after filtering.")
        events_diag = pd.DataFrame(
            columns=["state_hat_H1", "allow_id", "margin", "r", "win", "margin_bin"]
        )
        events_diag.index = pd.DatetimeIndex([])
        is_fallback = _apply_fallback_guardrails(0, args.fallback_min_samples)
        if is_fallback:
            print(
                "=== FALLBACK DIAGNÓSTICO: events_total_post_meta=0 < fallback_min_samples="
                f"{args.fallback_min_samples} ==="
            )
        diagnostic_report = _build_training_diagnostic_report(events_diag, df_m5_ctx, thresholds=args)
        if args.mode == "research":
            _print_research_summary_block(diagnostic_report.get("session_conditional_edge", pd.DataFrame()))
        return
    logger.info("Labeled events by family:\n%s", events["family_id"].value_counts().to_string())
    logger.info("Labeled events by type:\n%s", events["event_type"].value_counts().to_string())

    events_all = events.copy()
    allow_cols = [col for col in events_all.columns if col.startswith("ALLOW_")]
    if not allow_cols:
        logger.warning("No ALLOW_* columns found on events; meta policy will be empty.")
    margin_bins = _margin_bins(events_all["margin_H1"], q=3)
    margin_bin_label = margin_bins.map(_format_interval)
    state_label = events_all["state_hat_H1"].map(_state_label)
    events_all["state_label"] = state_label
    events_all["margin_bin"] = margin_bin_label
    events_all["allow_id"] = _build_allow_id(events_all, allow_cols)
    events_all["regime_id"] = (
        state_label.astype(str)
        + "|"
        + margin_bin_label.fillna("mNA")
        + "|"
        + events_all["allow_id"]
        + "|"
        + events_all["family_id"].astype(str)
        + "|"
        + events_all["event_type"].astype(str)
    )

    include_transition = args.include_transition == "on"
    events_state_filtered = (
        events_all if include_transition else events_all.loc[events_all["state_label"] != "TRANSITION"]
    )
    if not include_transition:
        logger.info(
            "STATE FILTER | include_transition=off removed=%s kept=%s",
            len(events_all) - len(events_state_filtered),
            len(events_state_filtered),
        )

    event_features_all = features_all.reindex(events_state_filtered.index)
    pseudo_features = _build_pseudo_features(events_state_filtered, event_features_all, atr_ratio)
    events_state_filtered = events_state_filtered.join(pseudo_features)
    pseudo_encoded = _encode_pseudo_features(pseudo_features)
    event_features_all = pd.concat([event_features_all, pseudo_encoded], axis=1)

    print("\n[STATE MIX | EVENTS POST FILTER]")
    state_counts_events = events_state_filtered["state_label"].value_counts()
    for line in _format_state_mix(state_counts_events, ["BALANCE", "TREND", "TRANSITION"]):
        print(line)

    vwap_mode = _resolve_vwap_report_mode(detected_events, df_m5_ctx, event_config)
    dist_abs = events_all["dist_to_vwap_atr"].abs()
    vwap_quantiles = dist_abs.quantile([0.1, 0.5, 0.9, 0.99]).to_dict() if not dist_abs.empty else {}
    print("\n[VWAP SANITY]")
    print(f"vwap_mode={vwap_mode} near_vwap_atr={event_config.near_vwap_atr}")
    if vwap_quantiles:
        print("dist_to_vwap_atr_abs_q=" + str({k: round(v, 4) for k, v in vwap_quantiles.items()}))

    print("\n[BASELINE BY STATE | POST FILTER]")
    for state_name in ["BALANCE", "TREND", "TRANSITION"]:
        state_mask = events_state_filtered["state_label"] == state_name
        state_events = events_state_filtered.loc[state_mask]
        trades = len(state_events)
        winrate = float(state_events["label"].mean()) if trades else float("nan")
        avg_pnl = float(state_events["r_outcome"].mean()) if trades else float("nan")
        total_pnl = float(state_events["r_outcome"].sum()) if trades else float("nan")
        print(
            f"{state_name}: trades={trades}, winrate={_format_metric(winrate)}, "
            f"avg_pnl={_format_metric(avg_pnl)}, total_pnl={_format_metric(total_pnl)}"
        )

    meta_policy_on = args.meta_policy == "on"
    allow_active_series = (
        events_state_filtered[allow_cols].fillna(0).sum(axis=1) > 0
        if allow_cols
        else pd.Series(False, index=events_state_filtered.index)
    )
    allow_active_pct = float(allow_active_series.mean() * 100) if len(allow_active_series) else 0.0
    allow_id_top = events_state_filtered["allow_id"].value_counts().head(10)
    print("\n[ALLOW SANITY]")
    print(f"allow_active_pct={allow_active_pct:.2f}%")
    if not allow_id_top.empty:
        print("allow_id_top:\n" + allow_id_top.to_string())
    margin_ok_series = events_state_filtered["margin_H1"].between(
        args.meta_margin_min,
        args.meta_margin_max,
        inclusive="both",
    )
    logger.info(
        "INFO | META | allow_active_count=%s margin_ok_count=%s",
        int(allow_active_series.sum()),
        int(margin_ok_series.sum()),
    )

    meta_mask = _meta_policy_mask(
        events_state_filtered,
        allow_cols,
        args.meta_margin_min,
        args.meta_margin_max,
    )
    events_meta = events_state_filtered.loc[meta_mask].copy()

    if meta_policy_on and events_meta.empty:
        logger.warning("Meta policy filtered all events; continuing with fallback diagnostics.")

    if meta_policy_on:
        kept_pct = (
            (len(events_meta) / len(events_state_filtered)) * 100 if len(events_state_filtered) else 0.0
        )
        logger.info(
            "INFO | META | before_rows=%s after_rows=%s kept_pct=%.2f",
            len(events_state_filtered),
            len(events_meta),
            kept_pct,
        )

        def _mix_table(series: pd.Series, title: str) -> pd.DataFrame:
            before_counts = series.value_counts(dropna=False)
            after_counts = series.loc[events_meta.index].value_counts(dropna=False)
            table = pd.DataFrame(
                {
                    "id": before_counts.index,
                    "before_count": before_counts.values,
                    "before_pct": (before_counts / before_counts.sum()).mul(100).round(2).values,
                }
            )
            table["after_count"] = table["id"].map(after_counts).fillna(0).astype(int)
            table["after_pct"] = (table["after_count"] / max(after_counts.sum(), 1)).mul(100).round(2)
            logger.info("INFO | META | %s before/after:\n%s", title, table.to_string(index=False))
            return table

        _mix_table(events_state_filtered["family_id"], "family mix")
        _mix_table(events_state_filtered["state_label"], "state mix")
        _mix_table(events_state_filtered["margin_bin"].fillna("mNA"), "margin bins")

    events_for_training = events_meta if meta_policy_on else events_state_filtered
    events = events_for_training
    print("\n[SUPPLY FUNNEL]")
    print(f"m5_total={m5_total}")
    print(f"after_merge={m5_ctx_merged}")
    print(f"after_ctx_dropna={m5_ctx_dropna}")
    print(f"events_detected={len(detected_events)}")
    print(f"events_labeled={labeled_total}")
    print(f"events_post_state_filter={len(events_state_filtered)}")
    print(f"events_post_meta={len(events_for_training)}")
    required_columns = {"state_label", "allow_id", "margin_H1", "r_outcome", "label"}
    missing_columns = sorted(required_columns - set(events_for_training.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns in events_for_training: {missing_columns}")
    if not isinstance(events_for_training.index, pd.DatetimeIndex):
        raise ValueError("events_for_training index must be DatetimeIndex")
    events_diag = pd.DataFrame(
        {
            "state_hat_H1": events_for_training["state_label"],
            "allow_id": events_for_training["allow_id"],
            "margin": events_for_training["margin_H1"],
            "r": events_for_training["r_outcome"],
            "win": events_for_training["label"].astype(int),
            "family_id": events_for_training["family_id"],
            "vwap_zone": events_for_training["vwap_zone"],
            "value_state": events_for_training["value_state"],
            "expansion_state": events_for_training["expansion_state"],
            "pf_session_bucket": _session_bucket_series(
                events_for_training.index,
                args.symbol,
                df_m5_ctx,
            ),
        },
        index=events_for_training.index,
    )
    events_diag["margin_bin"] = _make_margin_bin_rank(events_diag["margin"])
    events_total_post_meta = len(events_for_training)
    fallback_reasons = []
    if _apply_fallback_guardrails(events_total_post_meta, args.fallback_min_samples):
        fallback_reasons.append(f"events<{args.fallback_min_samples}")
    if events_for_training["label"].nunique() < 2:
        fallback_reasons.append("single_class_labels")
    is_fallback = bool(fallback_reasons)
    if is_fallback:
        print(
            "=== FALLBACK DIAGNÓSTICO: events_total_post_meta="
            f"{events_total_post_meta} reasons={','.join(fallback_reasons)} ==="
        )
    diagnostic_report = _build_training_diagnostic_report(events_diag, df_m5_ctx, thresholds=args)

    if is_fallback:
        args.model_dir.mkdir(parents=True, exist_ok=True)
        detected_path = _with_suffix(
            args.model_dir / f"events_detected_{''.join(ch if (ch.isalnum() or ch in '._-') else '_' for ch in args.symbol)}.csv",
            output_suffix,
        )
        labeled_path = _with_suffix(
            args.model_dir / f"events_labeled_{''.join(ch if (ch.isalnum() or ch in '._-') else '_' for ch in args.symbol)}.csv",
            output_suffix,
        )
        _reset_index_for_export(detected_events).to_csv(detected_path, index=False)
        _reset_index_for_export(labeled_events).to_csv(labeled_path, index=False)
        logger.info("fallback_events_detected_out=%s", detected_path)
        logger.info("fallback_events_labeled_out=%s", labeled_path)
        if not events_for_training.empty:
            r_values = events_for_training["r_outcome"]
            summary = {
                "base_rate": float(events_for_training["label"].mean()),
                "winrate": float(events_for_training["label"].mean()),
                "r_mean_all": float(r_values.mean()),
                "p10": float(r_values.quantile(0.1)),
                "p90": float(r_values.quantile(0.9)),
            }
            print("\n[FALLBACK METRICS]")
            print(pd.DataFrame([summary]).to_string(index=False))
        return

    event_features_all = feature_builder.add_family_features(
        event_features_all,
        events_state_filtered["family_id"],
    )

    def _split_dataset(events_df: pd.DataFrame, feature_df: pd.DataFrame) -> dict[str, pd.Series | pd.DataFrame]:
        labels_series = events_df["label"].astype(int)
        split_idx = int(len(events_df) * args.train_ratio)
        return {
            "X_train": feature_df.loc[events_df.index].iloc[:split_idx],
            "y_train": labels_series.iloc[:split_idx],
            "X_calib": feature_df.loc[events_df.index].iloc[split_idx:],
            "y_calib": labels_series.iloc[split_idx:],
            "r_train": events_df["r_outcome"].iloc[:split_idx],
            "r_calib": events_df["r_outcome"].iloc[split_idx:],
            "fam_train": events_df["family_id"].iloc[:split_idx],
            "fam_calib": events_df["family_id"].iloc[split_idx:],
            "regime_calib": events_df["regime_id"].iloc[split_idx:],
        }

    dataset_main = _split_dataset(events_for_training, event_features_all)
    dataset_no_meta = _split_dataset(events_state_filtered, event_features_all)

    X_train = dataset_main["X_train"]
    y_train = dataset_main["y_train"]
    X_calib = dataset_main["X_calib"]
    y_calib = dataset_main["y_calib"]
    r_train = dataset_main["r_train"]
    r_calib = dataset_main["r_calib"]
    fam_train = dataset_main["fam_train"]
    fam_calib = dataset_main["fam_calib"]
    regime_calib = dataset_main["regime_calib"]

    X_calib_no_meta = dataset_no_meta["X_calib"]
    y_calib_no_meta = dataset_no_meta["y_calib"]
    r_calib_no_meta = dataset_no_meta["r_calib"]
    fam_calib_no_meta = dataset_no_meta["fam_calib"]
    regime_calib_no_meta = dataset_no_meta["regime_calib"]

    transition_events_present = bool((events_state_filtered["state_label"] == "TRANSITION").any())
    transition_samples_train = int((events_for_training.loc[X_train.index, "state_label"] == "TRANSITION").sum())
    transition_samples_calib = int((events_for_training.loc[X_calib.index, "state_label"] == "TRANSITION").sum())

    family_summary = pd.DataFrame(
        {
            "family_id": sorted(set(events_for_training["family_id"])),
        }
    )
    family_summary["samples_train"] = family_summary["family_id"].map(fam_train.value_counts()).fillna(0).astype(int)
    family_summary["samples_calib"] = family_summary["family_id"].map(fam_calib.value_counts()).fillna(0).astype(int)
    family_summary["base_rate_train"] = family_summary["family_id"].map(y_train.groupby(fam_train).mean())
    family_summary["base_rate_calib"] = family_summary["family_id"].map(y_calib.groupby(fam_calib).mean())
    logger.info("Family counts:\n%s", family_summary.to_string(index=False))

    if y_train.nunique() < 2:
        logger.error("Global training labels have a single class; cannot train scorer.")
        args.model_dir.mkdir(parents=True, exist_ok=True)
        detected_path = _with_suffix(
            args.model_dir / f"events_detected_{''.join(ch if (ch.isalnum() or ch in '._-') else '_' for ch in args.symbol)}.csv",
            output_suffix,
        )
        labeled_path = _with_suffix(
            args.model_dir / f"events_labeled_{''.join(ch if (ch.isalnum() or ch in '._-') else '_' for ch in args.symbol)}.csv",
            output_suffix,
        )
        _reset_index_for_export(detected_events).to_csv(detected_path, index=False)
        _reset_index_for_export(labeled_events).to_csv(labeled_path, index=False)
        logger.info("fallback_events_detected_out=%s", detected_path)
        logger.info("fallback_events_labeled_out=%s", labeled_path)
        if not events_for_training.empty:
            r_values = events_for_training["r_outcome"]
            summary = {
                "base_rate": float(events_for_training["label"].mean()),
                "winrate": float(events_for_training["label"].mean()),
                "r_mean_all": float(r_values.mean()),
                "p10": float(r_values.quantile(0.1)),
                "p90": float(r_values.quantile(0.9)),
            }
            print("\n[FALLBACK METRICS]")
            print(pd.DataFrame([summary]).to_string(index=False))
        return

    warning_summary_rows: list[dict[str, int | str | float]] = []

    def _fit_with_warning_capture(
        scope: str,
        scorer_model: EventScorer,
        train_x: pd.DataFrame,
        train_y: pd.Series,
        calib_x: pd.DataFrame | None,
        calib_y: pd.Series | None,
    ) -> None:
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            scorer_model.fit(train_x, train_y, calib_x, calib_y)
        combined = stdout_buffer.getvalue() + stderr_buffer.getvalue()
        warning_hits = combined.count("No further splits with positive gain")
        warning_summary_rows.append(
            {
                "scope": scope,
                "n_samples": len(train_y),
                "n_features": train_x.shape[1],
                "base_rate": float(train_y.mean()) if len(train_y) else float("nan"),
                "split_warning_hits": warning_hits,
            }
        )

    scorer = EventScorerBundle(EventScorerConfig())
    global_scorer = EventScorer(scorer.config)
    _fit_with_warning_capture("global", global_scorer, X_train, y_train, X_calib, y_calib)
    scorer.scorers[scorer.global_key] = global_scorer

    family_summary["train_unique"] = family_summary["family_id"].map(y_train.groupby(fam_train).nunique())
    family_summary["calib_unique"] = family_summary["family_id"].map(y_calib.groupby(fam_calib).nunique())
    family_summary["status"] = "TRAINED"

    for family_id in family_summary["family_id"]:
        train_mask = fam_train == family_id
        calib_mask = fam_calib == family_id
        train_count = int(train_mask.sum())
        calib_count = int(calib_mask.sum())
        train_unique = int(family_summary.loc[family_summary["family_id"] == family_id, "train_unique"].fillna(0).iloc[0])
        calib_unique = int(family_summary.loc[family_summary["family_id"] == family_id, "calib_unique"].fillna(0).iloc[0])

        if train_count < min_samples_train:
            family_summary.loc[family_summary["family_id"] == family_id, "status"] = "SKIP_FAMILY_LOW_SAMPLES"
            logger.warning("Skip %s: low samples (%s)", family_id, train_count)
            continue
        if train_unique < 2 or calib_unique < 2:
            family_summary.loc[family_summary["family_id"] == family_id, "status"] = "SKIP_FAMILY_SINGLE_CLASS"
            logger.warning(
                "Skip %s: missing class (train_unique=%s calib_unique=%s)",
                family_id,
                train_unique,
                calib_unique,
            )
            continue

        scorer_family = EventScorer(scorer.config)
        calib_X = X_calib.loc[calib_mask] if calib_count else None
        calib_y = y_calib.loc[calib_mask] if calib_count else None
        _fit_with_warning_capture(
            str(family_id),
            scorer_family,
            X_train.loc[train_mask],
            y_train.loc[train_mask],
            calib_X,
            calib_y,
        )
        scorer.scorers[str(family_id)] = scorer_family

    logger.info("Family training status:\n%s", family_summary.to_string(index=False))

    def _topk_indices(scores: pd.Series, labels_: pd.Series, k: int, seed: int = 7) -> pd.Index:
        if scores.empty:
            return pd.Index([])
        k_eff = min(k, len(scores))
        if k_eff == 0:
            return pd.Index([])
        if scores.nunique(dropna=False) <= 1:
            rng = np.random.default_rng(seed)
            chosen = rng.choice(labels_.index.to_numpy(), size=k_eff, replace=False)
            return pd.Index(chosen)
        return scores.nlargest(k_eff).index

    def _bottomk_indices(scores: pd.Series, labels_: pd.Series, k: int, seed: int = 7) -> pd.Index:
        if scores.empty:
            return pd.Index([])
        k_eff = min(k, len(scores))
        if k_eff == 0:
            return pd.Index([])
        if scores.nunique(dropna=False) <= 1:
            rng = np.random.default_rng(seed)
            chosen = rng.choice(labels_.index.to_numpy(), size=k_eff, replace=False)
            return pd.Index(chosen)
        return scores.nsmallest(k_eff).index

    def precision_at_k(scores: pd.Series, labels_: pd.Series, k: int, seed: int = 7) -> float:
        top_idx = _topk_indices(scores, labels_, k, seed=seed)
        if top_idx.empty:
            return float("nan")
        return float(labels_.loc[top_idx].mean())

    def lift_at_k(scores: pd.Series, labels_: pd.Series, k: int, seed: int = 7) -> float:
        base_rate = float(labels_.mean())
        if base_rate == 0 or scores.empty:
            return float("nan")
        return precision_at_k(scores, labels_, k, seed=seed) / base_rate

    def mean_r_topk(scores: pd.Series, r_outcome: pd.Series, k: int, seed: int = 7) -> float:
        top_idx = _topk_indices(scores, r_outcome, k, seed=seed)
        if top_idx.empty:
            return float("nan")
        return float(r_outcome.loc[top_idx].mean())

    def median_r_topk(scores: pd.Series, r_outcome: pd.Series, k: int, seed: int = 7) -> float:
        top_idx = _topk_indices(scores, r_outcome, k, seed=seed)
        if top_idx.empty:
            return float("nan")
        return float(r_outcome.loc[top_idx].median())

    def summarize_metrics(
        scope: str,
        scores: pd.Series,
        labels_: pd.Series,
        r_outcome: pd.Series,
        seed: int = 7,
    ) -> dict[str, float]:
        base_rate = float(labels_.mean()) if not labels_.empty else float("nan")
        r_mean_all = float(r_outcome.mean()) if not r_outcome.empty else float("nan")
        metrics = {
            "scope": scope,
            "samples": len(labels_),
            "base_rate": base_rate,
            "auc": float("nan"),
            "lift@10": lift_at_k(scores, labels_, 10, seed=seed),
            "lift@20": lift_at_k(scores, labels_, 20, seed=seed),
            "lift@50": lift_at_k(scores, labels_, 50, seed=seed),
            "r_mean@10": mean_r_topk(scores, r_outcome, 10, seed=seed),
            "r_mean@20": mean_r_topk(scores, r_outcome, 20, seed=seed),
            "r_mean@50": mean_r_topk(scores, r_outcome, 50, seed=seed),
            "r_median@10": median_r_topk(scores, r_outcome, 10, seed=seed),
            "r_median@20": median_r_topk(scores, r_outcome, 20, seed=seed),
            "r_median@50": median_r_topk(scores, r_outcome, 50, seed=seed),
            "r_mean_all": r_mean_all,
        }
        if len(labels_.unique()) > 1:
            metrics["auc"] = roc_auc_score(labels_, scores)
        metrics["delta_r_mean@10"] = metrics["r_mean@10"] - r_mean_all
        metrics["delta_r_mean@20"] = metrics["r_mean@20"] - r_mean_all
        metrics["delta_r_mean@50"] = metrics["r_mean@50"] - r_mean_all
        metrics["delta_r_mean@20_neg"] = float(metrics["delta_r_mean@20"] < 0)
        return metrics

    def _global_metrics_table(
        label: str,
        scores: pd.Series,
        labels_: pd.Series,
        r_outcome: pd.Series,
        seed: int = 7,
    ) -> dict[str, float | str]:
        metrics = summarize_metrics(label, scores, labels_, r_outcome, seed=seed)
        spearman = scores.corr(r_outcome, method="spearman") if len(scores) else float("nan")
        return {
            "model": label,
            "auc": metrics["auc"],
            "lift@10": metrics["lift@10"],
            "lift@20": metrics["lift@20"],
            "lift@50": metrics["lift@50"],
            "r_mean@10": metrics["r_mean@10"],
            "r_mean@20": metrics["r_mean@20"],
            "r_mean@50": metrics["r_mean@50"],
            "delta_r_mean@20": metrics["delta_r_mean@20"],
            "spearman": spearman,
        }

    def _metrics_block(
        block_name: str,
        X_block: pd.DataFrame,
        y_block: pd.Series,
        r_block: pd.Series,
        fam_block: pd.Series,
    ) -> tuple[pd.DataFrame, pd.Series | None, pd.Series | None, pd.Series | None]:
        if y_block.empty:
            logger.warning("WARNING | DIAG | Empty calib set for %s metrics", block_name)
            return pd.DataFrame(), None, None, None
        scores_block = scorer.predict_proba(X_block, fam_block)
        rng = np.random.default_rng(seed)
        baseline_scores = pd.Series(rng.random(len(y_block)), index=y_block.index)
        table = pd.DataFrame(
            [
                _global_metrics_table("SCORER", scores_block, y_block, r_block, seed=seed),
                _global_metrics_table("BASELINE", baseline_scores, y_block, r_block, seed=seed),
            ]
        )
        logger.info("INFO | GLOBAL | METRICS (%s)\n%s", block_name, table.to_string(index=False))
        return table, scores_block, baseline_scores, y_block

    if not is_fallback:
        table_no_meta, preds_no_meta, baseline_no_meta, _ = _metrics_block(
            "NO_META",
            X_calib_no_meta,
            y_calib_no_meta,
            r_calib_no_meta,
            fam_calib_no_meta,
        )
        if meta_policy_on:
            table_meta, preds_meta, baseline_meta, _ = _metrics_block(
                "META",
                X_calib,
                y_calib,
                r_calib,
                fam_calib,
            )
        else:
            table_meta = table_no_meta.copy()
            preds_meta = preds_no_meta
            baseline_meta = baseline_no_meta
    else:
        table_no_meta = pd.DataFrame()
        table_meta = pd.DataFrame()
        preds_no_meta = None
        preds_meta = None
        baseline_no_meta = None
        baseline_meta = None

    def _print_global_metrics(title: str, table: pd.DataFrame) -> None:
        if table.empty:
            print(f"\n{title}")
            print("AUC=nan, lift@10=nan, lift@20=nan, r_mean@20=nan, spearman=nan")
            return
        scorer_row = table.loc[table["model"] == "SCORER"]
        if scorer_row.empty:
            scorer_row = table.iloc[[0]]
        row = scorer_row.iloc[0]
        print(f"\n{title}")
        print(
            "AUC={auc}, lift@10={lift10}, lift@20={lift20}, r_mean@20={rmean20}, spearman={spear}".format(
                auc=_format_metric(row["auc"]),
                lift10=_format_metric(row["lift@10"]),
                lift20=_format_metric(row["lift@20"]),
                rmean20=_format_metric(row["r_mean@20"]),
                spear=_format_metric(row["spearman"]),
            )
        )

    if not is_fallback:
        _print_global_metrics("[GLOBAL METRICS | NO_META | POST FILTER]", table_no_meta)
        _print_global_metrics("[GLOBAL METRICS | META | POST FILTER]", table_meta)
    print("\n[SANITY]")
    print(f"transition_events_present={transition_events_present}")
    print(f"transition_samples_train={transition_samples_train}")
    print(f"transition_samples_calib={transition_samples_calib}")

    metrics_df = table_meta.copy()
    report_header = {
        "symbol": args.symbol,
        "start": args.start,
        "end": args.end,
        "h1_cutoff": h1_cutoff,
        "m5_cutoff": m5_cutoff,
        "k_bars": k_bars,
        "reward_r": args.reward_r,
        "sl_mult": args.sl_mult,
        "r_thr": args.r_thr,
        "tie_break": args.tie_break,
        "feature_count": event_features_all.shape[1],
        "min_samples_train": min_samples_train,
        "seed": seed,
        "meta_policy": args.meta_policy,
        "meta_margin_min": args.meta_margin_min,
        "meta_margin_max": args.meta_margin_max,
        "include_transition": args.include_transition,
    }
    header_table = pd.DataFrame([report_header])
    logger.info("=" * 96)
    logger.info("EVENT SCORER TRAINING REPORT")
    logger.info("%s", header_table.to_string(index=False))

    coverage_global = pd.DataFrame(
        [
            {
                "detected_count": len(detected_events),
                "labeled_count": len(events),
                "dropna_count": dropna_count,
                "train_count": len(y_train),
                "calib_count": len(y_calib),
            }
        ]
    )
    family_detected = detected_events["family_id"].value_counts()
    family_labeled = events["family_id"].value_counts()
    family_share = (
        (family_labeled / family_labeled.sum()).mul(100).round(2)
        if not family_labeled.empty
        else pd.Series(dtype=float)
    )
    coverage_by_family = pd.DataFrame(
        {
            "family_id": family_summary["family_id"],
            "detected_count": family_summary["family_id"].map(family_detected).fillna(0).astype(int),
            "labeled_count": family_summary["family_id"].map(family_labeled).fillna(0).astype(int),
            "dropna_count": family_summary["family_id"]
            .map((dropna_mask).groupby(labeled_events["family_id"]).sum())
            .fillna(0)
            .astype(int),
            "train_count": family_summary["samples_train"],
            "calib_count": family_summary["samples_calib"],
            "share_pct": family_summary["family_id"].map(family_share).fillna(0.0),
        }
    )
    logger.debug("-" * 96)
    logger.debug("A) Coverage (supply)")
    logger.debug("Global coverage:\n%s", coverage_global.to_string(index=False))
    logger.debug("Coverage by family:\n%s", coverage_by_family.to_string(index=False))

    base_rate_global = pd.DataFrame(
        [
            {
                "scope": "global",
                "base_rate_train": float(y_train.mean()) if not y_train.empty else float("nan"),
                "base_rate_calib": float(y_calib.mean()) if not y_calib.empty else float("nan"),
            }
        ]
    )
    family_base_rate = family_summary[
        ["family_id", "base_rate_train", "base_rate_calib"]
    ].copy()
    logger.debug("-" * 96)
    logger.debug("B) Label quality & hardness")
    logger.debug("Base rates:\n%s", pd.concat([base_rate_global, family_base_rate]).to_string(index=False))

    r_stats = (
        r_calib.groupby(fam_calib)
        .agg(
            r_mean="mean",
            r_std="std",
            r_p50=lambda s: s.quantile(0.5),
            r_p75=lambda s: s.quantile(0.75),
            r_p90=lambda s: s.quantile(0.9),
            r_p95=lambda s: s.quantile(0.95),
        )
        .reset_index()
    )
    if not r_stats.empty:
        r_stats = r_stats.rename(columns={r_stats.columns[0]: "scope"})
    r_stats_global = pd.DataFrame(
        [
            {
                "scope": "global",
                "r_mean": float(r_calib.mean()) if not r_calib.empty else float("nan"),
                "r_std": float(r_calib.std()) if not r_calib.empty else float("nan"),
                "r_p50": float(r_calib.quantile(0.5)) if not r_calib.empty else float("nan"),
                "r_p75": float(r_calib.quantile(0.75)) if not r_calib.empty else float("nan"),
                "r_p90": float(r_calib.quantile(0.9)) if not r_calib.empty else float("nan"),
                "r_p95": float(r_calib.quantile(0.95)) if not r_calib.empty else float("nan"),
            }
        ]
    )
    logger.debug("r_outcome distribution (calib):\n%s", pd.concat([r_stats_global, r_stats]).to_string(index=False))

    def _regime_breakdown(
        scores: pd.Series,
        labels_: pd.Series,
        r_outcome: pd.Series,
        regimes: pd.Series,
        seed: int = 7,
    ) -> pd.DataFrame:
        if labels_.empty:
            return pd.DataFrame()
        rows: list[dict[str, float | str | int]] = []
        for regime_id, regime_labels in labels_.groupby(regimes, observed=True):
            if isinstance(regime_id, float) and np.isnan(regime_id):
                continue
            scope_scores = scores.loc[regime_labels.index]
            scope_r = r_outcome.loc[regime_labels.index]
            metrics = summarize_metrics(str(regime_id), scope_scores, regime_labels, scope_r, seed=seed)
            top_idx = _topk_indices(scope_scores, regime_labels, 10, seed=seed)
            bottom_idx = _bottomk_indices(scope_scores, regime_labels, 10, seed=seed)
            top_mean = float(scope_r.loc[top_idx].mean()) if len(top_idx) else float("nan")
            bottom_mean = float(scope_r.loc[bottom_idx].mean()) if len(bottom_idx) else float("nan")
            rank_inverted = bool(bottom_mean > top_mean)
            flag = "MIXED"
            if metrics["delta_r_mean@20"] >= 0.10 and metrics["lift@20"] > 1.05:
                flag = "WIN"
            elif metrics["delta_r_mean@20"] <= 0 or metrics["lift@20"] < 1.0:
                flag = "LOSE"
            flag_parts = [flag]
            if rank_inverted:
                flag_parts.append("RANK_INVERTED")
            if metrics["samples"] < 100:
                flag_parts.append("LOW_SAMPLES")
            rows.append(
                {
                    "regime_id": metrics["scope"],
                    "samples_calib": metrics["samples"],
                    "base_rate_calib": metrics["base_rate"],
                    "auc": metrics["auc"],
                    "lift@20": metrics["lift@20"],
                    "r_mean_all": metrics["r_mean_all"],
                    "r_mean@20": metrics["r_mean@20"],
                    "delta_r_mean@20": metrics["delta_r_mean@20"],
                    "spearman": scope_scores.corr(scope_r, method="spearman") if len(scope_scores) else float("nan"),
                    "flag": "|".join(flag_parts),
                }
            )
        return pd.DataFrame(rows)

    regime_df = pd.DataFrame()
    if not is_fallback and preds_meta is not None and not y_calib.empty:
        regime_df = _regime_breakdown(preds_meta, y_calib, r_calib, regime_calib, seed=seed)
        if not regime_df.empty:
            regime_top = regime_df.sort_values("samples_calib", ascending=False).head(8)
            logger.info("INFO | REGIME | TOP regimes (calib)\n%s", regime_top.to_string(index=False))

    best_regime = "NA"
    worst_regime = "NA"
    recommendation = "NO_EDGE_GENERAL; CONSIDER_SYMBOL_SPECIFIC"
    if not regime_df.empty:
        best_idx = regime_df["delta_r_mean@20"].idxmax()
        worst_idx = regime_df["delta_r_mean@20"].idxmin()
        best_regime = regime_df.loc[best_idx, "regime_id"]
        worst_regime = regime_df.loc[worst_idx, "regime_id"]
        win_mask = regime_df["flag"].str.contains("WIN") & (regime_df["samples_calib"] >= 300)
        if win_mask.any():
            recommendation = "EDGE_FOUND_META"
    logger.info(
        "INFO | SUMMARY | BEST_REGIME=%s WORST_REGIME=%s",
        best_regime,
        worst_regime,
    )
    logger.info("INFO | SUMMARY | RECOMMENDATION=%s", recommendation)

    metrics_summary = {}
    if not metrics_df.empty:
        scorer_row = metrics_df[metrics_df["model"] == "SCORER"]
        if not scorer_row.empty:
            metrics_summary = scorer_row.iloc[0].to_dict()

    metadata = {
        "symbol": args.symbol,
        "train_ratio": args.train_ratio,
        "k_bars": k_bars,
        "reward_r": args.reward_r,
        "sl_mult": args.sl_mult,
        "r_thr": args.r_thr,
        "tie_break": args.tie_break,
        "include_transition": args.include_transition,
        "meta_policy": args.meta_policy,
        "meta_margin_min": args.meta_margin_min,
        "meta_margin_max": args.meta_margin_max,
        "decision_thresholds": {
            "n_min": args.decision_n_min,
            "winrate_min": args.decision_winrate_min,
            "r_mean_min": args.decision_r_mean_min,
            "p10_min": args.decision_p10_min,
        },
        "config_path": str(args.config) if args.config else None,
        "config_mode": args.mode,
        "feature_count": event_features_all.shape[1],
        "train_date": datetime.now(timezone.utc).isoformat(),
        "metrics_summary": metrics_summary,
        "research_enabled": research_enabled,
    }
    if research_enabled:
        metadata["research_features"] = research_cfg.get("features", {})
        metadata["research_k_bars_grid"] = research_cfg.get("k_bars_grid")
    scorer_out = _with_suffix(scorer_out_base, output_suffix)
    _save_model_if_ready(is_fallback=is_fallback, scorer=scorer, path=scorer_out, metadata=metadata)

    summary_table = pd.DataFrame(
        [
            {
                "events_total": len(events),
                "labeled": len(events),
                "feature_count": event_features_all.shape[1],
            }
        ]
    )
    logger.info("Summary:\n%s", summary_table.to_string(index=False))
    logger.info("label_distribution=%s", events["label"].value_counts(normalize=True).to_dict())
    logger.info("model_out=%s", scorer_out)

    args.model_dir.mkdir(parents=True, exist_ok=True)
    if not metrics_df.empty:
        metrics_path = _with_suffix(
            args.model_dir / f"metrics_{''.join(ch if (ch.isalnum() or ch in '._-') else '_' for ch in args.symbol)}_event_scorer.csv",
            output_suffix,
        )
        metrics_df.to_csv(metrics_path, index=False)
        logger.info("metrics_out=%s", metrics_path)
    family_path = _with_suffix(
        args.model_dir / f"family_summary_{''.join(ch if (ch.isalnum() or ch in '._-') else '_' for ch in args.symbol)}_event_scorer.csv",
        output_suffix,
    )
    family_summary.to_csv(family_path, index=False)
    logger.info("family_summary_out=%s", family_path)

    if not y_calib.empty and preds_meta is not None:
        sample_cols = ["family_id", "side", "label", "r_outcome"]
        sample_df = events.loc[y_calib.index, sample_cols].copy()
        sample_df["score"] = preds_meta
        sample_df["margin_H1"] = df_m5_ctx["margin_H1"].reindex(sample_df.index)
        sample_df = sample_df.sort_values("score", ascending=False).head(10)
        sample_df = sample_df.reset_index().rename(columns={sample_df.index.name or "index": "time"})
        sample_path = _with_suffix(
            args.model_dir
            / f"calib_top_scored_{''.join(ch if (ch.isalnum() or ch in '._-') else '_' for ch in args.symbol)}_event_scorer.csv",
            output_suffix,
        )
        sample_df.to_csv(sample_path, index=False)
        logger.info("calib_top_scored_out=%s", sample_path)

    if global_scorer._model is not None and hasattr(global_scorer._model, "feature_importances_"):
        importances = pd.Series(global_scorer._model.feature_importances_, index=event_features_all.columns)
        top_features = importances.sort_values(ascending=False).head(20)
        logger.info("Model signature (features=%s):", event_features_all.shape[1])
        logger.info("%s", textwrap.fill(", ".join(event_features_all.columns), width=120))
        logger.info("Top-20 feature importances:\n%s", top_features.to_string())

    warning_summary = pd.DataFrame(warning_summary_rows)
    if not warning_summary.empty and warning_summary["split_warning_hits"].sum() > 0:
        logger.warning(
            "LightGBM split warnings detected. Consider raising min_samples_train or revisiting features."
        )
        logger.info("Split warning summary:\n%s", warning_summary.to_string(index=False))
    coverage_global = diagnostic_report.get("coverage_global", pd.DataFrame())

    def _scorer_metric(table: pd.DataFrame, metric: str) -> float | None:
        if table.empty:
            return None
        if "model" in table.columns:
            scorer_row = table.loc[table["model"] == "SCORER"]
            if not scorer_row.empty:
                return scorer_row.iloc[0].get(metric)
        return table.iloc[0].get(metric)

    def _coverage_metric(table: pd.DataFrame, metric: str) -> float | int | None:
        if table.empty or metric not in table.columns:
            return None
        return table.iloc[0].get(metric)

    def _split_warning_global(table: pd.DataFrame) -> int | None:
        if table.empty or "scope" not in table.columns or "split_warning_hits" not in table.columns:
            return None
        global_row = table.loc[table["scope"] == "global"]
        if global_row.empty:
            return None
        return int(global_row.iloc[0]["split_warning_hits"])

    summary_payload = {
        "symbol": args.symbol,
        "start": args.start,
        "end": args.end,
        "h1_cutoff": h1_cutoff,
        "m5_cutoff": m5_cutoff,
        "events_total_post_meta": _coverage_metric(coverage_global, "events_total_post_meta"),
        "events_per_day": _coverage_metric(coverage_global, "events_per_day"),
        "unique_days": _coverage_metric(coverage_global, "unique_days"),
        "auc_no_meta": _scorer_metric(table_no_meta, "auc"),
        "auc_meta": _scorer_metric(table_meta, "auc"),
        "lift20_meta": _scorer_metric(table_meta, "lift@20"),
        "r_mean20_meta": _scorer_metric(table_meta, "r_mean@20"),
        "spearman_meta": _scorer_metric(table_meta, "spearman"),
        "families_trained": int((family_summary["status"] == "TRAINED").sum())
        if "status" in family_summary.columns
        else None,
        "best_regime_id": best_regime,
        "worst_regime_id": worst_regime,
        "recommendation": recommendation,
        "split_warning_hits_global": _split_warning_global(warning_summary),
        "allow_active_pct": allow_active_pct,
        "feature_count": event_features_all.shape[1],
        "min_samples_train": min_samples_train,
        "k_bars": k_bars,
    }

    research_summary_payload = None
    if research_enabled and preds_meta is not None and not y_calib.empty:
        diagnostics_cfg = research_cfg.get("diagnostics", {}) if isinstance(research_cfg.get("diagnostics"), dict) else {}
        score_shape = _score_shape_diagnostics(preds_meta, fam_calib, y_calib.index, [10, 20, 50])
        train_scores = scorer.predict_proba(X_train, fam_train)
        train_metrics = summarize_metrics("TRAIN", train_scores, y_train, r_train, seed=seed)
        calib_metrics = summarize_metrics("CALIB", preds_meta, y_calib, r_calib, seed=seed)
        guardrails = _research_guardrails(
            score_shape,
            train_metrics=train_metrics,
            calib_metrics=calib_metrics,
            diagnostics_cfg=diagnostics_cfg,
            calib_samples=len(y_calib),
        )
        research_summary_payload = {
            "score_shape": score_shape.to_dict(orient="records"),
            "guardrails": guardrails.to_dict(orient="records"),
        }
        print("\n=== RESEARCH DIAGNOSTICS ===")
        print(_format_table_block("Score shape diagnostics", score_shape))
        print(_format_table_block("Research guardrails", guardrails))

    if research_summary_payload is not None:
        summary_payload["research"] = research_summary_payload

    summary_path = _with_suffix(
        args.model_dir / f"summary_{''.join(ch if (ch.isalnum() or ch in '._-') else '_' for ch in args.symbol)}_event_scorer.json",
        output_suffix,
    )
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2, default=str)
    logger.info("summary_out=%s", summary_path)
    logger.info("=" * 96)
    if args.mode == "research":
        _print_research_summary_block(diagnostic_report.get("session_conditional_edge", pd.DataFrame()))


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.log_level)
    min_samples_train = 200
    seed = 7
    research_cfg: dict[str, object] = {"enabled": False, "features": {}, "diagnostics": {}, "k_bars_grid": None}

    config_path = args.config
    if config_path is not None:
        defaults = _default_symbol_config(args)
        overrides = load_config(config_path)
        merged = deep_merge(defaults, overrides)

        args.symbol = merged.get("symbol", args.symbol)
        event_cfg = merged.get("event_scorer", {})
        if isinstance(event_cfg, dict):
            args.train_ratio = event_cfg.get("train_ratio", args.train_ratio)
            args.k_bars = event_cfg.get("k_bars", args.k_bars)
            args.reward_r = event_cfg.get("reward_r", args.reward_r)
            args.sl_mult = event_cfg.get("sl_mult", args.sl_mult)
            args.r_thr = event_cfg.get("r_thr", args.r_thr)
            args.tie_break = event_cfg.get("tie_break", args.tie_break)

            include_transition = event_cfg.get("include_transition")
            if include_transition is not None:
                args.include_transition = "on" if include_transition else "off"

            meta_cfg = event_cfg.get("meta_policy", {})
            if isinstance(meta_cfg, dict):
                enabled = meta_cfg.get("enabled")
                if enabled is not None:
                    args.meta_policy = "on" if enabled else "off"
                args.meta_margin_min = meta_cfg.get("meta_margin_min", args.meta_margin_min)
                args.meta_margin_max = meta_cfg.get("meta_margin_max", args.meta_margin_max)

            family_cfg = event_cfg.get("family_training", {})
            if isinstance(family_cfg, dict):
                min_samples_train = family_cfg.get("min_samples_train", min_samples_train)

            thresholds = _resolve_decision_thresholds(merged, args.mode)
            if thresholds:
                args.decision_n_min = thresholds.get("n_min", args.decision_n_min)
                args.decision_winrate_min = thresholds.get("winrate_min", args.decision_winrate_min)
                args.decision_r_mean_min = thresholds.get("r_mean_min", args.decision_r_mean_min)
                args.decision_p10_min = thresholds.get("p10_min", args.decision_p10_min)

            research_cfg = _resolve_research_config(merged, args.mode)

        logger.info("Loaded config overrides from %s (mode=%s)", config_path, args.mode)

    research_enabled = bool(research_cfg.get("enabled", False))
    k_bars_grid = research_cfg.get("k_bars_grid") if research_enabled else None
    if research_enabled and isinstance(k_bars_grid, list) and k_bars_grid:
        k_values = k_bars_grid
    else:
        k_values = [args.k_bars]
    multi_k = len(k_values) > 1

    def _safe_symbol(sym: str) -> str:
        return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in sym)

    model_path = args.state_model
    if model_path is None:
        model_path = args.model_dir / f"{_safe_symbol(args.symbol)}_state_engine.pkl"

    scorer_out_base = args.model_out
    if scorer_out_base is None:
        scorer_out_base = args.model_dir / f"{_safe_symbol(args.symbol)}_event_scorer.pkl"

    connector = MT5Connector()
    fecha_inicio = pd.to_datetime(args.start)
    fecha_fin = pd.to_datetime(args.end)

    ohlcv_h1 = connector.obtener_h1(args.symbol, fecha_inicio, fecha_fin)
    ohlcv_m5 = connector.obtener_m5(args.symbol, fecha_inicio, fecha_fin)
    ohlcv_m5["symbol"] = args.symbol
    server_now = connector.server_now(args.symbol).tz_localize(None)

    h1_cutoff = server_now.floor("h")
    m5_cutoff = server_now.floor("5min")
    ohlcv_h1 = ohlcv_h1[ohlcv_h1.index < h1_cutoff]
    ohlcv_m5 = ohlcv_m5[ohlcv_m5.index < m5_cutoff]
    m5_dupes = int(ohlcv_m5.index.duplicated().sum())
    h1_dupes = int(ohlcv_h1.index.duplicated().sum())
    logger.info(
        "Period: %s -> %s | h1_cutoff=%s m5_cutoff=%s",
        fecha_inicio,
        fecha_fin,
        h1_cutoff,
        m5_cutoff,
    )
    logger.info("Rows: H1=%s M5=%s", len(ohlcv_h1), len(ohlcv_m5))
    print("\n=== EVENT SCORER TRAINING ===")

    if not model_path.exists():
        raise FileNotFoundError(f"State model not found: {model_path}")

    state_model = StateEngineModel()
    state_model.load(model_path)

    feature_engineer = FeatureEngineer(FeatureConfig())
    gating = GatingPolicy()
    ctx_h1 = build_h1_context(ohlcv_h1, state_model, feature_engineer, gating)

    df_m5_ctx = merge_h1_m5(ctx_h1, ohlcv_m5)
    if "atr_14" not in df_m5_ctx.columns:
        df_m5_ctx["atr_14"] = _ensure_atr_14(df_m5_ctx)
    logger.info("Rows after merge: M5_ctx=%s", len(df_m5_ctx))
    m5_ctx_merged = len(df_m5_ctx)
    ctx_nan_cols = ["state_hat_H1", "margin_H1"]
    if "atr_short" in df_m5_ctx.columns:
        ctx_nan_cols.append("atr_short")
    ctx_nan = df_m5_ctx[ctx_nan_cols].isna().mean().mul(100).round(2)
    ctx_nan_table = pd.DataFrame(
        {
            "column": ctx_nan.index,
            "nan_pct": ctx_nan.values,
        }
    )
    logger.info("Context NaN rates:\n%s", ctx_nan_table.to_string(index=False))
    df_m5_ctx = df_m5_ctx.dropna(subset=["state_hat_H1", "margin_H1"])
    logger.info("Rows after dropna ctx: M5_ctx=%s", len(df_m5_ctx))
    m5_ctx_dropna = len(df_m5_ctx)
    feature_builder = FeatureBuilder()
    features_all = feature_builder.build(df_m5_ctx)
    # --- atr_ratio (diagnostic-only, comparable cross-symbol) ---
    atr14 = df_m5_ctx["atr_14"].reindex(features_all.index)
    
    if "atr_short" in features_all.columns:
        atr_short = pd.to_numeric(features_all["atr_short"], errors="coerce")
        atr_ratio = (atr_short / atr14.replace(0, np.nan)).astype(float)
    else:
        atr_ratio = pd.Series(np.nan, index=features_all.index)

    if research_enabled:
        research_features = _build_research_context_features(
            df_m5_ctx,
            ohlcv_h1,
            args.symbol,
            research_cfg,
        )
        if not research_features.empty:
            features_all = pd.concat([features_all, research_features], axis=1)
            logger.info("Research features enabled: %s", list(research_features.columns))

    state_labels_before = df_m5_ctx["state_hat_H1"].map(_state_label)
    state_counts_before = state_labels_before.value_counts()
    print("[STATE MIX | M5_CTX]")
    for line in _format_state_mix(state_counts_before, ["BALANCE", "TREND", "TRANSITION"]):
        print(line)

    m5_total = len(ohlcv_m5)

    event_config = EventDetectionConfig()
    detected_events = detect_events(df_m5_ctx, config=event_config)
    if detected_events.empty:
        logger.warning("No events detected; exiting.")
        events_diag = pd.DataFrame(
            columns=["state_hat_H1", "allow_id", "margin", "r", "win", "margin_bin"]
        )
        events_diag.index = pd.DatetimeIndex([])
        is_fallback = _apply_fallback_guardrails(0, args.fallback_min_samples)
        if is_fallback:
            print(
                "=== FALLBACK DIAGNÓSTICO: events_total_post_meta=0 < fallback_min_samples="
                f"{args.fallback_min_samples} ==="
            )
        diagnostic_report = _build_training_diagnostic_report(events_diag, df_m5_ctx, thresholds=args)
        if args.mode == "research":
            _print_research_summary_block(diagnostic_report.get("session_conditional_edge", pd.DataFrame()))
        return

    events_dupes = int(detected_events.index.duplicated().sum())
    logger.info("Detected events by type:\n%s", detected_events["event_type"].value_counts().to_string())
    logger.info("Detected events by family:\n%s", detected_events["family_id"].value_counts().to_string())

    event_indexer = ohlcv_m5.index.get_indexer(detected_events.index)
    missing_index = int((event_indexer == -1).sum())
    missing_future = int(((event_indexer != -1) & (event_indexer + 1 >= len(ohlcv_m5.index))).sum())
    events_index_match_pct = float(detected_events.index.isin(df_m5_ctx.index).mean())
    missing_atr_pct = float(detected_events["atr_14"].isna().mean() * 100)
    m5_atr14_nan_pct = float(df_m5_ctx["atr_14"].isna().mean() * 100)
    events_atr14_nan_pct = float(detected_events["atr_14"].isna().mean() * 100)
    sanity_table = pd.DataFrame(
        [
            {
                "m5_dupes_detected": m5_dupes,
                "h1_dupes_detected": h1_dupes,
                "event_dupes_detected": events_dupes,
                "events_missing_index": missing_index,
                "events_missing_future_slice": missing_future,
                "events_missing_atr_pct": round(missing_atr_pct, 2),
                "events_index_match_pct": round(events_index_match_pct, 4),
                "m5_atr14_nan_pct": round(m5_atr14_nan_pct, 2),
                "events_atr14_nan_pct": round(events_atr14_nan_pct, 2),
            }
        ]
    )
    logger.info("Data quality checks:\n%s", sanity_table.to_string(index=False))
    if events_index_match_pct < 0.99:
        event_ts = detected_events["ts"] if "ts" in detected_events.columns else detected_events.index
        event_sample = None if detected_events.empty else detected_events.index[0]
        ctx_sample = None if df_m5_ctx.empty else df_m5_ctx.index[0]
        logger.warning(
            "Index mismatch details: events_ts_dtype=%s m5_ctx_index_dtype=%s events_tz=%s m5_ctx_tz=%s "
            "events_sample=%r m5_ctx_sample=%r",
            getattr(event_ts, "dtype", None),
            getattr(df_m5_ctx.index, "dtype", None),
            getattr(getattr(event_ts, "dt", event_ts), "tz", None),
            getattr(df_m5_ctx.index, "tz", None),
            event_sample,
            ctx_sample,
        )

    for k_bars in k_values:
        output_suffix = f"_k{k_bars}" if multi_k else ""
        _run_training_for_k(
            args=args,
            k_bars=k_bars,
            output_suffix=output_suffix,
            detected_events=detected_events,
            event_config=event_config,
            df_m5_ctx=df_m5_ctx,
            ohlcv_m5=ohlcv_m5,
            features_all=features_all,
            atr_ratio=atr_ratio,
            feature_builder=feature_builder,
            m5_total=m5_total,
            m5_ctx_merged=m5_ctx_merged,
            m5_ctx_dropna=m5_ctx_dropna,
            h1_cutoff=h1_cutoff,
            m5_cutoff=m5_cutoff,
            min_samples_train=min_samples_train,
            seed=seed,
            scorer_out_base=scorer_out_base,
            research_enabled=research_enabled,
            research_cfg=research_cfg,
        )
    return


if __name__ == "__main__":
    main()
