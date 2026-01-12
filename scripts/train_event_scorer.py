"""Train the Event Scorer model from MT5 data.

The scorer uses a triple-barrier continuous outcome (r_outcome) and reports
ranking metrics like lift@K to gauge whether top-ranked events outperform the
base rate. lift@K = precision@K / base_rate.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta, timezone
import io
from pathlib import Path
import sys
import textwrap
from contextlib import redirect_stderr, redirect_stdout

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
    parser.add_argument("--log-level", default="INFO", help="Logging level (INFO, DEBUG, WARNING)")
    return parser.parse_args()


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
        if row["p10"] < thresholds.decision_p10_min:
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
    for frame in [coverage_global, coverage_by, regime_full, regime_ranked, stability, decision]:
        numeric_cols = frame.select_dtypes(include=[np.number]).columns
        frame[numeric_cols] = frame[numeric_cols].round(4)
    report["coverage_global"] = coverage_global
    report["coverage_by_state_margin"] = coverage_by
    report["regime_edge_full"] = regime_full
    report["regime_edge_ranked"] = regime_ranked
    report["stability"] = stability
    report["decision"] = decision
    print("=== TRAINING DIAGNOSTIC REPORT ===")
    print(_format_table_block("Coverage (global)", coverage_global))
    print(_format_table_block("Coverage (by state x margin_bin)", coverage_by))
    print(_format_table_block("Regime Edge (full)", regime_full))
    print(_format_table_block("Regime Edge (top 10 / bottom 10 by r_mean)", regime_ranked))
    print(_format_table_block("Stability temporal (regime edge por split)", stability))
    print(_format_table_block("Decision", decision))
    return report


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.log_level)
    min_samples_train = 200
    seed = 7

    def _safe_symbol(sym: str) -> str:
        return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in sym)

    model_path = args.state_model
    if model_path is None:
        model_path = args.model_dir / f"{_safe_symbol(args.symbol)}_state_engine.pkl"

    scorer_out = args.model_out
    if scorer_out is None:
        scorer_out = args.model_dir / f"{_safe_symbol(args.symbol)}_event_scorer.pkl"

    connector = MT5Connector()
    fecha_inicio = pd.to_datetime(args.start)
    fecha_fin = pd.to_datetime(args.end)

    ohlcv_h1 = connector.obtener_h1(args.symbol, fecha_inicio, fecha_fin)
    ohlcv_m5 = connector.obtener_m5(args.symbol, fecha_inicio, fecha_fin)
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
    print(
        "symbol={symbol} start={start} end={end} k_bars={k} reward_r={reward} sl_mult={sl} r_thr={thr} "
        "meta_policy={meta} include_transition={include_transition}".format(
            symbol=args.symbol,
            start=args.start,
            end=args.end,
            k=args.k_bars,
            reward=args.reward_r,
            sl=args.sl_mult,
            thr=args.r_thr,
            meta=args.meta_policy,
            include_transition=args.include_transition,
        )
    )

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

    state_labels_before = df_m5_ctx["state_hat_H1"].map(_state_label)
    state_counts_before = state_labels_before.value_counts()
    print("[STATE MIX | M5_CTX]")
    for line in _format_state_mix(state_counts_before, ["BALANCE", "TREND", "TRANSITION"]):
        print(line)

    m5_total = len(ohlcv_m5)

    event_config = EventDetectionConfig()
    events = detect_events(df_m5_ctx, config=event_config)
    if events.empty:
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
        _build_training_diagnostic_report(events_diag, df_m5_ctx, thresholds=args)
        return

    detected_events = events.copy()
    events_dupes = int(detected_events.index.duplicated().sum())
    logger.info("Detected events by type:\n%s", events["event_type"].value_counts().to_string())
    logger.info("Detected events by family:\n%s", events["family_id"].value_counts().to_string())

    event_indexer = ohlcv_m5.index.get_indexer(events.index)
    missing_index = int((event_indexer == -1).sum())
    missing_future = int(((event_indexer != -1) & (event_indexer + 1 >= len(ohlcv_m5.index))).sum())
    events_index_match_pct = float(detected_events.index.isin(df_m5_ctx.index).mean())
    missing_atr_pct = float(events["atr_14"].isna().mean() * 100)
    m5_atr14_nan_pct = float(df_m5_ctx["atr_14"].isna().mean() * 100)
    events_atr14_nan_pct = float(events["atr_14"].isna().mean() * 100)
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

    events = label_events(
        detected_events,
        ohlcv_m5,
        args.k_bars,
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
        _build_training_diagnostic_report(events_diag, df_m5_ctx, thresholds=args)
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

    print("\n[STATE MIX | EVENTS POST FILTER]")
    state_counts_events = events_state_filtered["state_label"].value_counts()
    for line in _format_state_mix(state_counts_events, ["BALANCE", "TREND", "TRANSITION"]):
        print(line)

    vwap_mode = "provided" if "vwap" in df_m5_ctx.columns else (
        event_config.vwap_reset_mode
        if event_config.vwap_reset_mode is not None
        else ("session" if any(col in df_m5_ctx.columns for col in ("session_id", "session")) else "daily")
    )
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
    print(f"events_labeled={len(labeled_events)}")
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
    _build_training_diagnostic_report(events_diag, df_m5_ctx, thresholds=args)

    if is_fallback:
        args.model_dir.mkdir(parents=True, exist_ok=True)
        detected_path = args.model_dir / f"events_detected_{_safe_symbol(args.symbol)}.csv"
        labeled_path = args.model_dir / f"events_labeled_{_safe_symbol(args.symbol)}.csv"
        detected_events.reset_index().to_csv(detected_path, index=False)
        labeled_events.reset_index().to_csv(labeled_path, index=False)
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

    feature_builder = FeatureBuilder()
    features_all = feature_builder.build(df_m5_ctx)
    event_features_all = features_all.reindex(events_state_filtered.index)
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
        detected_path = args.model_dir / f"events_detected_{_safe_symbol(args.symbol)}.csv"
        labeled_path = args.model_dir / f"events_labeled_{_safe_symbol(args.symbol)}.csv"
        detected_events.reset_index().to_csv(detected_path, index=False)
        labeled_events.reset_index().to_csv(labeled_path, index=False)
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
        "k_bars": args.k_bars,
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
        "k_bars": args.k_bars,
        "reward_r": args.reward_r,
        "sl_mult": args.sl_mult,
        "r_thr": args.r_thr,
        "tie_break": args.tie_break,
        "feature_count": event_features_all.shape[1],
        "train_date": datetime.now(timezone.utc).isoformat(),
        "metrics_summary": metrics_summary,
    }
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
        metrics_path = args.model_dir / f"metrics_{_safe_symbol(args.symbol)}_event_scorer.csv"
        metrics_df.to_csv(metrics_path, index=False)
        logger.info("metrics_out=%s", metrics_path)
    family_path = args.model_dir / f"family_summary_{_safe_symbol(args.symbol)}_event_scorer.csv"
    family_summary.to_csv(family_path, index=False)
    logger.info("family_summary_out=%s", family_path)

    if not y_calib.empty and preds_meta is not None:
        sample_cols = ["family_id", "side", "label", "r_outcome"]
        sample_df = events.loc[y_calib.index, sample_cols].copy()
        sample_df["score"] = preds_meta
        sample_df["margin_H1"] = df_m5_ctx["margin_H1"].reindex(sample_df.index)
        sample_df = sample_df.sort_values("score", ascending=False).head(10)
        sample_df = sample_df.reset_index().rename(columns={sample_df.index.name or "index": "time"})
        sample_path = args.model_dir / f"calib_top_scored_{_safe_symbol(args.symbol)}_event_scorer.csv"
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
    logger.info("=" * 96)


if __name__ == "__main__":
    main()
