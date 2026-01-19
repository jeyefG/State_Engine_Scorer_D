"""Train the State Engine model from MetaTrader 5 data."""

from __future__ import annotations

import argparse
import ast
import json
import importlib
import logging
import math
import itertools
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys

# --- ensure project root is on PYTHONPATH ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from typing import Any

import numpy as np
import pandas as pd

from state_engine.features import FeatureConfig
from state_engine.context_features import build_context_features
from state_engine.config_loader import load_config
from state_engine.gating import (
    GatingPolicy,
    build_transition_gating_thresholds,
)
from state_engine.pipeline_phase_d import validate_allow_context_requirements
from state_engine.labels import StateLabels
from state_engine.model import StateEngineModel, StateEngineModelConfig
from state_engine.mt5_connector import MT5Connector
from state_engine.pipeline import DatasetBuilder
from state_engine.quality import (
    assign_quality_labels,
    build_quality_diagnostics,
    load_quality_config,
)




def parse_args() -> argparse.Namespace:
    
    default_end = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    parser = argparse.ArgumentParser(description="Train State Engine model.")
    parser.add_argument("--symbol", required=True, help="Símbolo MT5 (ej. EURUSD)")
    parser.add_argument(
        "--start",
        required=True,
        help="Fecha inicio (YYYY-MM-DD) para descarga de velas",
    )
    parser.add_argument(
        "--end",
        default = default_end,
        help="Fecha fin (YYYY-MM-DD) para descarga de velas",
    )
    parser.add_argument(
        "--timeframe",
        choices=["M30", "H1", "H2", "H4"],
        default="H1",
        help="Timeframe de velas para el State Engine (M30, H1, H2, H4).",
    )
    parser.add_argument(
        "--window-hours",
        type=int,
        default=24,
        help="Ventana temporal total (en horas) usada como contexto estructural.",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=None,
        help="Model output path. Si no se indica, se genera automáticamente desde el símbolo.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(PROJECT_ROOT / "state_engine" / "models"),
        help="Directorio base para guardar el modelo cuando --model-out no se especifica.",
    )
    parser.add_argument(
        "--log-level",
        choices=["minimal", "verbose", "debug"],
        default="minimal",
        help="Nivel de reporte en consola (minimal, verbose, debug).",
    )
    parser.add_argument(
        "--logging-level",
        default="INFO",
        help="Logging level (INFO, DEBUG, WARNING)",
    )
    parser.add_argument("--min-samples", type=int, default=2000, help="Minimum samples required to train")
    parser.add_argument(
        "--ev-min-split-samples",
        type=int,
        default=50,
        help="Minimum samples per split for EV stability",
    )
    parser.add_argument("--split-ratio", type=float, default=0.8, help="Train/test split ratio (0-1)")
    parser.add_argument("--no-rich", action="store_true", help="Disable rich console output")
    parser.add_argument("--report-out", type=Path, help="Optional report output path (.json)")
    parser.add_argument("--class-weight-balanced", action="store_true", help="Use class_weight='balanced' in LightGBM")
    parser.add_argument("--quality", action="store_true", help="Enable Quality Layer labeling (Phase C)")
    parser.add_argument(
        "--quality-config",
        type=Path,
        default=None,
        help="Optional Quality Layer config path (YAML/JSON)",
    )
    parser.add_argument(
        "--diagnose-rescue-scans",
        action="store_true",
        help="Run diagnostic BALANCE/TRANSITION rescue scans (telemetry only).",
    )
    parser.add_argument(
        "--rescue-n-min",
        type=int,
        default=30,
        help="Minimum samples per rescue scan candidate bucket.",
    )
    parser.add_argument(
        "--rescue-delta-max",
        type=float,
        default=0.15,
        help="Max abs(delta_vs_state) for rescue scan candidates.",
    )
    parser.add_argument(
        "--rescue-top-k",
        type=int,
        default=12,
        help="Top-k rows to display for rescue scan tables.",
    )
    return parser.parse_args()


def try_import_rich() -> dict[str, Any] | None:
    try:
        from rich.console import Console
        from rich.table import Table

        return {"Console": Console, "Table": Table}
    except Exception:
        return None


def setup_logging(level: str) -> logging.Logger:
    logger = logging.getLogger("state_engine")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)-8s %(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def class_distribution(labels: np.ndarray, label_order: list[StateLabels]) -> list[dict[str, Any]]:
    total = len(labels)
    result = []
    for label in label_order:
        count = int(np.sum(labels == label))
        pct = (count / total) * 100 if total else 0.0
        result.append({"label": label.name, "count": count, "pct": pct})
    return result


def confusion_matrix(labels_true: np.ndarray, labels_pred: np.ndarray, label_order: list[StateLabels]) -> np.ndarray:
    matrix = np.zeros((len(label_order), len(label_order)), dtype=int)
    for i, actual in enumerate(label_order):
        for j, predicted in enumerate(label_order):
            matrix[i, j] = int(np.sum((labels_true == actual) & (labels_pred == predicted)))
    return matrix


def f1_macro(matrix: np.ndarray) -> float:
    f1_scores = []
    for idx in range(matrix.shape[0]):
        tp = matrix[idx, idx]
        fp = matrix[:, idx].sum() - tp
        fn = matrix[idx, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        f1_scores.append(f1)
    return float(np.mean(f1_scores)) if f1_scores else 0.0


def percentiles(series: pd.Series, qs: list[float]) -> dict[str, float | None]:
    """Return selected percentiles for a numeric Series.

    Parameters
    ----------
    series:
        Numeric series (non-numeric coerced to NaN).
    qs:
        Percentiles in [0, 100], e.g. [0, 50, 90, 95, 99, 100].

    Returns
    -------
    dict mapping percentile -> value (or None if empty after dropping NaN).
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {str(q): None for q in qs}
    qv = s.quantile([q / 100.0 for q in qs]).to_dict()
    return {str(q): float(qv[q / 100.0]) for q in qs}


def render_table(
    console: Any,
    table_class: Any,
    title: str,
    columns: list[str],
    rows: list[list[Any]],
) -> None:
    table = table_class(title=title)
    for col in columns:
        table.add_column(str(col))
    for row in rows:
        table.add_row(*[str(x) for x in row])
    console.print(table)


REPORT_LEVELS = {"minimal": 0, "verbose": 1, "debug": 2}


def _bucketize(series: pd.Series, bins: list[float], labels: list[str]) -> pd.Series:
    if series is None or series.empty or not series.notna().any():
        return pd.Series("NA", index=series.index if series is not None else None)
    bucketed = pd.cut(series.astype(float), bins=bins, labels=labels, include_lowest=True)
    return bucketed.astype(object).fillna("NA")


def _session_bucket_from_timestamp(index: pd.DatetimeIndex) -> pd.Series:
    hours = pd.Series(index.hour, index=index)
    bins = [
        (0, 6, "ASIA"),
        (7, 12, "LONDON"),
        (13, 17, "NY"),
        (18, 23, "NY_PM"),
    ]
    def _map_hour(hour: int) -> str:
        for start, end, label in bins:
            if start <= hour <= end:
                return label
        return "NY_PM"
    return hours.map(_map_hour).rename("ctx_session_bucket")


def _split_time_terciles(index: pd.DatetimeIndex) -> list[pd.Series]:
    index_ns = index.view("i8")
    t33 = np.quantile(index_ns, 0.33)
    t66 = np.quantile(index_ns, 0.66)
    return [
        index_ns <= t33,
        (index_ns > t33) & (index_ns <= t66),
        index_ns > t66,
    ]


def _format_bin_edge(value: float) -> str:
    if value == float("inf"):
        return "inf"
    if value == -float("inf"):
        return "-inf"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _build_range_labels(bins: list[float]) -> list[str]:
    labels = []
    for left, right in zip(bins[:-1], bins[1:]):
        if left == -float("inf"):
            labels.append(f"<={_format_bin_edge(right)}")
        elif right == float("inf"):
            labels.append(f">{_format_bin_edge(left)}")
        else:
            labels.append(f"{_format_bin_edge(left)}-{_format_bin_edge(right)}")
    return labels


def _coerce_bins(
    override: list[float] | tuple[float, ...] | None,
    fallback: tuple[float, ...],
    fallback_labels: tuple[str, ...],
) -> tuple[tuple[float, ...], tuple[str, ...]]:
    if not override:
        return fallback, fallback_labels
    bins = tuple(float(x) for x in override)
    labels = tuple(_build_range_labels(list(bins)))
    return bins, labels


def _format_rescue_table(
    df: pd.DataFrame,
    *,
    column_map: dict[str, str],
    ordered_columns: list[str],
) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.rename(columns=column_map)
    available_cols = [col for col in ordered_columns if col in df.columns]
    df = df[available_cols].copy()
    float_cols = [col for col in ["ev", "p10", "p50", "p90", "delta", "pct_state"] if col in df.columns]
    if float_cols:
        df[float_cols] = df[float_cols].round(6)
    return df


def _load_diagnostic_table_config(symbol: str, logger: logging.Logger) -> dict[str, Any]:
    config_path = Path("configs") / "symbols" / f"{symbol}.yaml"
    if not config_path.exists():
        return {}
    try:
        config = load_config(config_path)
    except Exception as exc:
        logger.warning("diagnostic_tables config load failed for %s: %s", symbol, exc)
        return {}
    diagnostic_cfg = config.get("diagnostic_tables", {})
    if diagnostic_cfg is None or not isinstance(diagnostic_cfg, dict):
        return {}
    return diagnostic_cfg


def _load_symbol_config(symbol: str, logger: logging.Logger) -> dict[str, Any]:
    config_path = PROJECT_ROOT / "configs" / "symbols" / f"{symbol}.yaml"
    logger.info("SYMBOL_CONFIG_PATH symbol=%s path=%s exists=%s", symbol, config_path, config_path.exists())
    if not config_path.exists():
        return {}
    try:
        config = load_config(config_path)
    except Exception as exc:
        logger.warning("symbol config load failed for %s: %s", symbol, exc)
        return {}
    if not isinstance(config, dict):
        return {}
    return config


def _allow_filter_config_rows(symbol_config: dict[str, Any], allow_rule: str) -> list[dict[str, str]]:
    allow_cfg = symbol_config.get("allow_context_filters", {})
    if not isinstance(allow_cfg, dict):
        return []
    rule_cfg = allow_cfg.get(allow_rule, {})
    if not isinstance(rule_cfg, dict):
        return []
    return [{"key": f"{allow_rule}.{key}", "source": "symbol"} for key in sorted(rule_cfg.keys())]


def _allow_context_filter_counts(
    allow_context_frame: pd.DataFrame,
    allow_rule: str,
    rule_cfg: dict[str, Any],
    logger: logging.Logger,
) -> pd.DataFrame:
    if allow_rule not in allow_context_frame.columns:
        logger.warning("allow_context_filters.%s missing in gating_df; skipping counts.", allow_rule)
        return pd.DataFrame()

    base_allow = allow_context_frame[allow_rule].astype(bool)
    total_base = int(base_allow.sum())
    if total_base == 0:
        return pd.DataFrame(
            [
                {
                    "filter": "base_allow",
                    "total_base": total_base,
                    "pass": 0,
                    "fail": 0,
                    "pass_pct": 0.0,
                    "fail_pct": 0.0,
                    "notes": "no base allow rows",
                }
            ]
        )

    def _mask_from_col(col_name: str, mask_fn, label: str) -> dict[str, Any]:
        if col_name not in allow_context_frame.columns:
            return {
                "filter": label,
                "total_base": total_base,
                "pass": 0,
                "fail": total_base,
                "pass_pct": 0.0,
                "fail_pct": 100.0,
                "notes": f"missing {col_name}",
            }
        series = allow_context_frame[col_name]
        mask = mask_fn(series)
        pass_count = int((base_allow & mask).sum())
        fail_count = total_base - pass_count
        pass_pct = (pass_count / total_base) * 100.0 if total_base else 0.0
        fail_pct = (fail_count / total_base) * 100.0 if total_base else 0.0
        return {
            "filter": label,
            "total_base": total_base,
            "pass": pass_count,
            "fail": fail_count,
            "pass_pct": pass_pct,
            "fail_pct": fail_pct,
            "notes": "",
        }

    rows: list[dict[str, Any]] = []
    sessions_in = rule_cfg.get("sessions_in")
    if sessions_in is not None:
        allowed = {str(val) for val in sessions_in}
        rows.append(
            _mask_from_col(
                "session_bucket",
                lambda s: s.astype(str).isin(allowed),
                f"sessions_in={sorted(allowed)}",
            )
        )
    state_age_min = rule_cfg.get("state_age_min")
    state_age_max = rule_cfg.get("state_age_max")
    if state_age_min is not None or state_age_max is not None:
        def _state_age_mask(series: pd.Series) -> pd.Series:
            numeric = pd.to_numeric(series, errors="coerce")
            mask = pd.Series(True, index=series.index)
            if state_age_min is not None:
                mask &= numeric >= float(state_age_min)
            if state_age_max is not None:
                mask &= numeric <= float(state_age_max)
            return mask

        label = f"state_age[{state_age_min},{state_age_max}]"
        rows.append(_mask_from_col("state_age", _state_age_mask, label))

    dist_vwap_atr_min = rule_cfg.get("dist_vwap_atr_min")
    dist_vwap_atr_max = rule_cfg.get("dist_vwap_atr_max")
    if dist_vwap_atr_min is not None or dist_vwap_atr_max is not None:
        def _dist_mask(series: pd.Series) -> pd.Series:
            numeric = pd.to_numeric(series, errors="coerce")
            mask = pd.Series(True, index=series.index)
            if dist_vwap_atr_min is not None:
                mask &= numeric >= float(dist_vwap_atr_min)
            if dist_vwap_atr_max is not None:
                mask &= numeric <= float(dist_vwap_atr_max)
            return mask

        label = f"dist_vwap_atr[{dist_vwap_atr_min},{dist_vwap_atr_max}]"
        rows.append(_mask_from_col("dist_vwap_atr", _dist_mask, label))

    return pd.DataFrame(rows)

def _print_block(
    title: str,
    df: pd.DataFrame,
    *,
    console: Any | None,
    table_class: Any | None,
    logger: logging.Logger,
    max_rows: int | None = None,
) -> None:
    if max_rows is not None:
        df = df.head(max_rows)
    if console and table_class:
        render_table(console, table_class, title, df.columns.tolist(), df.values.tolist())
    else:
        logger.info(
            "%s\n%s",
            title,
            df.to_string(index=False, max_colwidth=None, line_width=2000),
        )


def _export_rescue_csv(
    df: pd.DataFrame,
    *,
    title: str,
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    if df.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_title = title.lower().replace(" ", "_")
    output_path = output_dir / f"{safe_title}.csv"
    df.to_csv(output_path, index=False)
    logger.info("rescue_scan export_csv=%s rows=%s", output_path, len(df))


def _rescue_scan_tables(
    df_outputs: pd.DataFrame,
    *,
    target_state: str,
    state_col_candidates: tuple[str, ...] = ("state", "state_base"),
    quality_col: str = "quality",
    time_col: str = "time",
    split_col: str = "split",
    top_k: int = 12,
    n_min: int = 30,
    delta_max: float = 0.15,
    age_bins: tuple[float, ...] = (-float("inf"), 2, 5, 10, float("inf")),
    age_labels: tuple[str, ...] = ("0-2", "3-5", "6-10", "11+"),
    dist_bins: tuple[float, ...] = (-float("inf"), 0.5, 1.0, 2.0, float("inf")),
    dist_labels: tuple[str, ...] = ("<=0.5", "0.5-1", "1-2", ">2"),
    atr_ratio_bins: tuple[float, ...] = (-float("inf"), 0.75, 1.0, 1.25, 1.5, float("inf")),
    atr_ratio_labels: tuple[str, ...] = ("<=0.75", "0.75-1", "1-1.25", "1.25-1.5", ">1.5"),
    break_mag_bins: tuple[float, ...] = (-float("inf"), 0.5, 1.0, 1.5, 2.5, float("inf")),
    break_mag_labels: tuple[str, ...] = ("<=0.5", "0.5-1", "1-1.5", "1.5-2.5", ">2.5"),
    reentry_bins: tuple[float, ...] = (-float("inf"), 0.5, 1.5, 2.5, 4.5, float("inf")),
    reentry_labels: tuple[str, ...] = ("0", "1", "2", "3-4", "5+"),
    confidence_cols: tuple[str, ...] = ("score_margin", "margin", "confidence"),
    logger: logging.Logger | None = None,
    console: Any | None = None,
    table_class: Any | None = None,
    rescue_output_dir: Path | None = None,
) -> None:
    logger = logger or logging.getLogger(__name__)
    rescue_output_dir = rescue_output_dir or (PROJECT_ROOT / "state_engine" / "models" / "rescue")

    def _emit(title: str, df: pd.DataFrame, max_rows: int | None = None) -> None:
        _export_rescue_csv(df, title=title, output_dir=rescue_output_dir, logger=logger)
        _print_block(
            title,
            df,
            console=console,
            table_class=table_class,
            logger=logger,
            max_rows=max_rows,
        )

    state_col = next((col for col in state_col_candidates if col in df_outputs.columns), None)
    if state_col is None:
        logger.warning(
            "rescue_scan target=%s skipped: missing state column (candidates=%s)",
            target_state,
            state_col_candidates,
        )
        return

    total_rows = len(df_outputs)
    state_df = df_outputs.loc[df_outputs[state_col] == target_state].copy()
    n_state = len(state_df)
    pct_total = (n_state / total_rows * 100.0) if total_rows else 0.0

    if "session_bucket" in state_df.columns:
        state_df["session_bucket"] = state_df["session_bucket"].fillna("UNKNOWN")
    else:
        logger.info("rescue_scan target=%s session_bucket missing; using UNKNOWN", target_state)
        state_df["session_bucket"] = "UNKNOWN"

    if "state_age" in state_df.columns:
        state_df["state_age_bucket"] = pd.cut(
            pd.to_numeric(state_df["state_age"], errors="coerce"),
            bins=list(age_bins),
            labels=list(age_labels),
            include_lowest=True,
        ).astype(object).fillna("MISSING")
    else:
        logger.info("rescue_scan target=%s state_age missing; using MISSING", target_state)
        state_df["state_age_bucket"] = "MISSING"

    if "dist_vwap_atr" in state_df.columns:
        state_df["dist_vwap_atr_bucket"] = pd.cut(
            pd.to_numeric(state_df["dist_vwap_atr"], errors="coerce"),
            bins=list(dist_bins),
            labels=list(dist_labels),
            include_lowest=True,
        ).astype(object).fillna("MISSING")
    else:
        logger.info("rescue_scan target=%s dist_vwap_atr missing; using MISSING", target_state)
        state_df["dist_vwap_atr_bucket"] = "MISSING"

    if "ATR_Ratio" in state_df.columns and state_df["ATR_Ratio"].notna().any():
        state_df["atr_ratio_bucket"] = pd.cut(
            pd.to_numeric(state_df["ATR_Ratio"], errors="coerce"),
            bins=list(atr_ratio_bins),
            labels=list(atr_ratio_labels),
            include_lowest=True,
        ).astype(object).fillna("MISSING")
    else:
        state_df["atr_ratio_bucket"] = "MISSING"

    if "BreakMag" in state_df.columns and state_df["BreakMag"].notna().any():
        state_df["breakmag_bucket"] = pd.cut(
            pd.to_numeric(state_df["BreakMag"], errors="coerce"),
            bins=list(break_mag_bins),
            labels=list(break_mag_labels),
            include_lowest=True,
        ).astype(object).fillna("MISSING")
    else:
        state_df["breakmag_bucket"] = "MISSING"

    if "ReentryCount" in state_df.columns and state_df["ReentryCount"].notna().any():
        state_df["reentry_bucket"] = pd.cut(
            pd.to_numeric(state_df["ReentryCount"], errors="coerce"),
            bins=list(reentry_bins),
            labels=list(reentry_labels),
            include_lowest=True,
        ).astype(object).fillna("MISSING")
    else:
        state_df["reentry_bucket"] = "MISSING"

    composition_rows = [
        {
            "bucket": "TOTAL_STATE",
            "n": n_state,
            "pct_total": pct_total,
            "pct_state": 100.0 if n_state else 0.0,
        }
    ]
    if quality_col in state_df.columns:
        quality_counts = state_df[quality_col].fillna("NA").astype(str).value_counts()
        top_quality = quality_counts.head(top_k)
        for label, count in top_quality.items():
            composition_rows.append(
                {
                    "bucket": f"quality:{label}",
                    "n": int(count),
                    "pct_total": (count / total_rows * 100.0) if total_rows else 0.0,
                    "pct_state": (count / n_state * 100.0) if n_state else 0.0,
                }
            )
        if len(quality_counts) > top_k:
            other_count = int(quality_counts.iloc[top_k:].sum())
            composition_rows.append(
                {
                    "bucket": "quality:OTHER",
                    "n": other_count,
                    "pct_total": (other_count / total_rows * 100.0) if total_rows else 0.0,
                    "pct_state": (other_count / n_state * 100.0) if n_state else 0.0,
                }
            )
    else:
        logger.info("rescue_scan target=%s skipped quality (missing %s)", target_state, quality_col)

    composition_df = pd.DataFrame(composition_rows)
    _emit(f"{target_state}_COMPOSITION", composition_df)

    if "ret_struct" not in state_df.columns:
        logger.info("rescue_scan target=%s skipped extended grid (missing ret_struct)", target_state)
        return

    state_df["ret_struct"] = pd.to_numeric(state_df["ret_struct"], errors="coerce")
    state_df_ev = state_df.dropna(subset=["ret_struct"]).copy()
    n_state_ev = len(state_df_ev)
    state_ev_mean = float(state_df_ev["ret_struct"].mean()) if n_state_ev else np.nan
    min_split_samples = max(10, int(n_min / 3))

    axes_map = {
        "BALANCE": [
            "session_bucket",
            "state_age_bucket",
            "dist_vwap_atr_bucket",
            "atr_ratio_bucket",
        ],
        "TRANSITION": [
            "session_bucket",
            "breakmag_bucket",
            "reentry_bucket",
            "state_age_bucket",
            "dist_vwap_atr_bucket",
        ],
    }
    candidate_axes = axes_map.get(
        target_state,
        ["session_bucket", "state_age_bucket", "dist_vwap_atr_bucket"],
    )
    candidate_axes = [
        axis
        for axis in candidate_axes
        if axis in state_df_ev.columns and state_df_ev[axis].notna().any()
    ]
    if not candidate_axes:
        logger.info("rescue_scan target=%s skipped extended grid (no axes)", target_state)
        return

    if len(candidate_axes) <= 3:
        axis_combos = [tuple(candidate_axes)]
    else:
        axis_combos = list(itertools.combinations(candidate_axes, 3))

    time_index: pd.DatetimeIndex | None = None
    if time_col in state_df_ev.columns:
        time_index = pd.DatetimeIndex(pd.to_datetime(state_df_ev[time_col], errors="coerce"))
    elif isinstance(state_df_ev.index, pd.DatetimeIndex):
        time_index = state_df_ev.index

    stability_counts: dict[tuple[Any, ...], int] = {}
    if time_index is not None and n_state_ev:
        split_masks = _split_time_terciles(time_index)
        for mask in split_masks:
            split = state_df_ev.loc[mask]
            split_state_n = int(split["ret_struct"].count())
            if split_state_n < min_split_samples:
                continue
            for combo in axis_combos:
                grouped = split.groupby(list(combo))["ret_struct"].agg(["count", "mean"])
                for key, row in grouped.iterrows():
                    if int(row["count"]) < min_split_samples:
                        continue
                    bucket_key = (",".join(combo),) + (key if isinstance(key, tuple) else (key,))
                    stability_counts[bucket_key] = stability_counts.get(bucket_key, 0) + 1

    rows: list[pd.DataFrame] = []
    for combo in axis_combos:
        grouped = state_df_ev.groupby(list(combo))["ret_struct"]
        summary = grouped.agg(
            n_samples="count",
            ev_mean="mean",
            p10=lambda s: s.quantile(0.10),
            p50=lambda s: s.quantile(0.50),
            p90=lambda s: s.quantile(0.90),
        ).reset_index()
        summary["pct_state"] = (summary["n_samples"] / n_state_ev * 100.0) if n_state_ev else 0.0
        summary["delta_vs_state"] = summary["ev_mean"] - state_ev_mean
        summary["axes"] = ",".join(combo)
        if stability_counts:
            def _stability_row(row: pd.Series) -> int:
                key = (summary["axes"].iloc[0],) + tuple(row[col] for col in combo)
                return stability_counts.get(key, 0)
            summary["stability_splits"] = summary.apply(_stability_row, axis=1)
        else:
            summary["stability_splits"] = 0
        summary = summary[
            ["axes", *combo, "n_samples", "pct_state", "ev_mean", "p10", "p50", "p90", "delta_vs_state", "stability_splits"]
        ]
        rows.append(summary)

    grid_extended = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not grid_extended.empty:
        grid_extended = grid_extended.sort_values(
            by=["pct_state", "n_samples"], ascending=[False, False]
        )
    column_map = {
        "axes": "axis",
        "session_bucket": "ses",
        "state_age_bucket": "age",
        "dist_vwap_atr_bucket": "dvwap",
        "breakmag_bucket": "breakmag",
        "reentry_bucket": "reentry",
        "atr_ratio_bucket": "atr_bucket",
        "n_samples": "n",
        "pct_state": "pct_state",
        "ev_mean": "ev",
        "delta_vs_state": "delta",
        "stability_splits": "stable_n",
    }
    if target_state == "TRANSITION":
        ordered_columns = [
            "axis",
            "ses",
            "age",
            "dvwap",
            "breakmag",
            "reentry",
            "n",
            "pct_state",
            "ev",
            "p10",
            "p50",
            "p90",
            "delta",
            "stable_n",
        ]
    else:
        ordered_columns = [
            "axis",
            "ses",
            "age",
            "dvwap",
            "n",
            "pct_state",
            "ev",
            "p10",
            "p50",
            "p90",
            "delta",
            "stable_n",
            "atr_bucket",
        ]
    grid_display = _format_rescue_table(
        grid_extended.sort_values(by=["n_samples", "ev_mean"], ascending=[False, False]),
        column_map=column_map,
        ordered_columns=ordered_columns,
    )
    _emit(f"{target_state}_RESCUE_GRID_EXTENDED", grid_display, max_rows=30)

    candidates = grid_extended.loc[
        (grid_extended["n_samples"] >= n_min)
        & (grid_extended["delta_vs_state"].abs() <= delta_max)
    ].copy()
    if not candidates.empty:
        candidates = candidates.sort_values(
            by=["pct_state", "n_samples"], ascending=[False, False]
        )
    else:
        logger.info("rescue_scan target=%s no candidates meet filters", target_state)
    candidates_display = _format_rescue_table(
        candidates,
        column_map=column_map,
        ordered_columns=ordered_columns,
    )
    _emit(f"{target_state}_TOP_CANDIDATES", candidates_display, max_rows=top_k)

    if not grid_extended.empty:
        decision_df = grid_extended.copy()
        def _decision(row: pd.Series) -> str:
            if row["n_samples"] >= n_min and row["ev_mean"] > 0 and row["p10"] > 0 and abs(row["delta_vs_state"]) <= delta_max:
                return "KEEP"
            if row["n_samples"] >= n_min and (row["ev_mean"] > 0 or row["p10"] > 0):
                return "REVIEW"
            return "REJECT"
        decision_df["decision"] = decision_df.apply(_decision, axis=1)
        decision_df = decision_df.loc[
            (decision_df["n_samples"] >= n_min)
            & (decision_df["delta_vs_state"].abs() <= delta_max)
        ]
        decision_df = decision_df.sort_values(by=["n_samples", "pct_state"], ascending=[False, False])
    else:
        decision_df = pd.DataFrame(
            {"decision": ["no data"]},
        )
    decision_display = _format_rescue_table(
        decision_df,
        column_map=column_map,
        ordered_columns=[*ordered_columns, "decision"],
    )
    _emit(f"{target_state}_DECISION_TABLE", decision_display, max_rows=top_k)

    group_cols = [
        col
        for col in ["session_bucket", "state_age_bucket", "dist_vwap_atr_bucket"]
        if col in state_df.columns
    ]
    conf_col = next((col for col in confidence_cols if col in state_df.columns), None)
    if conf_col is None:
        logger.info("rescue_scan target=%s skipped confidence summary (missing)", target_state)
        return
    state_df[conf_col] = pd.to_numeric(state_df[conf_col], errors="coerce")
    if not state_df[conf_col].notna().any():
        logger.info("rescue_scan target=%s skipped confidence summary (empty)", target_state)
        return

    conf_summary = (
        state_df.groupby(group_cols, dropna=False)[conf_col]
        .agg(
            mean="mean",
            p10=lambda s: s.quantile(0.10),
            p50=lambda s: s.quantile(0.50),
            p90=lambda s: s.quantile(0.90),
        )
        .reset_index()
    )
    _emit(f"{target_state}_CONFIDENCE_SUMMARY", conf_summary)


def _build_ev_extended_table(
    ev_frame: pd.DataFrame,
    outputs: pd.DataFrame,
    gating: pd.DataFrame,
    ctx_features: pd.DataFrame,
    min_hvc_samples: int,
    logger: logging.Logger,
    warnings_state: dict[str, bool],
) -> tuple[pd.DataFrame, dict[str, int], pd.DataFrame]:
    base = ev_frame[["ret_struct", "state_hat"]].copy()
    state_name = base["state_hat"].map(lambda v: StateLabels(v).name if not pd.isna(v) else "NA")
    base["state"] = state_name
    if "quality_label" in outputs.columns:
        base["quality_label"] = outputs["quality_label"].reindex(base.index).fillna("NA")
    else:
        base["quality_label"] = "NA"
    if "ctx_session_bucket" in ctx_features.columns and ctx_features["ctx_session_bucket"].notna().any():
        base["ctx_session_bucket"] = (
            ctx_features["ctx_session_bucket"].reindex(base.index).fillna("NA")
        )
    else:
        if not warnings_state.get("missing_session_bucket", False):
            logger.warning(
                "ctx_session_bucket missing; fallback to timestamp hour bucket (tz-naive)."
            )
            warnings_state["missing_session_bucket"] = True
        base["ctx_session_bucket"] = _session_bucket_from_timestamp(base.index)
    if "ctx_state_age" in ctx_features.columns and ctx_features["ctx_state_age"].notna().any():
        base["ctx_state_age_bucket"] = _bucketize(
            ctx_features["ctx_state_age"].reindex(base.index),
            bins=[0, 2, 5, 10, np.inf],
            labels=["0-2", "3-5", "6-10", "11+"],
        )
    else:
        base["ctx_state_age_bucket"] = "NA"
    if "ctx_dist_vwap_atr" in ctx_features.columns and ctx_features["ctx_dist_vwap_atr"].notna().any():
        base["ctx_dist_vwap_atr_bucket"] = _bucketize(
            ctx_features["ctx_dist_vwap_atr"].reindex(base.index),
            bins=[-np.inf, 0.5, 1.0, 2.0, np.inf],
            labels=["<=0.5", "0.5-1", "1-2", ">2"],
        )
    else:
        base["ctx_dist_vwap_atr_bucket"] = "NA"

    allow_cols = [col for col in gating.columns if col.startswith("ALLOW_")]
    if not allow_cols:
        if not warnings_state.get("missing_allow_cols", False):
            logger.warning("No ALLOW_* columns found; skipping EV extended allow conditioning.")
            warnings_state["missing_allow_cols"] = True
        coverage = {
            "total_rows_used": int(len(base)),
            "total_rows_allowed": 0,
            "rows_before_explode": 0,
            "rows_after_explode": 0,
        }
        return pd.DataFrame(), coverage, pd.DataFrame()
    allow_df = gating[allow_cols].reindex(base.index).fillna(False).astype(bool)
    allow_rules = allow_df.apply(
        lambda row: [col for col, val in row.items() if bool(val)],
        axis=1,
    )
    allowed_mask = allow_rules.map(bool)
    before_explode_rows = int(allowed_mask.sum())
    exploded = base.loc[allowed_mask].assign(allow_rule=allow_rules.loc[allowed_mask]).explode(
        "allow_rule"
    )
    after_explode_rows = int(len(exploded))
    coverage = {
        "total_rows_used": int(len(base)),
        "total_rows_allowed": int(before_explode_rows),
        "rows_before_explode": before_explode_rows,
        "rows_after_explode": after_explode_rows,
    }
    if exploded.empty:
        return pd.DataFrame(), coverage, pd.DataFrame()

    group_cols = [
        "state",
        "quality_label",
        "allow_rule",
        "ctx_session_bucket",
        "ctx_state_age_bucket",
        "ctx_dist_vwap_atr_bucket",
    ]
    state_means = exploded.groupby("state")["ret_struct"].mean()

    grouped = exploded.groupby(group_cols)["ret_struct"]
    summary = grouped.agg(
        n_samples="count",
        ev_mean="mean",
        ev_p10=lambda s: s.quantile(0.1),
        ev_p50="median",
        winrate=lambda s: float((s > 0).mean() * 100.0),
    ).reset_index()
    summary["uplift_vs_state"] = summary.apply(
        lambda row: row["ev_mean"] - state_means.get(row["state"], np.nan),
        axis=1,
    )
    summary = summary[summary["n_samples"] >= min_hvc_samples].reset_index(drop=True)
    summary = summary[
        [
            "state",
            "allow_rule",
            "quality_label",
            "ctx_session_bucket",
            "ctx_state_age_bucket",
            "ctx_dist_vwap_atr_bucket",
            "n_samples",
            "ev_mean",
            "ev_p10",
            "ev_p50",
            "winrate",
            "uplift_vs_state",
        ]
    ]
    return summary, coverage, exploded


def _build_ev_extended_stability(
    exploded: pd.DataFrame,
    ev_extended: pd.DataFrame,
    ev_min_split_samples: int,
) -> pd.DataFrame:
    if exploded.empty or ev_extended.empty:
        return pd.DataFrame()

    group_cols = [
        "state",
        "quality_label",
        "allow_rule",
        "ctx_session_bucket",
        "ctx_state_age_bucket",
        "ctx_dist_vwap_atr_bucket",
    ]
    valid_keys = set(tuple(row) for row in ev_extended[group_cols].to_numpy())
    index = exploded.sort_index().index
    split_masks = _split_time_terciles(index)
    uplifts_by_group: dict[tuple[Any, ...], list[float]] = {}
    for mask in split_masks:
        split = exploded.loc[mask]
        if split.empty:
            continue
        state_counts = split.groupby("state")["ret_struct"].count()
        state_means = split.groupby("state")["ret_struct"].mean()
        group_counts = split.groupby(group_cols)["ret_struct"].count()
        group_means = split.groupby(group_cols)["ret_struct"].mean()
        for key, ev_mean in group_means.items():
            state_key = key[0]
            if key not in valid_keys:
                continue
            if (
                state_key not in state_means.index
                or state_counts.get(state_key, 0) < ev_min_split_samples
                or group_counts.get(key, 0) < ev_min_split_samples
            ):
                continue
            uplift = ev_mean - state_means.loc[state_key]
            uplifts_by_group.setdefault(key, []).append(float(uplift))

    rows: list[dict[str, Any]] = []
    for key, uplifts in uplifts_by_group.items():
        uplifts_arr = np.array(uplifts, dtype=float)
        rows.append(
            {
                "state": key[0],
                "quality_label": key[1],
                "allow_rule": key[2],
                "ctx_session_bucket": key[3],
                "ctx_state_age_bucket": key[4],
                "ctx_dist_vwap_atr_bucket": key[5],
                "n_splits_present": int(len(uplifts_arr)),
                "uplift_mean": float(np.mean(uplifts_arr)) if len(uplifts_arr) else np.nan,
                "uplift_std": float(np.std(uplifts_arr)) if len(uplifts_arr) else np.nan,
                "pct_splits_uplift_pos": float(np.mean(uplifts_arr > 0) * 100.0)
                if len(uplifts_arr)
                else 0.0,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    
    def _safe_symbol(sym: str) -> str:
    # deja letras/números/._- y reemplaza lo demás por _
        return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in sym)

    if args.model_out is None:
        sym = _safe_symbol(args.symbol)
        args.model_out = args.model_dir / f"{sym}_state_engine.pkl"
        
    rich_modules = try_import_rich() if not args.no_rich else None
    logger = setup_logging(args.logging_level)
    report_rank = REPORT_LEVELS[args.log_level]
    is_verbose = report_rank >= REPORT_LEVELS["verbose"]
    is_debug = report_rank >= REPORT_LEVELS["debug"]
    warnings_state: dict[str, bool] = {}
    logging.getLogger("state_engine.context_features").setLevel(logging.WARNING)

    console = None
    use_rich = bool(rich_modules)
    if use_rich and rich_modules:
        console = rich_modules["Console"]()

    bars_per_hour = {
        "M30": 2,
        "H1": 1,
        "H2": 0.5,
        "H4": 0.25,
    }
    derived_bars = math.ceil(args.window_hours * bars_per_hour[args.timeframe])
    if derived_bars < 4:
        raise ValueError(f"Context window too small: {derived_bars} bars (min=4).")
    logger.info(
        "timeframe=%s window_hours=%s derived_bars=%s",
        args.timeframe,
        args.window_hours,
        derived_bars,
    )

    timeframe_floor = {
        "M30": "30min",
        "H1": "h",
        "H2": "2h",
        "H4": "4h",
    }[args.timeframe]

    def step(name: str) -> float:
        logger.info("stage=%s", name)
        return datetime.now(tz=timezone.utc).timestamp()

    def step_done(start_ts: float) -> float:
        return datetime.now(tz=timezone.utc).timestamp() - start_ts

    stage_start = step("descarga_tf")
    connector = MT5Connector()
    fecha_inicio = pd.to_datetime(args.start)
    fecha_fin = pd.to_datetime(args.end)

    ohlcv = connector.obtener_ohlcv(args.symbol, args.timeframe, fecha_inicio, fecha_fin)
    
    # 1) Hora servidor MT5 (inferida desde último tick)
    server_now = connector.server_now(args.symbol).tz_localize(None)
    
    # 2) Guardrail correcto: usar solo velas cerradas según hora servidor
    #    La vela en formación empieza en server_now.floor(timeframe)
    cutoff = server_now.floor(timeframe_floor)
    ohlcv = ohlcv[ohlcv.index < cutoff]
    
    # 3) Timestamp última vela usada
    last_bar_ts = pd.Timestamp(ohlcv.index.max()).tz_localize(None)
    
    # 4) Edad en minutos (server_now - last_bar_ts)
    bar_age_minutes = (server_now - last_bar_ts).total_seconds() / 60.0
    
    # 5) Diagnóstico: si el tick está viejo, esto será grande
    now_utc = pd.Timestamp.utcnow().tz_localize(None)
    tick_age_min_utc = (now_utc - server_now).total_seconds() / 60.0

    elapsed_download = step_done(stage_start)
    logger.info("download_rows=%s elapsed=%.2fs", len(ohlcv), elapsed_download)

    stage_start = step("build_dataset")
    feature_config = FeatureConfig(window=derived_bars)
    builder = DatasetBuilder(feature_config=feature_config)
    artifacts = builder.build(ohlcv)
    full_features = artifacts.full_features
    features_raw = artifacts.features
    labels_raw = artifacts.labels
    elapsed_build = step_done(stage_start)
    logger.info(
        "features_raw=%s labels_raw=%s elapsed=%.2fs",
        len(features_raw),
        len(labels_raw),
        elapsed_build,
    )

    stage_start = step("align_and_clean")
    aligned = features_raw.join(labels_raw.rename("label"), how="inner")
    dropped_nan = int(aligned.isna().any(axis=1).sum())
    aligned = aligned.dropna()
    features = aligned.drop(columns=["label"])
    labels = aligned["label"].astype(int)
    elapsed_clean = step_done(stage_start)
    logger.info("aligned_samples=%s dropped_nan=%s elapsed=%.2fs", len(aligned), dropped_nan, elapsed_clean)
    logger.info("n_features=%s feature_names=%s", features.shape[1], features.columns.tolist())

    if len(features) < args.min_samples:
        raise RuntimeError(f"Not enough samples to train: {len(features)} < {args.min_samples}")

    stage_start = step("split")
    n_total = len(features)
    n_train = int(n_total * args.split_ratio)
    features_train = features.iloc[:n_train]
    labels_train = labels.iloc[:n_train]
    features_test = features.iloc[n_train:]
    labels_test = labels.iloc[n_train:]
    elapsed_split = step_done(stage_start)
    logger.info(
        "n_train=%s n_test=%s split_ratio=%.2f elapsed=%.2fs",
        len(features_train),
        len(features_test),
        args.split_ratio,
        elapsed_split,
    )

    label_order = [StateLabels.BALANCE, StateLabels.TRANSITION, StateLabels.TREND]

    stage_start = step("train_model")
    model_config = StateEngineModelConfig(
        class_weight="balanced" if args.class_weight_balanced else None,
    )
    model = StateEngineModel(model_config)
    model.fit(features_train, labels_train)
    elapsed_train = step_done(stage_start)
    logger.info("train_elapsed=%.2fs", elapsed_train)

    stage_start = step("evaluate")
    preds_test = model.predict_state(features_test).to_numpy()
    labels_test_np = labels_test.to_numpy()
    matrix = confusion_matrix(labels_test_np, preds_test, label_order)
    acc = float(np.mean(preds_test == labels_test_np)) if len(labels_test_np) else 0.0
    f1 = f1_macro(matrix)
    elapsed_eval = step_done(stage_start)
    logger.info("accuracy=%.4f f1_macro=%.4f elapsed=%.2fs", acc, f1, elapsed_eval)

    stage_start = step("predict_outputs")
    outputs = model.predict_outputs(features)
    elapsed_outputs = step_done(stage_start)
    logger.info("outputs_rows=%s elapsed=%.2fs", len(outputs), elapsed_outputs)

    quality_diagnostics = None
    quality_warnings = []
    if args.quality:
        quality_config, quality_sources, quality_warnings = load_quality_config(
            args.symbol,
            args.quality_config,
        )
        quality_features = full_features.reindex(outputs.index)
        quality_labels, quality_warnings_assign = assign_quality_labels(
            outputs["state_hat"],
            quality_features,
            quality_config,
        )
        outputs["quality_label"] = quality_labels
        quality_diagnostics = build_quality_diagnostics(
            outputs["state_hat"],
            quality_labels,
            quality_sources,
            [*quality_warnings, *quality_warnings_assign],
        )

    ctx_features = build_context_features(
        ohlcv,
        outputs,
        symbol=args.symbol,
        timeframe=args.timeframe,
    )
    if ctx_features is None:
        ctx_features = pd.DataFrame(index=outputs.index)
    else:
        ctx_features = ctx_features.reindex(outputs.index)
    if is_debug:
        logger.info("ctx_features_cols=%s", list(ctx_features.columns))
        logger.info("ctx_features_nonnull_tail=\n%s", ctx_features.tail(5).to_string())
    ctx_cols = [col for col in ctx_features.columns if col.startswith("ctx_")]
    if ctx_features.empty or not ctx_cols:
        logger.warning(
            "ctx_features empty or missing ctx_* columns; context gating filters will be inactive."
        )

    symbol_config = _load_symbol_config(args.symbol, logger)
    gating_thresholds, _ = build_transition_gating_thresholds(
        args.symbol,
        symbol_config,
        logger=logger,
    )
    gating_config_meta: dict[str, Any] = {}
    if isinstance(symbol_config, dict):
        allow_cfg = symbol_config.get("allow_context_filters")
        if isinstance(allow_cfg, dict) and allow_cfg:
            config_path = Path("configs") / "symbols" / f"{args.symbol}.yaml"
            config_path_str = str(config_path) if config_path.exists() else "unknown"
            allow_meta: dict[str, dict[str, Any]] = {}
            for allow_name, rule_cfg in allow_cfg.items():
                if not isinstance(rule_cfg, dict):
                    continue
                rule_meta = dict(rule_cfg)
                rule_meta.update(
                    {
                        "source": "symbol",
                        "selected_path": f"allow_context_filters.{allow_name}",
                        "keys_present": sorted(rule_cfg.keys()),
                        "config_path": config_path_str,
                    }
                )
                allow_meta[allow_name] = rule_meta
            if allow_meta:
                gating_config_meta["allow_context_filters"] = allow_meta

    # Extra reporting helpers
    state_hat_dist = class_distribution(outputs["state_hat"].to_numpy(), label_order)
    q_list = [0, 50, 75, 90, 95, 99, 100]
    breakmag_p = percentiles(full_features["BreakMag"], q_list) if "BreakMag" in full_features.columns else {str(q): None for q in q_list}
    reentry_p = percentiles(full_features["ReentryCount"], q_list) if "ReentryCount" in full_features.columns else {str(q): None for q in q_list}

    stage_start = step("gating")
    logger.info("gating_module=%s", GatingPolicy.__module__)
    gating_mod = importlib.import_module(GatingPolicy.__module__)
    logger.info("gating_file=%s", getattr(gating_mod, "__file__", None))
    logger.info("context_features_module=%s", build_context_features.__module__)
    context_mod = importlib.import_module(build_context_features.__module__)
    logger.info("context_features_file=%s", getattr(context_mod, "__file__", None))
    gating_policy = GatingPolicy(gating_thresholds)
    features_for_gating = full_features.join(ctx_features, how="left").reindex(outputs.index)
    validate_allow_context_requirements(
        gating_config_meta,
        set(features_for_gating.columns) | set(outputs.columns),
        logger=logger,
    )
    gating = gating_policy.apply(
        outputs,
        features_for_gating,
        logger=logger,
        symbol=args.symbol,
        config_meta=gating_config_meta,
    )
    allow_cols = list(gating.columns)
    allow_any = gating.any(axis=1)
    allow_context_frame = pd.DataFrame(index=outputs.index).join(gating[allow_cols], how="left")

    def _attach_ctx_column(name: str, candidates: list[str]) -> None:
        series = None
        for candidate in candidates:
            if candidate in ctx_features.columns:
                series = ctx_features[candidate]
                break
            if candidate in outputs.columns:
                series = outputs[candidate]
                break
            if candidate in full_features.columns:
                series = full_features[candidate]
                break
        if series is not None:
            allow_context_frame[name] = series.reindex(outputs.index)

    _attach_ctx_column("session_bucket", ["ctx_session_bucket", "session_bucket", "session"])
    _attach_ctx_column("state_age", ["ctx_state_age", "state_age"])
    _attach_ctx_column("dist_vwap_atr", ["ctx_dist_vwap_atr", "dist_vwap_atr", "ctx_dist_vwap_atr_abs"])

    # EV estructural (diagnóstico): ret_struct basado en rango direccional futuro
    ev_k_bars = 2
    min_hvc_samples = 100
    high = ohlcv["high"]
    low = ohlcv["low"]
    close = ohlcv["close"]
    future_high = high.shift(-1).rolling(window=ev_k_bars, min_periods=ev_k_bars).max().shift(-(ev_k_bars - 1))
    future_low = low.shift(-1).rolling(window=ev_k_bars, min_periods=ev_k_bars).min().shift(-(ev_k_bars - 1))
    up_move = np.log(future_high / close)
    down_move = np.log(close / future_low)
    ret_struct = (up_move - down_move).rename("ret_struct").reindex(outputs.index)
    ev_frame = pd.DataFrame(
        {
            "ret_struct": ret_struct,
            "state_hat": outputs["state_hat"],
        },
        index=outputs.index,
    ).join(gating, how="left")
    ev_frame = ev_frame.dropna(subset=["ret_struct"])

    ev_base = float(ev_frame["ret_struct"].mean()) if not ev_frame.empty else 0.0
    ev_by_state: list[dict[str, Any]] = []
    for label in label_order:
        mask_state = ev_frame["state_hat"] == label
        state_ret = ev_frame.loc[mask_state, "ret_struct"]
        ev_by_state.append(
            {
                "state": label.name,
                "n_samples": int(mask_state.sum()),
                "ev": float(state_ret.mean()) if not state_ret.empty else 0.0,
            }
        )

    ev_extended_table, ev_extended_coverage, ev_extended_exploded = _build_ev_extended_table(
        ev_frame=ev_frame,
        outputs=outputs,
        gating=gating,
        ctx_features=ctx_features,
        min_hvc_samples=min_hvc_samples,
        logger=logger,
        warnings_state=warnings_state,
    )
    ev_extended_stability = _build_ev_extended_stability(
        ev_extended_exploded,
        ev_extended_table,
        args.ev_min_split_samples,
    )
    
    # --- "Última vela" para reporting
    last_idx = outputs.index.max()
    
    # ALLOW final (cualquier regla true)
    last_allow = bool(allow_any.loc[last_idx]) if last_idx in allow_any.index else False
    
    gating_allow_rate = float(allow_any.mean()) if len(gating) else 0.0
    gating_block_rate = 1.0 - gating_allow_rate
    elapsed_gating = step_done(stage_start)
    logger.info("gating_allow_rate=%.2f%% elapsed=%.2fs", gating_allow_rate * 100, elapsed_gating)
    gating_thresholds = asdict(gating_policy.thresholds)
    logger.info("gating_thresholds=%s", gating_thresholds)
    required_threshold_fields = {
        "allowed_sessions",
        "state_age_min",
        "state_age_max",
        "dist_vwap_atr_min",
        "dist_vwap_atr_max",
    }
    if not required_threshold_fields.issubset(gating_thresholds.keys()):
        logger.error(
            "Loaded gating.py does not include context thresholds fields. Check PYTHONPATH / repo root."
        )

    if is_debug:
        table_class = rich_modules["Table"] if use_rich and console else None

        def _emit_table_debug(title: str, df: pd.DataFrame, max_rows: int | None = None) -> None:
            _print_block(
                title,
                df,
                console=console if use_rich else None,
                table_class=table_class,
                logger=logger,
                max_rows=max_rows,
            )

        if ctx_cols:
            allow_cols = [col for col in gating.columns if col.startswith("ALLOW_")]
            debug_frame = (
                outputs[["state_hat", "margin"]]
                .join(ctx_features[ctx_cols], how="left")
                .join(full_features[["BreakMag", "ReentryCount"]], how="left")
                .join(gating[allow_cols], how="left")
            )
            logger.info("[CTX DIAGNOSTIC TABLE] last_rows=10\n%s", debug_frame.tail(10).to_string())
        else:
            logger.warning("ctx_features missing ctx_* columns; skipping ctx diagnostic table.")

        allow_cfg = symbol_config.get("allow_context_filters", {}) if isinstance(symbol_config, dict) else {}
        transition_cfg = allow_cfg.get("ALLOW_transition_failure", {}) if isinstance(allow_cfg, dict) else {}
        allow_config_rows = _allow_filter_config_rows(symbol_config, "ALLOW_transition_failure")
        if allow_config_rows:
            allow_config_df = pd.DataFrame(allow_config_rows)
            _emit_table_debug("GATING_CONFIG_EFFECTIVE", allow_config_df)
        if isinstance(transition_cfg, dict) and transition_cfg:
            allow_filter_counts = _allow_context_filter_counts(
                allow_context_frame,
                "ALLOW_transition_failure",
                transition_cfg,
                logger,
            )
            if not allow_filter_counts.empty:
                _emit_table_debug("ALLOW_transition_context_counts", allow_filter_counts)

    stage_start = step("save_model")
    metadata = {
        "feature_names_used": features.columns.tolist(),
        "label_names": {label.name: int(label) for label in StateLabels},
        "classes": [label.name for label in StateLabels],
        "feature_config": asdict(feature_config),
        "trained_at": datetime.now(tz=timezone.utc).isoformat(),
        "symbol": args.symbol,
        "start": args.start,
        "end": args.end,
        "timeframe": args.timeframe,
        "window_hours": args.window_hours,
        "derived_bars": derived_bars,
        "n_samples": len(features),
        "n_train": len(features_train),
        "n_test": len(features_test),
        "split_ratio": args.split_ratio,
        "accuracy": acc,
        "f1_macro": f1,
        "latest_bar": {
            "last_bar_time": str(last_bar_ts),
            "server_now": str(server_now),
            "bar_age_minutes": float(bar_age_minutes),
            "tick_age_min_vs_utc": float(tick_age_min_utc),
        },
    }
    model.save(args.model_out, metadata=metadata)
    elapsed_save = step_done(stage_start)
    logger.info("model_path=%s elapsed=%.2fs", str(args.model_out), elapsed_save)

    # Baseline (majority class)
    label_dist = class_distribution(labels.to_numpy(), label_order)
    baseline = max(label_dist, key=lambda r: r["count"]) if label_dist else None
    baseline_label = baseline["label"] if baseline else "NA"
    baseline_pct = baseline["pct"] if baseline else 0.0
    logger.info("baseline_state=%s baseline_pct=%.2f", baseline_label, baseline_pct)

    importances = model.feature_importances().head(15)

    # Guardrail detail counts (kept as in your existing report)
    gating_thresholds = gating_policy.thresholds
    transition_rule = outputs["margin"] >= gating_thresholds.transition_margin_min
    transition_break = full_features["BreakMag"] >= gating_thresholds.transition_breakmag_min
    transition_reentry = full_features["ReentryCount"] >= gating_thresholds.transition_reentry_min

    table_class = rich_modules["Table"] if use_rich and console else None

    def _emit_table(title: str, df: pd.DataFrame, max_rows: int | None = None) -> None:
        _print_block(
            title,
            df,
            console=console if use_rich else None,
            table_class=table_class if use_rich else None,
            logger=logger,
            max_rows=max_rows,
        )

    if is_verbose or is_debug:
        summary_lines = [
            f"Symbol: {args.symbol}",
            f"Period: {args.start} -> {args.end}",
            f"Timeframe: {args.timeframe}",
            f"Window: {args.window_hours}h (~{derived_bars} bars)",
            f"Samples: {len(features)} (train={len(features_train)}, test={len(features_test)})",
            f"Baseline: {baseline_label} ({baseline_pct:.2f}%)",
            f"Gating allow rate: {gating_allow_rate*100:.2f}%",
        ]

        if console and use_rich:
            console.print("=== State Engine Training Summary ===")
            for line in summary_lines:
                console.print(line)
        else:
            logger.info("=== State Engine Training Summary ===\n%s", "\n".join(summary_lines))

    if is_verbose:
        state_hat_df = pd.DataFrame(state_hat_dist)
        _emit_table("Distribución state_hat (predicción)", state_hat_df)

        if quality_diagnostics is not None:
            quality_dist_df = pd.DataFrame(quality_diagnostics.distribution_rows)
            _emit_table("QUALITY_DISTRIBUTION", quality_dist_df)
            quality_splits_df = pd.DataFrame(quality_diagnostics.split_rows)
            _emit_table("QUALITY_SPLITS", quality_splits_df)

            incomplete_warnings = [w for w in quality_warnings if w.startswith("config_incomplete")]
            missing_total = 0
            for warning in incomplete_warnings:
                if "missing=" in warning:
                    missing_str = warning.split("missing=", 1)[1]
                    try:
                        missing_list = ast.literal_eval(missing_str)
                    except (ValueError, SyntaxError):
                        missing_list = []
                    missing_total += len(missing_list)
            quality_summary = pd.DataFrame(
                [
                    {
                        "config_incomplete": bool(incomplete_warnings),
                        "missing_count": missing_total,
                    }
                ]
            )
            _emit_table("QUALITY_CONFIG_EFFECTIVE (summary)", quality_summary)

    if is_debug and quality_diagnostics is not None:
        quality_config_df = pd.DataFrame(
            [
                {"key": key, "source": src}
                for key, src in sorted(quality_diagnostics.config_sources.items())
            ]
        )
        _emit_table("QUALITY_CONFIG_EFFECTIVE", quality_config_df)
        quality_persist_df = pd.DataFrame(quality_diagnostics.persistence_rows)
        _emit_table("QUALITY_PERSISTENCE", quality_persist_df)
        quality_warn_df = pd.DataFrame({"warning": quality_diagnostics.warnings or ["none"]})
        _emit_table("QUALITY_WARNINGS", quality_warn_df)

    ev_base_df = pd.DataFrame(
        [
            {
                "EV_BASE": ev_base,
                "k_bars": ev_k_bars,
                "n_samples": len(ev_frame),
            }
        ]
    )
    _emit_table("EV_BASE", ev_base_df)

    ev_state_df = pd.DataFrame(ev_by_state)
    _emit_table("EV_state", ev_state_df)

    if not ev_extended_exploded.empty:
        ev_hvc_grouped = (
            ev_extended_exploded.groupby(["state", "allow_rule"])["ret_struct"]
            .agg(n_samples="count", ev_mean="mean")
            .reset_index()
        )
        ev_state_map = {
            row["state"]: row["ev"] for row in ev_by_state
        }
        ev_hvc_grouped["uplift_vs_state"] = ev_hvc_grouped.apply(
            lambda row: row["ev_mean"] - ev_state_map.get(row["state"], np.nan),
            axis=1,
        )
        ev_hvc_grouped = ev_hvc_grouped[ev_hvc_grouped["n_samples"] >= min_hvc_samples]
    else:
        ev_hvc_grouped = pd.DataFrame(columns=["state", "allow_rule", "n_samples", "ev_mean", "uplift_vs_state"])
    _emit_table("EV_HVC (state, allow)", ev_hvc_grouped)

    if args.diagnose_rescue_scans:
        diagnostic_cfg = _load_diagnostic_table_config(args.symbol, logger)
        diagnostic_bins = diagnostic_cfg.get("bins", {})
        if diagnostic_bins is None or not isinstance(diagnostic_bins, dict):
            diagnostic_bins = {}
        rescue_top_k = int(diagnostic_cfg.get("top_k", args.rescue_top_k))
        rescue_n_min = int(diagnostic_cfg.get("n_min", args.rescue_n_min))
        rescue_delta_max = float(diagnostic_cfg.get("delta_max", args.rescue_delta_max))
        age_bins, age_labels = _coerce_bins(
            diagnostic_bins.get("state_age"),
            (-float("inf"), 2, 5, 10, float("inf")),
            ("0-2", "3-5", "6-10", "11+"),
        )
        dist_bins, dist_labels = _coerce_bins(
            diagnostic_bins.get("dist_vwap_atr"),
            (-float("inf"), 0.5, 1.0, 2.0, float("inf")),
            ("<=0.5", "0.5-1", "1-2", ">2"),
        )
        breakmag_bins, breakmag_labels = _coerce_bins(
            diagnostic_bins.get("breakmag"),
            (-float("inf"), 0.5, 1.0, 1.5, 2.5, float("inf")),
            ("<=0.5", "0.5-1", "1-1.5", "1.5-2.5", ">2.5"),
        )
        reentry_bins, reentry_labels = _coerce_bins(
            diagnostic_bins.get("reentry"),
            (-float("inf"), 0.5, 1.5, 2.5, 4.5, float("inf")),
            ("0", "1", "2", "3-4", "5+"),
        )
        df_outputs = outputs.copy()
        df_outputs["state"] = outputs["state_hat"].map(
            lambda v: StateLabels(v).name if not pd.isna(v) else "NA"
        )
        df_outputs["ret_struct"] = ev_frame["ret_struct"].reindex(outputs.index)
        if "ctx_session_bucket" in ctx_features.columns:
            df_outputs["session_bucket"] = ctx_features["ctx_session_bucket"]
        if "ctx_state_age" in ctx_features.columns:
            df_outputs["state_age"] = ctx_features["ctx_state_age"]
        if "ctx_dist_vwap_atr" in ctx_features.columns:
            df_outputs["dist_vwap_atr"] = ctx_features["ctx_dist_vwap_atr"]
        for col in ("ATR_Ratio", "BreakMag", "ReentryCount"):
            if col in full_features.columns:
                df_outputs[col] = full_features[col].reindex(outputs.index)
        df_outputs["time"] = df_outputs.index
        _rescue_scan_tables(
            df_outputs,
            target_state="BALANCE",
            quality_col="quality_label",
            top_k=rescue_top_k,
            n_min=rescue_n_min,
            delta_max=rescue_delta_max,
            age_bins=age_bins,
            age_labels=age_labels,
            dist_bins=dist_bins,
            dist_labels=dist_labels,
            break_mag_bins=breakmag_bins,
            break_mag_labels=breakmag_labels,
            reentry_bins=reentry_bins,
            reentry_labels=reentry_labels,
            logger=logger,
            console=console if use_rich else None,
            table_class=table_class if use_rich else None,
            rescue_output_dir=PROJECT_ROOT / "state_engine" / "models" / "rescue",
        )
        _rescue_scan_tables(
            df_outputs,
            target_state="TRANSITION",
            quality_col="quality_label",
            top_k=rescue_top_k,
            n_min=rescue_n_min,
            delta_max=rescue_delta_max,
            age_bins=age_bins,
            age_labels=age_labels,
            dist_bins=dist_bins,
            dist_labels=dist_labels,
            break_mag_bins=breakmag_bins,
            break_mag_labels=breakmag_labels,
            reentry_bins=reentry_bins,
            reentry_labels=reentry_labels,
            logger=logger,
            console=console if use_rich else None,
            table_class=table_class if use_rich else None,
            rescue_output_dir=PROJECT_ROOT / "state_engine" / "models" / "rescue",
        )

    group_cols = [
        "state",
        "quality_label",
        "allow_rule",
        "ctx_session_bucket",
        "ctx_state_age_bucket",
        "ctx_dist_vwap_atr_bucket",
    ]
    if not ev_extended_stability.empty:
        stability_ranked = ev_extended_stability.merge(
            ev_extended_table[group_cols + ["n_samples"]],
            on=group_cols,
            how="left",
        )
        stability_ranked["n_splits_present"] = stability_ranked["n_splits_present"].fillna(0)
        stability_ranked["pct_splits_uplift_pos"] = stability_ranked["pct_splits_uplift_pos"].fillna(0.0)
        stability_ranked["uplift_mean"] = stability_ranked["uplift_mean"].fillna(0.0)
        stability_ranked["n_samples"] = stability_ranked["n_samples"].fillna(0)
        stability_ranked = stability_ranked.sort_values(
            by=[
                "n_splits_present",
                "pct_splits_uplift_pos",
                "uplift_mean",
                "n_samples",
            ],
            ascending=[False, False, False, False],
        )
        top_stability = stability_ranked.head(20)
        top_extended = ev_extended_table.merge(
            stability_ranked[group_cols],
            on=group_cols,
            how="inner",
        )
        top_extended = top_extended.set_index(group_cols).loc[
            top_stability.set_index(group_cols).index
        ].reset_index()
    else:
        top_stability = ev_extended_stability
        top_extended = ev_extended_table.head(20)

    if not ev_extended_table.empty:
        _emit_table("EV_EXTENDED_TABLE (top 20)", top_extended, max_rows=20)
        if not top_stability.empty:
            _emit_table("EV_EXTENDED_STABILITY (top 20)", top_stability, max_rows=20)

    coverage_rows = [
        {
            "total_rows_used": ev_extended_coverage.get("total_rows_used", 0),
            "total_rows_allowed": ev_extended_coverage.get("total_rows_allowed", 0),
            "rows_before_explode": ev_extended_coverage.get("rows_before_explode", 0),
            "rows_after_explode": ev_extended_coverage.get("rows_after_explode", 0),
            "coverage_before_explode_pct": (
                ev_extended_coverage.get("rows_before_explode", 0)
                / max(ev_extended_coverage.get("total_rows_used", 1), 1)
                * 100.0
            ),
            "coverage_after_explode_pct": (
                ev_extended_coverage.get("rows_after_explode", 0)
                / max(ev_extended_coverage.get("total_rows_used", 1), 1)
                * 100.0
            ),
        }
    ]
    _emit_table("COVERAGE_SUMMARY", pd.DataFrame(coverage_rows))
    if is_debug:
        state_cov_df = pd.DataFrame(
            ev_frame["state_hat"].map(lambda v: StateLabels(v).name).value_counts()
        ).reset_index()
        state_cov_df.columns = ["state_hat", "count"]
        _emit_table("COVERAGE_STATE", state_cov_df)

        if "quality_label" in outputs.columns:
            quality_cov_df = outputs.loc[ev_frame.index, "quality_label"].value_counts().reset_index()
            quality_cov_df.columns = ["quality_label", "count"]
            _emit_table("COVERAGE_QUALITY", quality_cov_df)

        if not ev_extended_exploded.empty:
            allow_cov_df = ev_extended_exploded["allow_rule"].value_counts().reset_index()
            allow_cov_df.columns = ["allow_rule", "count"]
            _emit_table("COVERAGE_ALLOW", allow_cov_df)

        if "ctx_session_bucket" in ctx_features.columns:
            session_cov_df = ctx_features.loc[ev_frame.index, "ctx_session_bucket"].value_counts().reset_index()
            session_cov_df.columns = ["ctx_session_bucket", "count"]
            _emit_table("COVERAGE_SESSION", session_cov_df)

    if is_verbose:
        if "ctx_dist_vwap_atr" in ctx_features.columns:
            dist_series = pd.to_numeric(
                ctx_features.loc[ev_frame.index, "ctx_dist_vwap_atr"], errors="coerce"
            )
            vwap_summary = pd.DataFrame(
                [
                    {
                        "nan_pct": float(dist_series.isna().mean() * 100.0),
                        "dist_nan_pct": float(dist_series.isna().mean() * 100.0),
                        "dist_p50": float(dist_series.quantile(0.5)) if dist_series.notna().any() else np.nan,
                        "dist_p90": float(dist_series.quantile(0.9)) if dist_series.notna().any() else np.nan,
                    }
                ]
            )
            _emit_table("VWAP_VALIDATION (summary)", vwap_summary)

    if is_debug:
        class_dist_df = pd.DataFrame(label_dist)
        _emit_table("Distribución de clases", class_dist_df)
        percentiles_df = pd.DataFrame(
            [
                {"metric": "BreakMag", **breakmag_p},
                {"metric": "ReentryCount", **reentry_p},
            ]
        )
        _emit_table("Percentiles (BreakMag / ReentryCount)", percentiles_df)
        metrics_df = pd.DataFrame([{"accuracy": acc, "f1_macro": f1}])
        _emit_table("Métricas (Test)", metrics_df)
        confusion_df = pd.DataFrame(
            matrix,
            columns=[l.name for l in label_order],
            index=[l.name for l in label_order],
        ).reset_index().rename(columns={"index": "Actual"})
        _emit_table("Matriz de confusión", confusion_df)
        importances_df = importances.reset_index()
        importances_df.columns = ["feature", "importance"]
        _emit_table("Top features (importance)", importances_df)
        gating_summary_df = pd.DataFrame(
            [
                {
                    "allow_rule": col,
                    "count": int(gating[col].sum()),
                    "pct": float(gating[col].mean() * 100.0) if len(gating) else 0.0,
                }
                for col in gating.columns
            ]
        )
        _emit_table("Resumen gating", gating_summary_df)
        transition_df = pd.DataFrame(
            [
                {"condition": "margin>=min", "count": int(transition_rule.sum())},
                {
                    "condition": "margin+breakmag",
                    "count": int((transition_rule & transition_break).sum()),
                },
                {
                    "condition": "margin+breakmag+reentry",
                    "count": int(
                        (transition_rule & transition_break & transition_reentry).sum()
                    ),
                },
            ]
        )
        _emit_table("Detalles transición (guardrails)", transition_df)
        if "ctx_dist_vwap_atr" in ctx_features.columns and "ctx_session_bucket" in ctx_features.columns:
            dist_series = pd.to_numeric(
                ctx_features.loc[ev_frame.index, "ctx_dist_vwap_atr"], errors="coerce"
            )
            session_series = ctx_features.loc[ev_frame.index, "ctx_session_bucket"]
            bucket_rows = []
            for bucket, values in dist_series.groupby(session_series):
                values = values.dropna()
                if values.empty:
                    continue
                bucket_rows.append(
                    {
                        "bucket": bucket,
                        "count": int(values.shape[0]),
                        "dist_p50": float(values.quantile(0.5)),
                        "dist_p90": float(values.quantile(0.9)),
                    }
                )
            if bucket_rows:
                _emit_table("VWAP_BUCKET_VALIDATION", pd.DataFrame(bucket_rows))

    # Optional: JSON report
    report_payload = {
        "symbol": args.symbol,
        "start": args.start,
        "end": args.end,
        "n_samples": len(features),
        "n_train": len(features_train),
        "n_test": len(features_test),
        "split_ratio": args.split_ratio,
        "baseline": {"state": baseline_label, "pct": baseline_pct},
        "metrics": {"accuracy": acc, "f1_macro": f1},
        "confusion_matrix": {
            "labels": [label.name for label in label_order],
            "matrix": matrix.tolist(),
        },
        "class_distribution": label_dist,
        "state_hat_distribution": state_hat_dist,
        "guardrail_percentiles": {
            "quantiles": q_list,
            "BreakMag": breakmag_p,
            "ReentryCount": reentry_p,
        },
        "feature_importances": importances.to_dict(),
        "gating": {
            "thresholds": asdict(gating_policy.thresholds),
            "allow_rate": gating_allow_rate,
            "block_rate": gating_block_rate,
        },
        "model_path": str(args.model_out),
        "metadata": metadata,
        "training": {"class_weight": model_config.class_weight},
        "timings": {
            "download": elapsed_download,
            "build_dataset": elapsed_build,
            "align_and_clean": elapsed_clean,
            "split": elapsed_split,
            "train": elapsed_train,
            "evaluate": elapsed_eval,
            "predict_outputs": elapsed_outputs,
            "gating": elapsed_gating,
            "save_model": elapsed_save,
        },
        "latest_bar": {
            "last_bar_time": str(last_bar_ts),
            "server_now": str(server_now),
            "bar_age_minutes": float(bar_age_minutes),
            "tick_age_min_vs_utc": float(tick_age_min_utc),
        },
    }

    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
