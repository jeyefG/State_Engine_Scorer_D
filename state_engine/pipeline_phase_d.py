"""Phase D context builder for canonical State Engine outputs."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import pandas as pd

from .context_features import build_context_features
from .features import FeatureEngineer
from .gating import GatingPolicy
from .model import StateEngineModel


@dataclass(frozen=True)
class ContextBundle:
    ctx_df: pd.DataFrame
    ohlcv_score: pd.DataFrame
    df_score_ctx: pd.DataFrame
    meta: dict[str, Any]


def _active_allow_filters(symbol_cfg: dict | None) -> list[str]:
    if not isinstance(symbol_cfg, dict):
        return []
    allow_cfg = symbol_cfg.get("allow_context_filters")
    if not isinstance(allow_cfg, dict):
        return []
    return sorted(
        [
            allow_name
            for allow_name, rule_cfg in allow_cfg.items()
            if isinstance(rule_cfg, dict) and rule_cfg.get("enabled", False)
        ]
    )


def validate_allow_context_requirements(
    symbol_cfg: dict | None,
    available_columns: set[str],
    *,
    logger: logging.Logger,
) -> None:
    if not isinstance(symbol_cfg, dict):
        return
    allow_cfg = symbol_cfg.get("allow_context_filters")
    if not isinstance(allow_cfg, dict):
        return

    def _has_any(candidates: list[str]) -> bool:
        return any(col in available_columns for col in candidates)

    missing_by_rule: dict[str, list[str]] = {}
    for allow_rule, rule_cfg in allow_cfg.items():
        if not isinstance(rule_cfg, dict) or not rule_cfg.get("enabled", False):
            continue
        required: list[str] = []
        if rule_cfg.get("sessions_in") is not None and not _has_any(
            ["ctx_session_bucket", "session", "session_bucket"]
        ):
            required.append("session")
        if (
            rule_cfg.get("state_age_min") is not None
            or rule_cfg.get("state_age_max") is not None
        ) and not _has_any(["ctx_state_age", "state_age"]):
            required.append("state_age")
        if (
            rule_cfg.get("dist_vwap_atr_min") is not None
            or rule_cfg.get("dist_vwap_atr_max") is not None
        ) and not _has_any(
            ["ctx_dist_vwap_atr", "dist_vwap_atr", "ctx_dist_vwap_atr_abs"]
        ):
            required.append("dist_vwap_atr")
        if (
            rule_cfg.get("breakmag_min") is not None
            or rule_cfg.get("breakmag_max") is not None
        ) and "BreakMag" not in available_columns:
            required.append("BreakMag")
        if (
            rule_cfg.get("reentry_min") is not None
            or rule_cfg.get("reentry_max") is not None
        ) and "ReentryCount" not in available_columns:
            required.append("ReentryCount")

        if required:
            missing_by_rule[allow_rule] = sorted(set(required))

    if missing_by_rule:
        missing_details = "; ".join(
            f"{allow_rule} missing={cols}"
            for allow_rule, cols in sorted(missing_by_rule.items())
        )
        raise ValueError(
            "ALLOW context filters enabled but missing columns. "
            f"{missing_details}. Available={sorted(available_columns)}"
        )
    logger.info("Phase D allow requirements OK | allows=%s", sorted(allow_cfg.keys()))


def _validate_allow_columns(
    ctx_df: pd.DataFrame,
    active_allows: list[str],
    *,
    logger: logging.Logger,
) -> dict[str, float]:
    if not active_allows:
        logger.info("Phase D context contract OK | allows=[] allow_rates={}")
        return {}
    missing = sorted(set(active_allows) - set(ctx_df.columns))
    if missing:
        raise ValueError(f"Missing ALLOW columns in Phase D context: {missing}")
    allow_rates: dict[str, float] = {}
    for allow_name in active_allows:
        series = pd.to_numeric(ctx_df[allow_name], errors="coerce")
        if series.isna().any():
            raise ValueError(f"ALLOW column {allow_name} has NaN values in Phase D context.")
        unique_vals = set(series.unique())
        if not unique_vals.issubset({0, 1}):
            raise ValueError(
                f"ALLOW column {allow_name} must be binary (0/1). Observed={sorted(unique_vals)}"
            )
        allow_rates[allow_name] = float(series.mean() * 100.0) if len(series) else 0.0
    logger.info(
        "Phase D context contract OK | allows=%s allow_rates=%s",
        active_allows,
        {k: round(v, 2) for k, v in allow_rates.items()},
    )
    return allow_rates


def _merge_context_score(
    ctx_df: pd.DataFrame,
    ohlcv_score: pd.DataFrame,
    *,
    context_tf: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    ctx = ctx_df.copy().sort_index()
    score = ohlcv_score.copy().sort_index()
    if getattr(ctx.index, "tz", None) is not None:
        ctx.index = ctx.index.tz_localize(None)
    if getattr(score.index, "tz", None) is not None:
        score.index = score.index.tz_localize(None)
    ctx = ctx.reset_index().rename(columns={ctx.index.name or "index": "time"})
    score = score.reset_index().rename(columns={score.index.name or "index": "time"})
    merged = pd.merge_asof(score, ctx, on="time", direction="backward")
    merged = merged.set_index("time")
    context_tf_norm = str(context_tf).upper()
    state_col = f"state_hat_{context_tf_norm}"
    margin_col = f"margin_{context_tf_norm}"
    merged = merged.rename(columns={"state_hat": state_col, "margin": margin_col})
    allow_cols = [col for col in merged.columns if col.startswith("ALLOW_")]
    if allow_cols:
        merged[allow_cols] = merged[allow_cols].fillna(0).astype(int)
    missing_ctx = merged[[state_col, margin_col]].isna().mean()
    if (missing_ctx > 0.25).any():
        logger.warning("High missing context after merge: %s", missing_ctx.to_dict())
    return merged


def build_context_bundle(
    *,
    symbol: str,
    context_tf: str,
    score_tf: str,
    ohlcv_ctx: pd.DataFrame,
    ohlcv_score: pd.DataFrame,
    state_model: StateEngineModel,
    feature_engineer: FeatureEngineer,
    gating_policy: GatingPolicy,
    symbol_cfg: dict | None,
    phase_e: bool,
    logger: logging.Logger,
) -> ContextBundle:
    """Build canonical Phase D context and merge into score timeframe."""
    full_features = feature_engineer.compute_features(ohlcv_ctx)
    features = feature_engineer.training_features(full_features)
    outputs = state_model.predict_outputs(features)
    ctx_features = build_context_features(
        ohlcv_ctx,
        outputs,
        symbol=symbol,
        timeframe=context_tf,
    )
    features_for_gating = full_features.join(ctx_features, how="left").reindex(outputs.index)
    available_columns = set(features_for_gating.columns) | set(outputs.columns)
    validate_allow_context_requirements(
        symbol_cfg,
        available_columns,
        logger=logger,
    )
    allows = gating_policy.apply(
        outputs,
        features=features_for_gating,
        logger=logger,
        symbol=symbol,
        config_meta=symbol_cfg,
    )
    ctx_cols = [col for col in outputs.columns if col.startswith("ctx_")]
    ctx_df = pd.concat([outputs[["state_hat", "margin", *ctx_cols]], ctx_features, allows], axis=1)
    ctx_df = ctx_df.shift(1)
    allow_cols = [col for col in ctx_df.columns if col.startswith("ALLOW_")]
    if allow_cols:
        ctx_df[allow_cols] = ctx_df[allow_cols].fillna(0).astype(int)

    active_allows = _active_allow_filters(symbol_cfg)
    allow_rates = _validate_allow_columns(ctx_df, active_allows, logger=logger)
    if phase_e:
        logger.info("Phase D context bundle ready for Phase E consumer.")

    df_score_ctx = _merge_context_score(ctx_df, ohlcv_score, context_tf=context_tf, logger=logger)
    meta = {
        "symbol": symbol,
        "context_tf": context_tf,
        "score_tf": score_tf,
        "active_allows": active_allows,
        "allow_rates": allow_rates,
    }
    return ContextBundle(
        ctx_df=ctx_df,
        ohlcv_score=ohlcv_score,
        df_score_ctx=df_score_ctx,
        meta=meta,
    )


__all__ = [
    "ContextBundle",
    "build_context_bundle",
    "validate_allow_context_requirements",
]
