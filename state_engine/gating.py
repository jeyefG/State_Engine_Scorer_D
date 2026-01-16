"""Deterministic gating logic for PA State Engine outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging

from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from .labels import StateLabels


_CTX_DIAG_LOGGED = False


@dataclass(frozen=True)
class GatingThresholds:
    trend_margin_min: float = 0.15
    balance_margin_min: float = 0.10
    transition_margin_min: float = 0.10
    transition_breakmag_min: float = 0.25
    transition_reentry_min: float = 1.0
    allowed_sessions: Iterable[str] | None = None
    state_age_min: int | None = None
    state_age_max: int | None = None
    dist_vwap_atr_min: float | None = None
    dist_vwap_atr_max: float | None = None
    transition_ctx_enabled: bool = True
    transition_ctx_require_all: bool = True


def get_transition_ctx_cfg(
    cfg: dict[str, Any] | None,
    allow_name: str,
) -> tuple[dict[str, Any], str, list[str], list[str]]:
    tried_paths = [
        f"allow_context_filters.{allow_name}",
        f"gating.allow_context_filters.{allow_name}",
        f"gating.allows.{allow_name}",
    ]
    if not isinstance(cfg, dict):
        return {}, "missing", [], tried_paths

    candidates = [
        cfg.get("allow_context_filters", {}).get(allow_name),
        cfg.get("gating", {}).get("allow_context_filters", {}).get(allow_name),
        cfg.get("gating", {}).get("allows", {}).get(allow_name),
    ]
    selected_path = "missing"
    rule_cfg: dict[str, Any] = {}
    for path, candidate in zip(tried_paths, candidates):
        if isinstance(candidate, dict):
            rule_cfg = dict(candidate)
            selected_path = path
            break
    keys_present = sorted(rule_cfg.keys()) if rule_cfg else []
    return rule_cfg, selected_path, keys_present, tried_paths


def build_transition_gating_thresholds(
    symbol: str | None,
    symbol_config: dict[str, Any] | None,
    *,
    allow_name: str = "ALLOW_transition_failure",
    logger=None,
) -> tuple[GatingThresholds, dict[str, Any]]:
    if logger is not None:
        top_keys = list(symbol_config.keys()) if isinstance(symbol_config, dict) else []
        allow_keys = []
        if isinstance(symbol_config, dict):
            allow_ctx = symbol_config.get("allow_context_filters", {})
            if isinstance(allow_ctx, dict):
                allow_keys = list(allow_ctx.keys())
        logger.info("GATING_CFG_TOP_KEYS symbol=%s top_keys=%s", symbol, top_keys)
        logger.info(
            "GATING_CFG_ALLOWCTX_KEYS symbol=%s allow_context_filters_keys=%s",
            symbol,
            allow_keys,
        )

    rule_cfg, selected_path, keys_present, tried_paths = get_transition_ctx_cfg(
        symbol_config,
        allow_name,
    )
    if logger is not None:
        logger.info(
            "GATING_CFG_LOOKUP symbol=%s allow_name=%s tried_paths=%s selected_path=%s keys_present=%s",
            symbol,
            allow_name,
            tried_paths,
            selected_path,
            keys_present,
        )

    enabled = bool(rule_cfg.get("enabled", False)) if rule_cfg else False
    require_all = bool(rule_cfg.get("require_all", True)) if rule_cfg else True

    active_cfg = rule_cfg if enabled else {}
    overrides: dict[str, Any] = {
        "transition_ctx_enabled": enabled,
        "transition_ctx_require_all": require_all,
    }
    if active_cfg.get("sessions_in") is not None:
        overrides["allowed_sessions"] = active_cfg.get("sessions_in")
    if active_cfg.get("state_age_min") is not None:
        overrides["state_age_min"] = int(active_cfg["state_age_min"])
    if active_cfg.get("state_age_max") is not None:
        overrides["state_age_max"] = int(active_cfg["state_age_max"])
    if active_cfg.get("dist_vwap_atr_min") is not None:
        overrides["dist_vwap_atr_min"] = float(active_cfg["dist_vwap_atr_min"])
    if active_cfg.get("dist_vwap_atr_max") is not None:
        overrides["dist_vwap_atr_max"] = float(active_cfg["dist_vwap_atr_max"])
    if active_cfg.get("breakmag_min") is not None:
        overrides["transition_breakmag_min"] = float(active_cfg["breakmag_min"])
    if active_cfg.get("reentry_min") is not None:
        overrides["transition_reentry_min"] = float(active_cfg["reentry_min"])

    thresholds = GatingThresholds(**{**asdict(GatingThresholds()), **overrides})
    config_path = Path("configs") / "symbols" / f"{symbol}.yaml" if symbol else None
    return thresholds, {
        "keys_present": keys_present,
        "source": "symbol" if rule_cfg else "missing",
        "config_path": str(config_path) if config_path and config_path.exists() else "unknown",
        "selected_path": selected_path,
        "tried_paths": tried_paths,
        "enabled": enabled,
        "require_all": require_all,
        "sessions_in": rule_cfg.get("sessions_in") if rule_cfg else None,
        "state_age_max": rule_cfg.get("state_age_max") if rule_cfg else None,
        "dist_vwap_atr_max": rule_cfg.get("dist_vwap_atr_max") if rule_cfg else None,
        "breakmag_min": rule_cfg.get("breakmag_min") if rule_cfg else None,
        "reentry_min": rule_cfg.get("reentry_min") if rule_cfg else None,
    }


class GatingPolicy:
    """Apply ALLOW_* rules based on StateEngine state and margin."""

    def __init__(self, thresholds: GatingThresholds | None = None) -> None:
        self.thresholds = thresholds or GatingThresholds()

    def apply(
        self,
        outputs: pd.DataFrame,
        features: pd.DataFrame | None = None,
        *,
        logger=None,
        symbol: str | None = None,
        config_meta: dict | None = None,
    ) -> pd.DataFrame:
        """Return DataFrame with ALLOW_* columns."""
        required = {"state_hat", "margin"}
        missing = required - set(outputs.columns)
        if missing:
            raise ValueError(f"Missing required output columns: {sorted(missing)}")

        th = self.thresholds
        state_hat = outputs["state_hat"]
        margin = outputs["margin"]
        allow_trend_pullback = (state_hat == StateLabels.TREND) & (margin >= th.trend_margin_min)
        allow_trend_continuation = (state_hat == StateLabels.TREND) & (margin >= th.trend_margin_min)
        allow_balance_fade = (state_hat == StateLabels.BALANCE) & (margin >= th.balance_margin_min)

        transition_candidates = (state_hat == StateLabels.TRANSITION)
        guardrails_breakmag_min = th.transition_breakmag_min
        guardrails_reentry_min = th.transition_reentry_min
        if isinstance(config_meta, dict):
            if config_meta.get("breakmag_min") is not None:
                guardrails_breakmag_min = float(config_meta["breakmag_min"])
            if config_meta.get("reentry_min") is not None:
                guardrails_reentry_min = float(config_meta["reentry_min"])
        guardrails_ok = transition_candidates & (margin >= th.transition_margin_min)
        if features is not None:
            required_features = {"BreakMag", "ReentryCount"}
            missing_features = required_features - set(features.columns)
            if missing_features:
                raise ValueError(f"Missing required features for gating: {sorted(missing_features)}")
            guardrails_ok &= (
                (features["BreakMag"] >= guardrails_breakmag_min)
                & (features["ReentryCount"] >= guardrails_reentry_min)
            )

        ctx_filters_pass, ctx_counts, ctx_fail_samples = self._apply_context_filters(
            transition_candidates,
            guardrails_ok,
            outputs,
            features,
        )
        allow_transition_failure = guardrails_ok & ctx_filters_pass

        if logger is not None:
            config_path = "unknown"
            source = "missing"
            selected_path = "missing"
            keys_present: list[str] = []
            enabled = th.transition_ctx_enabled
            require_all = th.transition_ctx_require_all
            sessions_in = th.allowed_sessions
            state_age_max = th.state_age_max
            dist_vwap_atr_max = th.dist_vwap_atr_max
            breakmag_min = th.transition_breakmag_min
            reentry_min = th.transition_reentry_min
            if isinstance(config_meta, dict):
                config_path = config_meta.get("config_path", config_path)
                source = config_meta.get("source", source)
                selected_path = config_meta.get("selected_path", selected_path)
                keys_present = config_meta.get("keys_present", keys_present)
                enabled = config_meta.get("enabled", enabled)
                require_all = config_meta.get("require_all", require_all)
                sessions_in = config_meta.get("sessions_in", sessions_in)
                state_age_max = config_meta.get("state_age_max", state_age_max)
                dist_vwap_atr_max = config_meta.get("dist_vwap_atr_max", dist_vwap_atr_max)
                breakmag_min = config_meta.get("breakmag_min", breakmag_min)
                reentry_min = config_meta.get("reentry_min", reentry_min)
            logger.info(
                "GATING_CONFIG_EFFECTIVE symbol=%s allow_name=%s config_path=%s source=%s selected_path=%s "
                "keys_present=%s enabled=%s require_all=%s sessions_in=%s state_age_max=%s dist_vwap_atr_max=%s "
                "breakmag_min=%s reentry_min=%s",
                symbol,
                "ALLOW_transition_failure",
                config_path,
                source,
                selected_path,
                keys_present,
                enabled,
                require_all,
                sessions_in,
                state_age_max,
                dist_vwap_atr_max,
                breakmag_min,
                reentry_min,
            )
            if not th.transition_ctx_enabled:
                logger.info(
                    "TRANSITION_CTX_FILTER_DISABLED symbol=%s allow_name=%s",
                    symbol,
                    "ALLOW_transition_failure",
                )
            logger.info(
                "TRANSITION_CTX_FILTER_COUNTS symbol=%s n_transition_candidates=%s n_after_guardrails=%s "
                "n_ctx_pass=%s n_ctx_fail=%s pass_session=%s pass_state_age=%s pass_dist=%s",
                symbol,
                ctx_counts.get("n_transition_candidates"),
                ctx_counts.get("n_after_guardrails"),
                ctx_counts.get("n_ctx_pass"),
                ctx_counts.get("n_ctx_fail"),
                ctx_counts.get("pass_session"),
                ctx_counts.get("pass_state_age"),
                ctx_counts.get("pass_dist"),
            )
            if ctx_fail_samples is not None and not ctx_fail_samples.empty:
                logger.info(
                    "TRANSITION_CTX_FILTER_FAIL_SAMPLES\n%s",
                    ctx_fail_samples.to_string(index=False),
                )

        return pd.DataFrame(
            {
                "ALLOW_trend_pullback": allow_trend_pullback.astype(int),
                "ALLOW_trend_continuation": allow_trend_continuation.astype(int),
                "ALLOW_balance_fade": allow_balance_fade.astype(int),
                "ALLOW_transition_failure": allow_transition_failure.astype(int),
            },
            index=outputs.index,
        )

    def _apply_context_filters(
        self,
        transition_candidates: pd.Series,
        allow_transition_failure: pd.Series,
        outputs: pd.DataFrame,
        features: pd.DataFrame | None,
    ) -> tuple[pd.Series, dict[str, int], pd.DataFrame | None]:
        def _get_col(column: str) -> pd.Series | None:
            if features is not None and column in features.columns:
                return features[column]
            if column in outputs.columns:
                return outputs[column]
            return None

        th = self.thresholds
        idx = outputs.index
        transition_candidates = transition_candidates.reindex(idx).fillna(False)
        allow_transition_failure = allow_transition_failure.reindex(idx).fillna(False)
        session_series = _get_col("ctx_session_bucket")
        state_age_series = _get_col("ctx_state_age")
        dist_series = _get_col("ctx_dist_vwap_atr")
        has_session = session_series is not None
        has_state_age = state_age_series is not None
        has_dist = dist_series is not None
        if not has_session:
            session_series = pd.Series([None] * len(idx), index=idx)
        else:
            session_series = session_series.reindex(idx)
        if not has_state_age:
            state_age_series = pd.Series([None] * len(idx), index=idx)
        else:
            state_age_series = state_age_series.reindex(idx)
        if not has_dist:
            dist_series = pd.Series([None] * len(idx), index=idx)
        else:
            dist_series = dist_series.reindex(idx)

        session_mask = pd.Series(True, index=idx)
        state_age_mask = pd.Series(True, index=idx)
        dist_mask = pd.Series(True, index=idx)
        active_masks: list[pd.Series] = []

        if th.allowed_sessions is not None and has_session:
            allowed = {str(val) for val in th.allowed_sessions}
            session_mask = session_series.astype(str).isin(allowed)
            active_masks.append(session_mask)
        if has_state_age:
            applied_state_age = False
            if th.state_age_min is not None:
                state_age_mask &= state_age_series >= th.state_age_min
                applied_state_age = True
            if th.state_age_max is not None:
                state_age_mask &= state_age_series <= th.state_age_max
                applied_state_age = True
            if applied_state_age:
                active_masks.append(state_age_mask)
        if has_dist:
            applied_dist = False
            if th.dist_vwap_atr_min is not None:
                dist_mask &= dist_series >= th.dist_vwap_atr_min
                applied_dist = True
            if th.dist_vwap_atr_max is not None:
                dist_mask &= dist_series <= th.dist_vwap_atr_max
                applied_dist = True
            if applied_dist:
                active_masks.append(dist_mask)

        if not th.transition_ctx_enabled:
            ctx_pass = pd.Series(True, index=outputs.index)
        elif not active_masks:
            ctx_pass = pd.Series(True, index=outputs.index)
        else:
            if th.transition_ctx_require_all:
                ctx_pass = active_masks[0]
                for mask in active_masks[1:]:
                    ctx_pass = ctx_pass & mask
            else:
                ctx_pass = active_masks[0]
                for mask in active_masks[1:]:
                    ctx_pass = ctx_pass | mask
        after_guardrails = allow_transition_failure.reindex(idx).fillna(False)
        ctx_pass = ctx_pass.reindex(idx).fillna(False)
        session_mask = session_mask.reindex(idx).fillna(False)
        state_age_mask = state_age_mask.reindex(idx).fillna(False)
        dist_mask = dist_mask.reindex(idx).fillna(False)

        n_transition_candidates = int(transition_candidates.sum())
        n_after_guardrails = int(after_guardrails.sum())
        n_ctx_pass = int((after_guardrails & ctx_pass).sum())
        n_ctx_fail = n_after_guardrails - n_ctx_pass

        fail_mask = after_guardrails & ~ctx_pass

        pass_session = int((after_guardrails & session_mask).sum())
        pass_state_age = int((after_guardrails & state_age_mask).sum())
        pass_dist = int((after_guardrails & dist_mask).sum())

        counts = {
            "n_transition_candidates": n_transition_candidates,
            "n_after_guardrails": n_after_guardrails,
            "n_ctx_pass": n_ctx_pass,
            "n_ctx_fail": n_ctx_fail,
            "pass_session": pass_session,
            "pass_state_age": pass_state_age,
            "pass_dist": pass_dist,
        }

        logger = logging.getLogger(__name__)
        global _CTX_DIAG_LOGGED
        if not _CTX_DIAG_LOGGED:
            features_len = len(features.index) if features is not None else None
            idx_equal = outputs.index.equals(features.index) if features is not None else None
            logger.info(
                "CTX_FILTER_DIAG OUTPUTS_LEN=%s FEATURES_LEN=%s IDX_EQUAL=%s "
                "MASK_LENS after_guardrails=%s ctx_pass=%s fail_mask=%s",
                len(outputs.index),
                features_len,
                idx_equal,
                len(after_guardrails),
                len(ctx_pass),
                len(fail_mask),
            )
            _CTX_DIAG_LOGGED = True

        fail_samples = None
        if n_ctx_fail > 0:
            failure_rows = pd.DataFrame(
                {
                    "index": idx.astype(str),
                    "ctx_session_bucket": session_series,
                    "ctx_state_age": state_age_series,
                    "ctx_dist_vwap_atr": dist_series,
                },
                index=idx,
            ).loc[fail_mask]
            failure_rows = failure_rows.head(5).copy()
            reasons: list[str] = []
            for idx in failure_rows.index:
                parts: list[str] = []
                if not bool(session_mask.loc[idx]):
                    parts.append("session")
                if not bool(state_age_mask.loc[idx]):
                    parts.append("state_age")
                if not bool(dist_mask.loc[idx]):
                    parts.append("dist_vwap_atr")
                reasons.append("|".join(parts) if parts else "unknown")
            failure_rows.insert(1, "reason", reasons)
            fail_samples = failure_rows

        return ctx_pass, counts, fail_samples


def apply_allow_context_filters(
    gating_df: pd.DataFrame,
    symbol_cfg: dict | None,
    logger,
) -> pd.DataFrame:
    """Apply config-driven context filters to ALLOW_* rules."""
    if not symbol_cfg or not isinstance(symbol_cfg, dict):
        return gating_df
    allow_cfg = symbol_cfg.get("allow_context_filters")
    if not allow_cfg or not isinstance(allow_cfg, dict):
        return gating_df

    def _resolve_column(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
        for name in candidates:
            if name in df.columns:
                return name
        return None

    filtered = gating_df.copy()
    for allow_rule, rule_cfg in allow_cfg.items():
        if not isinstance(rule_cfg, dict):
            logger.warning("allow_context_filters.%s must be a mapping; skipping.", allow_rule)
            continue
        if not rule_cfg.get("enabled", False):
            continue
        if allow_rule not in filtered.columns:
            logger.warning("allow_context_filters.%s missing in gating_df; skipping.", allow_rule)
            continue

        allow_series = filtered[allow_rule].astype(bool)
        before_rate = float(allow_series.mean()) if len(allow_series) else 0.0
        require_all = rule_cfg.get("require_all", True)

        masks: list[pd.Series] = []
        applied_conditions: list[str] = []

        sessions_in = rule_cfg.get("sessions_in")
        if sessions_in is not None:
            col_name = _resolve_column(filtered, ["session", "session_bucket"])
            if col_name is None:
                logger.warning(
                    "allow_context_filters.%s missing session column; skipping sessions_in.",
                    allow_rule,
                )
            else:
                allowed = {str(val) for val in sessions_in}
                mask = filtered[col_name].astype(str).isin(allowed)
                masks.append(mask)
                applied_conditions.append(f"sessions_in={sorted(allowed)} via {col_name}")

        state_age_min = rule_cfg.get("state_age_min")
        if state_age_min is not None:
            col_name = _resolve_column(filtered, ["state_age"])
            if col_name is None:
                logger.warning(
                    "allow_context_filters.%s missing state_age column; skipping state_age_min.",
                    allow_rule,
                )
            else:
                series = pd.to_numeric(filtered[col_name], errors="coerce")
                mask = series >= float(state_age_min)
                masks.append(mask)
                applied_conditions.append(f"state_age_min>={state_age_min} via {col_name}")

        state_age_max = rule_cfg.get("state_age_max")
        if state_age_max is not None:
            col_name = _resolve_column(filtered, ["state_age"])
            if col_name is None:
                logger.warning(
                    "allow_context_filters.%s missing state_age column; skipping state_age_max.",
                    allow_rule,
                )
            else:
                series = pd.to_numeric(filtered[col_name], errors="coerce")
                mask = series <= float(state_age_max)
                masks.append(mask)
                applied_conditions.append(f"state_age_max<={state_age_max} via {col_name}")

        dist_vwap_atr_min = rule_cfg.get("dist_vwap_atr_min")
        if dist_vwap_atr_min is not None:
            col_name = _resolve_column(filtered, ["dist_vwap_atr"])
            if col_name is None:
                logger.warning(
                    "allow_context_filters.%s missing dist_vwap_atr column; skipping dist_vwap_atr_min.",
                    allow_rule,
                )
            else:
                series = pd.to_numeric(filtered[col_name], errors="coerce")
                mask = series >= float(dist_vwap_atr_min)
                masks.append(mask)
                applied_conditions.append(f"dist_vwap_atr_min>={dist_vwap_atr_min} via {col_name}")

        dist_vwap_atr_max = rule_cfg.get("dist_vwap_atr_max")
        if dist_vwap_atr_max is not None:
            col_name = _resolve_column(filtered, ["dist_vwap_atr"])
            if col_name is None:
                logger.warning(
                    "allow_context_filters.%s missing dist_vwap_atr column; skipping dist_vwap_atr_max.",
                    allow_rule,
                )
            else:
                series = pd.to_numeric(filtered[col_name], errors="coerce")
                mask = series <= float(dist_vwap_atr_max)
                masks.append(mask)
                applied_conditions.append(f"dist_vwap_atr_max<={dist_vwap_atr_max} via {col_name}")

        breakmag_min = rule_cfg.get("breakmag_min")
        if breakmag_min is not None:
            col_name = _resolve_column(filtered, ["BreakMag"])
            if col_name is None:
                logger.warning(
                    "allow_context_filters.%s missing BreakMag column; skipping breakmag_min.",
                    allow_rule,
                )
            else:
                series = pd.to_numeric(filtered[col_name], errors="coerce")
                mask = series >= float(breakmag_min)
                masks.append(mask)
                applied_conditions.append(f"breakmag_min>={breakmag_min} via {col_name}")

        breakmag_max = rule_cfg.get("breakmag_max")
        if breakmag_max is not None:
            col_name = _resolve_column(filtered, ["BreakMag"])
            if col_name is None:
                logger.warning(
                    "allow_context_filters.%s missing BreakMag column; skipping breakmag_max.",
                    allow_rule,
                )
            else:
                series = pd.to_numeric(filtered[col_name], errors="coerce")
                mask = series <= float(breakmag_max)
                masks.append(mask)
                applied_conditions.append(f"breakmag_max<={breakmag_max} via {col_name}")

        reentry_min = rule_cfg.get("reentry_min")
        if reentry_min is not None:
            col_name = _resolve_column(filtered, ["ReentryCount"])
            if col_name is None:
                logger.warning(
                    "allow_context_filters.%s missing ReentryCount column; skipping reentry_min.",
                    allow_rule,
                )
            else:
                series = pd.to_numeric(filtered[col_name], errors="coerce")
                mask = series >= float(reentry_min)
                masks.append(mask)
                applied_conditions.append(f"reentry_min>={reentry_min} via {col_name}")

        reentry_max = rule_cfg.get("reentry_max")
        if reentry_max is not None:
            col_name = _resolve_column(filtered, ["ReentryCount"])
            if col_name is None:
                logger.warning(
                    "allow_context_filters.%s missing ReentryCount column; skipping reentry_max.",
                    allow_rule,
                )
            else:
                series = pd.to_numeric(filtered[col_name], errors="coerce")
                mask = series <= float(reentry_max)
                masks.append(mask)
                applied_conditions.append(f"reentry_max<={reentry_max} via {col_name}")

        if not masks:
            logger.info("allow_context_filters.%s no effective conditions; skipping.", allow_rule)
            continue

        if require_all:
            combined_mask = masks[0]
            for mask in masks[1:]:
                combined_mask = combined_mask & mask
        else:
            combined_mask = masks[0]
            for mask in masks[1:]:
                combined_mask = combined_mask | mask

        filtered[allow_rule] = (allow_series & combined_mask.fillna(False)).astype(int)
        after_rate = float(filtered[allow_rule].mean()) if len(filtered) else 0.0
        delta = after_rate - before_rate
        logger.info(
            "%s: before=%.2f%% after=%.2f%% delta=%.2f%%",
            allow_rule,
            before_rate * 100.0,
            after_rate * 100.0,
            delta * 100.0,
        )
        logger.info("%s conditions=%s", allow_rule, "; ".join(applied_conditions))

    return filtered


__all__ = [
    "GatingThresholds",
    "GatingPolicy",
    "apply_allow_context_filters",
    "build_transition_gating_thresholds",
]
