"""Deterministic gating logic for PA State Engine outputs."""

from __future__ import annotations

from dataclasses import dataclass

from typing import Iterable, Sequence

import pandas as pd

from .labels import StateLabels


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

        allow_transition_failure = (state_hat == StateLabels.TRANSITION) & (margin >= th.transition_margin_min)
        if features is not None:
            required_features = {"BreakMag", "ReentryCount"}
            missing_features = required_features - set(features.columns)
            if missing_features:
                raise ValueError(f"Missing required features for gating: {sorted(missing_features)}")
            allow_transition_failure &= (
                (features["BreakMag"] >= th.transition_breakmag_min)
                & (features["ReentryCount"] >= th.transition_reentry_min)
            )

        transition_candidates = (state_hat == StateLabels.TRANSITION)
        ctx_filters_pass, ctx_counts, ctx_fail_samples = self._apply_context_filters(
            transition_candidates,
            allow_transition_failure,
            outputs,
            features,
        )
        allow_transition_failure &= ctx_filters_pass

        if logger is not None:
            config_path = "unknown"
            source = "missing"
            keys_present: list[str] = []
            if isinstance(config_meta, dict):
                config_path = config_meta.get("config_path", config_path)
                source = config_meta.get("source", source)
                keys_present = config_meta.get("keys_present", keys_present)
            logger.info(
                "GATING_CONFIG_EFFECTIVE symbol=%s allow_name=%s config_path=%s source=%s keys_present=%s "
                "allowed_sessions=%s state_age_min=%s state_age_max=%s dist_vwap_atr_min=%s dist_vwap_atr_max=%s",
                symbol,
                "ALLOW_transition_failure",
                config_path,
                source,
                keys_present,
                th.allowed_sessions,
                th.state_age_min,
                th.state_age_max,
                th.dist_vwap_atr_min,
                th.dist_vwap_atr_max,
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
        session_series = _get_col("ctx_session_bucket")
        state_age_series = _get_col("ctx_state_age")
        dist_series = _get_col("ctx_dist_vwap_atr")
        has_session = session_series is not None
        has_state_age = state_age_series is not None
        has_dist = dist_series is not None
        if not has_session:
            session_series = pd.Series([None] * len(outputs.index), index=outputs.index)
        if not has_state_age:
            state_age_series = pd.Series([None] * len(outputs.index), index=outputs.index)
        if not has_dist:
            dist_series = pd.Series([None] * len(outputs.index), index=outputs.index)

        session_mask = pd.Series(True, index=outputs.index)
        state_age_mask = pd.Series(True, index=outputs.index)
        dist_mask = pd.Series(True, index=outputs.index)

        if th.allowed_sessions is not None and has_session:
            allowed = {str(val) for val in th.allowed_sessions}
            session_mask = session_series.astype(str).isin(allowed)
        if has_state_age:
            if th.state_age_min is not None:
                state_age_mask &= state_age_series >= th.state_age_min
            if th.state_age_max is not None:
                state_age_mask &= state_age_series <= th.state_age_max
        if has_dist:
            if th.dist_vwap_atr_min is not None:
                dist_mask &= dist_series >= th.dist_vwap_atr_min
            if th.dist_vwap_atr_max is not None:
                dist_mask &= dist_series <= th.dist_vwap_atr_max

        ctx_pass = session_mask & state_age_mask & dist_mask
        after_guardrails = allow_transition_failure

        n_transition_candidates = int(transition_candidates.sum())
        n_after_guardrails = int(after_guardrails.sum())
        n_ctx_pass = int((after_guardrails & ctx_pass).sum())
        n_ctx_fail = n_after_guardrails - n_ctx_pass

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

        fail_samples = None
        if n_ctx_fail > 0:
            fail_mask = after_guardrails & ~ctx_pass
            failure_rows = pd.DataFrame(
                {
                    "index": outputs.index.astype(str),
                    "ctx_session_bucket": session_series,
                    "ctx_state_age": state_age_series,
                    "ctx_dist_vwap_atr": dist_series,
                }
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


__all__ = ["GatingThresholds", "GatingPolicy", "apply_allow_context_filters"]
