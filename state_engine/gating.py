"""Deterministic gating logic for PA State Engine outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from pathlib import Path
from typing import Any, Iterable, Sequence

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
        # --- HARD ALIGN: features must follow outputs index (outputs is post-clean) ---
        if features is not None and not features.index.equals(outputs.index):
            if logger is not None:
                logger.info(
                    "GATING_ALIGN symbol=%s outputs_len=%s features_len_before=%s idx_equal_before=%s",
                    symbol, len(outputs), len(features), features.index.equals(outputs.index)
                )
            features = features.reindex(outputs.index)
            if logger is not None:
                logger.info(
                    "GATING_ALIGN_DONE symbol=%s outputs_len=%s features_len_after=%s idx_equal_after=%s",
                    symbol,
                    len(outputs.index),
                    len(features.index),
                    features.index.equals(outputs.index),
                )
        # ---------------------------------------------------------------------------
        allow_trend_pullback = (state_hat == StateLabels.TREND) & (margin >= th.trend_margin_min)
        allow_trend_continuation = (state_hat == StateLabels.TREND) & (margin >= th.trend_margin_min)
        allow_balance_fade = (state_hat == StateLabels.BALANCE) & (margin >= th.balance_margin_min)

        transition_candidates = (state_hat == StateLabels.TRANSITION)
        guardrails_breakmag_min = th.transition_breakmag_min
        guardrails_reentry_min = th.transition_reentry_min
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
        allow_transition_failure = guardrails_ok

        # Fase D: context filters are semantic filters, not signals.
        base_allow_names = [
            "ALLOW_trend_pullback",
            "ALLOW_trend_continuation",
            "ALLOW_balance_fade",
            "ALLOW_transition_failure",
        ]
        allow_map = {
            "ALLOW_trend_pullback": allow_trend_pullback,
            "ALLOW_trend_continuation": allow_trend_continuation,
            "ALLOW_balance_fade": allow_balance_fade,
            "ALLOW_transition_failure": allow_transition_failure,
        }

        ctx_allows = {}
        if isinstance(config_meta, dict):
            ctx_allows = config_meta.get("allow_context_filters", {})
            if not isinstance(ctx_allows, dict):
                ctx_allows = {}

        extra_allow_names = sorted(
            [name for name in ctx_allows.keys() if name not in allow_map],
        )
        enabled_extras = [
            name
            for name in extra_allow_names
            if isinstance(ctx_allows.get(name), dict) and ctx_allows[name].get("enabled", False)
        ]
        if logger is not None:
            logger.info(
                "GATING_ALLOWS_BUILT base=%s extras=%s enabled_extras=%s",
                base_allow_names,
                extra_allow_names,
                enabled_extras,
            )

        for allow_name in extra_allow_names:
            lower_name = allow_name.lower()
            if "trend" in lower_name:
                allow_map[allow_name] = allow_trend_pullback
            elif "transition" in lower_name:
                allow_map[allow_name] = guardrails_ok
            elif "balance" in lower_name:
                allow_map[allow_name] = allow_balance_fade
            else:
                if logger is not None:
                    logger.warning(
                        "GATING_ALLOW_UNKNOWN_STATE symbol=%s allow_name=%s base_mask=false",
                        symbol,
                        allow_name,
                    )
                allow_map[allow_name] = pd.Series(False, index=outputs.index)

        allow_names = base_allow_names + extra_allow_names
        for allow_name in allow_names:
            ctx_meta = None
            if isinstance(config_meta, dict):
                ctx_all = config_meta.get("allow_context_filters")
                if isinstance(ctx_all, dict):
                    ctx_meta = ctx_all.get(allow_name)
            filtered_mask, _, _ = self._apply_allow_context_filter(
                allow_name,
                allow_map[allow_name],
                outputs,
                features,
                ctx_meta,
                logger,
                symbol,
            )
            allow_map[allow_name] = filtered_mask
        allow_payload = {allow_name: allow_map[allow_name].astype(int) for allow_name in allow_names}
        return pd.DataFrame(allow_payload, index=outputs.index)

    def _apply_allow_context_filter(
        self,
        allow_name: str,
        base_mask: pd.Series,
        outputs: pd.DataFrame,
        features: pd.DataFrame | None,
        ctx_meta: dict[str, Any] | None,
        logger,
        symbol: str | None,
    ) -> tuple[pd.Series, dict[str, Any], pd.DataFrame | None]:
        def _get_col(column: str) -> pd.Series | None:
            if features is not None and column in features.columns:
                return features[column]
            if column in outputs.columns:
                return outputs[column]
            return None

        idx = outputs.index
        base_mask = base_mask.reindex(idx).fillna(False).astype(bool)
        if features is not None and not features.index.equals(idx):
            features = features.reindex(idx)

        ctx_meta = ctx_meta if isinstance(ctx_meta, dict) else None
        enabled = bool(ctx_meta.get("enabled", False)) if ctx_meta else False
        require_all = bool(ctx_meta.get("require_all", True)) if ctx_meta else True

        session_series = _get_col("ctx_session_bucket")
        state_age_series = _get_col("ctx_state_age")
        dist_series = _get_col("ctx_dist_vwap_atr")
        breakmag_series = _get_col("BreakMag")
        reentry_series = _get_col("ReentryCount")

        masks: list[pd.Series] = []
        mask_map: dict[str, pd.Series] = {}
        applied: set[str] = set()

        sessions_in = ctx_meta.get("sessions_in") if ctx_meta else None
        if sessions_in is not None and session_series is not None:
            allowed = {str(val) for val in sessions_in}
            mask = session_series.astype(str).isin(allowed)
            masks.append(mask)
            mask_map["session"] = mask
            applied.add("session")

        state_age_min = ctx_meta.get("state_age_min") if ctx_meta else None
        state_age_max = ctx_meta.get("state_age_max") if ctx_meta else None
        if (state_age_min is not None or state_age_max is not None) and state_age_series is not None:
            series = pd.to_numeric(state_age_series, errors="coerce")
            mask = pd.Series(True, index=idx)
            if state_age_min is not None:
                mask &= series >= float(state_age_min)
            if state_age_max is not None:
                mask &= series <= float(state_age_max)
            masks.append(mask)
            mask_map["state_age"] = mask
            applied.add("state_age")

        dist_min = ctx_meta.get("dist_vwap_atr_min") if ctx_meta else None
        dist_max = ctx_meta.get("dist_vwap_atr_max") if ctx_meta else None
        if (dist_min is not None or dist_max is not None) and dist_series is not None:
            series = pd.to_numeric(dist_series, errors="coerce")
            mask = pd.Series(True, index=idx)
            if dist_min is not None:
                mask &= series >= float(dist_min)
            if dist_max is not None:
                mask &= series <= float(dist_max)
            masks.append(mask)
            mask_map["dist_vwap_atr"] = mask
            applied.add("dist_vwap_atr")

        breakmag_min = ctx_meta.get("breakmag_min") if ctx_meta else None
        breakmag_max = ctx_meta.get("breakmag_max") if ctx_meta else None
        if (breakmag_min is not None or breakmag_max is not None) and breakmag_series is not None:
            series = pd.to_numeric(breakmag_series, errors="coerce")
            mask = pd.Series(True, index=idx)
            if breakmag_min is not None:
                mask &= series >= float(breakmag_min)
            if breakmag_max is not None:
                mask &= series <= float(breakmag_max)
            masks.append(mask)
            mask_map["breakmag"] = mask
            applied.add("breakmag")

        reentry_min = ctx_meta.get("reentry_min") if ctx_meta else None
        reentry_max = ctx_meta.get("reentry_max") if ctx_meta else None
        if (reentry_min is not None or reentry_max is not None) and reentry_series is not None:
            series = pd.to_numeric(reentry_series, errors="coerce")
            mask = pd.Series(True, index=idx)
            if reentry_min is not None:
                mask &= series >= float(reentry_min)
            if reentry_max is not None:
                mask &= series <= float(reentry_max)
            masks.append(mask)
            mask_map["reentry"] = mask
            applied.add("reentry")

        if not enabled:
            counts = {
                "total_rows": int(len(idx)),
                "n_base": int(base_mask.sum()),
                "n_pass": int(base_mask.sum()),
                "n_fail": 0,
                "status": "disabled",
            }
            return base_mask, counts, None

        if logger is not None:
            logger.info(
                "GATING_CONFIG_EFFECTIVE symbol=%s allow_name=%s config_path=%s source=%s selected_path=%s "
                "keys_present=%s enabled=%s require_all=%s sessions_in=%s state_age_min=%s state_age_max=%s "
                "dist_vwap_atr_min=%s dist_vwap_atr_max=%s breakmag_min=%s breakmag_max=%s reentry_min=%s "
                "reentry_max=%s",
                symbol,
                allow_name,
                ctx_meta.get("config_path") if ctx_meta else None,
                ctx_meta.get("source") if ctx_meta else None,
                ctx_meta.get("selected_path") if ctx_meta else None,
                ctx_meta.get("keys_present") if ctx_meta else None,
                enabled,
                require_all,
                sessions_in,
                state_age_min,
                state_age_max,
                dist_min,
                dist_max,
                breakmag_min,
                breakmag_max,
                reentry_min,
                reentry_max,
            )

        if not masks:
            if logger is not None:
                logger.info(
                    "NO_EFFECTIVE_CONDITIONS symbol=%s allow_name=%s",
                    symbol,
                    allow_name,
                )
            counts = {
                "total_rows": int(len(idx)),
                "n_base": int(base_mask.sum()),
                "n_pass": int(base_mask.sum()),
                "n_fail": 0,
                "status": "no_effective_conditions",
            }
            return base_mask, counts, None

        if require_all:
            combined_mask = masks[0]
            for mask in masks[1:]:
                combined_mask = combined_mask & mask
        else:
            combined_mask = masks[0]
            for mask in masks[1:]:
                combined_mask = combined_mask | mask
        combined_mask = combined_mask.reindex(idx).fillna(False)

        filtered_mask = base_mask & combined_mask
        total_rows = int(len(idx))
        n_base = int(base_mask.sum())
        n_pass = int(filtered_mask.sum())
        n_fail = n_base - n_pass

        def _pass_count(key: str) -> int | None:
            if key not in applied:
                return None
            return int((base_mask & mask_map[key].reindex(idx).fillna(False)).sum())

        counts = {
            "total_rows": total_rows,
            "n_base": n_base,
            "n_pass": n_pass,
            "n_fail": n_fail,
            "pass_session": _pass_count("session"),
            "pass_state_age": _pass_count("state_age"),
            "pass_dist": _pass_count("dist_vwap_atr"),
            "pass_breakmag": _pass_count("breakmag"),
            "pass_reentry": _pass_count("reentry"),
        }

        if logger is not None:
            logger.info(
                "ALLOW_CTX_FILTER_COUNTS symbol=%s allow_name=%s n_base=%s n_pass=%s n_fail=%s "
                "pass_session=%s pass_state_age=%s pass_dist=%s pass_breakmag=%s pass_reentry=%s",
                symbol,
                allow_name,
                counts["n_base"],
                counts["n_pass"],
                counts["n_fail"],
                counts.get("pass_session"),
                counts.get("pass_state_age"),
                counts.get("pass_dist"),
                counts.get("pass_breakmag"),
                counts.get("pass_reentry"),
            )
            total_rows = max(total_rows, 1)
            before_pct = 100.0 * n_base / total_rows
            after_pct = 100.0 * n_pass / total_rows
            logger.info(
                "ALLOW_RATE symbol=%s allow_name=%s total=%s before=%.2f%% after=%.2f%% delta=%.2f%%",
                symbol,
                allow_name,
                total_rows,
                before_pct,
                after_pct,
                after_pct - before_pct,
            )

        fail_samples = None
        if n_fail > 0:
            fail_mask = base_mask & ~combined_mask
            failure_rows = pd.DataFrame(index=idx)
            failure_rows["index"] = idx.astype(str)
            failure_rows["ctx_session_bucket"] = (
                session_series.reindex(idx).values if session_series is not None else [None] * len(idx)
            )
            failure_rows["ctx_state_age"] = (
                state_age_series.reindex(idx).values if state_age_series is not None else [None] * len(idx)
            )
            failure_rows["ctx_dist_vwap_atr"] = (
                dist_series.reindex(idx).values if dist_series is not None else [None] * len(idx)
            )
            failure_rows = failure_rows.loc[fail_mask]
            failure_rows = failure_rows.head(5).copy()
            reasons: list[str] = []
            for row_idx in failure_rows.index:
                parts: list[str] = []
                for key, label in [
                    ("session", "session"),
                    ("state_age", "state_age"),
                    ("dist_vwap_atr", "dist_vwap_atr"),
                    ("breakmag", "breakmag"),
                    ("reentry", "reentry"),
                ]:
                    if key in applied and not bool(mask_map[key].reindex(idx).fillna(False).loc[row_idx]):
                        parts.append(label)
                reasons.append("|".join(parts) if parts else "unknown")
            failure_rows.insert(1, "reason", reasons)
            fail_samples = failure_rows
            if logger is not None:
                logger.info(
                    "ALLOW_CTX_FILTER_FAIL_SAMPLES symbol=%s allow_name=%s\n%s",
                    symbol,
                    allow_name,
                    fail_samples.to_string(index=False),
                )

        return filtered_mask, counts, fail_samples


def apply_allow_context_filters(
    gating_df: pd.DataFrame,
    symbol_cfg: dict | None,
    logger,
    *,
    phase_e: bool = False,
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
    created_allows: list[str] = []
    forced_zero_allows: list[str] = []
    enabled_allows: list[str] = []
    for allow_rule, rule_cfg in allow_cfg.items():
        if not isinstance(rule_cfg, dict):
            logger.warning("allow_context_filters.%s must be a mapping; skipping.", allow_rule)
            continue
        if not rule_cfg.get("enabled", False):
            continue
        enabled_allows.append(allow_rule)
        # Transition is handled inside GatingPolicy.apply() (Fase D). Avoid double-apply and misleading logs.
        if allow_rule == "ALLOW_transition_failure":
            if allow_rule not in filtered.columns:
                msg = f"allow_context_filters.{allow_rule} enabled but missing in gating_df."
                if phase_e:
                    raise ValueError(msg)
                logger.warning("%s Filling with zeros.", msg)
                filtered[allow_rule] = 0
                created_allows.append(allow_rule)
                forced_zero_allows.append(allow_rule)
            logger.info("allow_context_filters.%s handled in gating.py; skipping second pass.", allow_rule)
            continue
        
        if allow_rule not in filtered.columns:
            msg = f"allow_context_filters.{allow_rule} missing in gating_df"
            if phase_e:
                raise ValueError(f"{msg}. Available={sorted(filtered.columns)}")
            logger.warning("%s; creating column with zeros.", msg)
            filtered[allow_rule] = 0
            created_allows.append(allow_rule)
            forced_zero_allows.append(allow_rule)
            continue

        allow_series = filtered[allow_rule].astype(bool)
        before_rate = float(allow_series.mean()) if len(allow_series) else 0.0
        require_all = rule_cfg.get("require_all", True)

        masks: list[pd.Series] = []
        applied_conditions: list[str] = []

        missing_required: list[str] = []

        sessions_in = rule_cfg.get("sessions_in")
        if sessions_in is not None:
            col_name = _resolve_column(filtered, ["session", "session_bucket"])
            if col_name is None:
                missing_required.append("session")
            else:
                allowed = {str(val) for val in sessions_in}
                mask = filtered[col_name].astype(str).isin(allowed)
                masks.append(mask)
                applied_conditions.append(f"sessions_in={sorted(allowed)} via {col_name}")

        state_age_min = rule_cfg.get("state_age_min")
        if state_age_min is not None:
            col_name = _resolve_column(filtered, ["state_age"])
            if col_name is None:
                missing_required.append("state_age")
            else:
                series = pd.to_numeric(filtered[col_name], errors="coerce")
                mask = series >= float(state_age_min)
                masks.append(mask)
                applied_conditions.append(f"state_age_min>={state_age_min} via {col_name}")

        state_age_max = rule_cfg.get("state_age_max")
        if state_age_max is not None:
            col_name = _resolve_column(filtered, ["state_age"])
            if col_name is None:
                missing_required.append("state_age")
            else:
                series = pd.to_numeric(filtered[col_name], errors="coerce")
                mask = series <= float(state_age_max)
                masks.append(mask)
                applied_conditions.append(f"state_age_max<={state_age_max} via {col_name}")

        dist_vwap_atr_min = rule_cfg.get("dist_vwap_atr_min")
        if dist_vwap_atr_min is not None:
            col_name = _resolve_column(filtered, ["dist_vwap_atr"])
            if col_name is None:
                missing_required.append("dist_vwap_atr")
            else:
                series = pd.to_numeric(filtered[col_name], errors="coerce")
                mask = series >= float(dist_vwap_atr_min)
                masks.append(mask)
                applied_conditions.append(f"dist_vwap_atr_min>={dist_vwap_atr_min} via {col_name}")

        dist_vwap_atr_max = rule_cfg.get("dist_vwap_atr_max")
        if dist_vwap_atr_max is not None:
            col_name = _resolve_column(filtered, ["dist_vwap_atr"])
            if col_name is None:
                missing_required.append("dist_vwap_atr")
            else:
                series = pd.to_numeric(filtered[col_name], errors="coerce")
                mask = series <= float(dist_vwap_atr_max)
                masks.append(mask)
                applied_conditions.append(f"dist_vwap_atr_max<={dist_vwap_atr_max} via {col_name}")

        breakmag_min = rule_cfg.get("breakmag_min")
        if breakmag_min is not None:
            col_name = _resolve_column(filtered, ["BreakMag"])
            if col_name is None:
                missing_required.append("BreakMag")
            else:
                series = pd.to_numeric(filtered[col_name], errors="coerce")
                mask = series >= float(breakmag_min)
                masks.append(mask)
                applied_conditions.append(f"breakmag_min>={breakmag_min} via {col_name}")

        breakmag_max = rule_cfg.get("breakmag_max")
        if breakmag_max is not None:
            col_name = _resolve_column(filtered, ["BreakMag"])
            if col_name is None:
                missing_required.append("BreakMag")
            else:
                series = pd.to_numeric(filtered[col_name], errors="coerce")
                mask = series <= float(breakmag_max)
                masks.append(mask)
                applied_conditions.append(f"breakmag_max<={breakmag_max} via {col_name}")

        reentry_min = rule_cfg.get("reentry_min")
        if reentry_min is not None:
            col_name = _resolve_column(filtered, ["ReentryCount"])
            if col_name is None:
                missing_required.append("ReentryCount")
            else:
                series = pd.to_numeric(filtered[col_name], errors="coerce")
                mask = series >= float(reentry_min)
                masks.append(mask)
                applied_conditions.append(f"reentry_min>={reentry_min} via {col_name}")

        reentry_max = rule_cfg.get("reentry_max")
        if reentry_max is not None:
            col_name = _resolve_column(filtered, ["ReentryCount"])
            if col_name is None:
                missing_required.append("ReentryCount")
            else:
                series = pd.to_numeric(filtered[col_name], errors="coerce")
                mask = series <= float(reentry_max)
                masks.append(mask)
                applied_conditions.append(f"reentry_max<={reentry_max} via {col_name}")

        if missing_required:
            missing_required = sorted(set(missing_required))
            msg = (
                f"allow_context_filters.{allow_rule} missing required columns={missing_required}"
            )
            if phase_e:
                raise ValueError(f"{msg}. Available={sorted(filtered.columns)}")
            logger.warning("%s; forcing ALLOW=0.", msg)
            filtered[allow_rule] = 0
            forced_zero_allows.append(allow_rule)
            continue

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

    if enabled_allows:
        allow_pcts = {}
        for allow_rule in enabled_allows:
            if allow_rule not in filtered.columns:
                continue
            pct = float(filtered[allow_rule].fillna(0).astype(int).mean() * 100.0)
            allow_pcts[allow_rule] = pct
            if pct == 0.0:
                logger.warning("ALLOW %s has 0%% of 1s after context filters.", allow_rule)
        logger.info(
            "ALLOW context columns created=%s forced_zero=%s allow_pct_ones=%s",
            sorted(set(created_allows)),
            sorted(set(forced_zero_allows)),
            allow_pcts,
        )

    return filtered


__all__ = [
    "GatingThresholds",
    "GatingPolicy",
    "apply_allow_context_filters",
    "build_transition_gating_thresholds",
]
