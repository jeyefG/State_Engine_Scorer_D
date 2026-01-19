import logging

import pandas as pd

from state_engine.gating import apply_allow_context_filters


def test_apply_allow_context_filters_no_config() -> None:
    gating_df = pd.DataFrame({
        "ALLOW_transition_failure": [1, 0, 1],
        "state_age": [1, 2, 3],
    })
    result = apply_allow_context_filters(gating_df, None, logging.getLogger("test"))
    pd.testing.assert_frame_equal(result, gating_df)


def test_apply_allow_context_filters_reduces_allow() -> None:
    gating_df = pd.DataFrame(
        {
            "ALLOW_transition_failure": [1, 1, 1, 0],
            "session_bucket": ["LONDON", "NY", "ASIA", "LONDON"],
            "state_age": [3, 6, 4, 2],
            "dist_vwap_atr": [1.0, 1.2, 1.6, 0.8],
            "BreakMag": [0.3, 0.5, 0.4, 0.3],
            "ReentryCount": [1.2, 0.8, 1.5, 1.1],
        }
    )
    symbol_cfg = {
        "allow_context_filters": {
            "ALLOW_transition_failure": {
                "enabled": True,
                "require_all": True,
                "sessions_in": ["LONDON", "NY"],
                "state_age_max": 5,
                "dist_vwap_atr_max": 1.5,
                "breakmag_min": 0.25,
                "reentry_min": 1.0,
            }
        }
    }
    result = apply_allow_context_filters(gating_df, symbol_cfg, logging.getLogger("test"))
    assert result["ALLOW_transition_failure"].tolist() == [1, 0, 0, 0]


def test_apply_allow_context_filters_missing_column_warns(caplog) -> None:
    gating_df = pd.DataFrame({"ALLOW_transition_failure": [1, 1, 1]})
    symbol_cfg = {
        "allow_context_filters": {
            "ALLOW_transition_failure": {"enabled": True, "dist_vwap_atr_max": 1.5}
        }
    }
    logger = logging.getLogger("test")
    with caplog.at_level(logging.WARNING):
        result = apply_allow_context_filters(gating_df, symbol_cfg, logger)
    assert result["ALLOW_transition_failure"].tolist() == [0, 0, 0]
    assert "missing required columns" in caplog.text
