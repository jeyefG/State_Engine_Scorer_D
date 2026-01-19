import logging

import pytest

from state_engine.pipeline_phase_d import validate_allow_context_requirements


def test_validate_allow_context_requirements_missing_columns_raises() -> None:
    cfg = {
        "allow_context_filters": {
            "ALLOW_transition_failure": {"enabled": True, "sessions_in": ["NY"]}
        }
    }
    available = {"BreakMag", "ReentryCount"}
    with pytest.raises(ValueError, match="missing columns"):
        validate_allow_context_requirements(cfg, available, logger=logging.getLogger("test"))


def test_validate_allow_context_requirements_ok() -> None:
    cfg = {
        "allow_context_filters": {
            "ALLOW_transition_failure": {
                "enabled": True,
                "sessions_in": ["NY"],
                "state_age_max": 2,
                "dist_vwap_atr_max": 1.5,
            }
        }
    }
    available = {"ctx_session_bucket", "ctx_state_age", "ctx_dist_vwap_atr", "BreakMag", "ReentryCount"}
    validate_allow_context_requirements(cfg, available, logger=logging.getLogger("test"))
