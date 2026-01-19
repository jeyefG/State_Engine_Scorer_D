import pytest
import pandas as pd

from state_engine.gating import GatingPolicy
from state_engine.labels import StateLabels


def test_gating_requires_context_columns_for_enabled_allow() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="h")
    outputs = pd.DataFrame(
        {"state_hat": [StateLabels.TRANSITION] * 3, "margin": [0.5, 0.5, 0.5]},
        index=idx,
    )
    features = pd.DataFrame(
        {"BreakMag": [0.3, 0.3, 0.3], "ReentryCount": [1.2, 1.2, 1.2]},
        index=idx,
    )
    cfg = {
        "allow_context_filters": {
            "ALLOW_transition_failure": {"enabled": True, "sessions_in": ["NY"]}
        }
    }
    gating = GatingPolicy()
    with pytest.raises(ValueError, match="session column"):
        gating.apply(outputs, features=features, config_meta=cfg)


def test_gating_materializes_allow_columns_with_context_filters() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="h")
    outputs = pd.DataFrame(
        {"state_hat": [StateLabels.TRANSITION] * 3, "margin": [0.5, 0.5, 0.5]},
        index=idx,
    )
    features = pd.DataFrame(
        {
            "BreakMag": [0.3, 0.3, 0.3],
            "ReentryCount": [1.2, 1.2, 1.2],
            "ctx_session_bucket": ["NY", "NY", "ASIA"],
            "ctx_state_age": [1, 3, 2],
        },
        index=idx,
    )
    cfg = {
        "allow_context_filters": {
            "ALLOW_transition_failure": {
                "enabled": True,
                "sessions_in": ["NY"],
                "state_age_max": 2,
            }
        }
    }
    gating = GatingPolicy()
    result = gating.apply(outputs, features=features, config_meta=cfg)
    assert "ALLOW_transition_failure" in result.columns
    assert set(result["ALLOW_transition_failure"].unique()).issubset({0, 1})
    assert result["ALLOW_transition_failure"].tolist() == [1, 0, 0]
