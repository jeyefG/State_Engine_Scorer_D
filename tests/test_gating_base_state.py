import pandas as pd
import pytest

from state_engine.gating import GatingPolicy, GatingThresholds
from state_engine.labels import StateLabels


def test_gating_requires_base_state_for_custom_allow() -> None:
    outputs = pd.DataFrame(
        {
            "state_hat": [StateLabels.BALANCE, StateLabels.TREND],
            "margin": [0.2, 0.3],
        }
    )
    features = pd.DataFrame({"BreakMag": [0.3, 0.4], "ReentryCount": [1.2, 1.0]})
    config_meta = {
        "allow_context_filters": {
            "ALLOW_custom_context": {
                "enabled": True,
                "sessions_in": ["LONDON"],
            }
        }
    }

    policy = GatingPolicy(GatingThresholds())
    with pytest.raises(ValueError, match="base_state"):
        policy.apply(outputs, features, config_meta=config_meta)
