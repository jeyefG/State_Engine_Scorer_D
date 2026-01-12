from pathlib import Path

import pandas as pd

from scripts.train_event_scorer import _resolve_vwap_report_mode, _save_model_if_ready
from state_engine.events import EventDetectionConfig, EventExtractor
from state_engine.scoring import EventScorerBundle


def test_fallback_no_model_saved(tmp_path: Path) -> None:
    scorer = EventScorerBundle()
    model_path = tmp_path / "scorer.pkl"

    saved = _save_model_if_ready(is_fallback=True, scorer=scorer, path=model_path, metadata={})

    assert saved is False
    assert not model_path.exists()


def test_vwap_reporting_uses_fallback_metadata_session_mode() -> None:
    idx = pd.date_range("2024-01-01 21:50", periods=6, freq="5min")
    df_m5 = pd.DataFrame(
        {
            "open": [100.0] * len(idx),
            "high": [100.0] * len(idx),
            "low": [100.0] * len(idx),
            "close": [100.0] * len(idx),
            "volume": [10.0] * len(idx),
        },
        index=idx,
    )
    config = EventDetectionConfig(vwap_reset_mode=None, vwap_session_cut_hour=22)
    extractor = EventExtractor(config=config)

    events = extractor.extract(df_m5, symbol="TEST")

    assert events.attrs["vwap_reset_mode_effective"] == "session"
    assert events.attrs["vwap_session_cut_hour"] == 22

    report_mode = _resolve_vwap_report_mode(events, df_m5, config)
    assert report_mode == "session"
    assert "daily" not in report_mode
