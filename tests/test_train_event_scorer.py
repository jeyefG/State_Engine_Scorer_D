from pathlib import Path

from scripts.train_event_scorer import _save_model_if_ready
from state_engine.scoring import EventScorerBundle


def test_fallback_no_model_saved(tmp_path: Path) -> None:
    scorer = EventScorerBundle()
    model_path = tmp_path / "scorer.pkl"

    saved = _save_model_if_ready(is_fallback=True, scorer=scorer, path=model_path, metadata={})

    assert saved is False
    assert not model_path.exists()
