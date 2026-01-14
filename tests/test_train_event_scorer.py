from pathlib import Path

import pandas as pd

from scripts.train_event_scorer import (
    _resolve_vwap_report_mode,
    _save_model_if_ready,
    _research_guardrails,
    _build_research_summary_from_grid,
    _persist_research_outputs,
)
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


def test_guardrails_flags_low_calib_samples() -> None:
    score_shape = pd.DataFrame(
        [
            {
                "k": 20,
                "edge_decay": 0.1,
                "temporal_dispersion": 0.5,
                "family_concentration": 0.2,
                "score_tail_slope": 1.0,
            }
        ]
    )
    diagnostics_cfg = {
        "max_family_concentration": 0.6,
        "min_temporal_dispersion": 0.3,
        "max_score_tail_slope": 2.5,
        "min_calib_samples": 200,
    }

    report = _research_guardrails(
        score_shape=score_shape,
        train_metrics=None,
        calib_metrics=None,
        diagnostics_cfg=diagnostics_cfg,
        calib_samples=50,
    )

    assert report.loc[0, "status"] == "RESEARCH_UNSTABLE"
    assert "LOW_CALIB_SAMPLES<200" in report.loc[0, "reasons"]


def test_research_outputs_persisted_only_in_research(tmp_path: Path) -> None:
    grid_results = pd.DataFrame(
        [
            {
                "symbol": "TEST",
                "k_bars": 12,
                "state": "BALANCE",
                "family": "FAMILY_A",
                "session_bucket": "LONDON",
                "month_bucket": "2024-01",
                "n": 250,
                "winrate": 0.6,
                "r_mean": 0.12,
                "p10": -0.05,
                "lift_at_k": 1.2,
                "score_tail_slope": 1.1,
                "qualified": True,
                "fail_reason": "",
            },
            {
                "symbol": "TEST",
                "k_bars": 24,
                "state": "TREND",
                "family": "FAMILY_B",
                "session_bucket": "NY",
                "month_bucket": "2024-02",
                "n": 300,
                "winrate": 0.58,
                "r_mean": 0.08,
                "p10": -0.02,
                "lift_at_k": 1.15,
                "score_tail_slope": 1.05,
                "qualified": True,
                "fail_reason": "",
            },
        ]
    )
    summary = _build_research_summary_from_grid(grid_results)
    research_cfg = {
        "enabled": True,
        "d1_anchor_hour": 0,
        "features": {"session_bucket": True},
        "k_bars_grid": [12, 24],
    }

    _persist_research_outputs(
        model_dir=tmp_path,
        symbol="TEST",
        grid_results=grid_results,
        summary_payload=summary,
        research_cfg=research_cfg,
        config_hash="abc123",
        prompt_version="v1",
        mode_suffix="_reas",
        research_enabled=True,
    )

    grid_path = tmp_path / "research_grid_results_TEST_reas.csv"
    assert grid_path.exists()
    assert len(pd.read_csv(grid_path)) > 1

    summary_path = tmp_path / "research_summary_TEST_reas.json"
    assert summary_path.exists()

    prod_dir = tmp_path / "prod"
    _persist_research_outputs(
        model_dir=prod_dir,
        symbol="TEST",
        grid_results=grid_results,
        summary_payload=summary,
        research_cfg=research_cfg,
        config_hash="abc123",
        prompt_version="v1",
        mode_suffix="_prod",
        research_enabled=False,
    )

    prod_grid_path = prod_dir / "research_grid_results_TEST_prod.csv"
    assert not prod_grid_path.exists()
