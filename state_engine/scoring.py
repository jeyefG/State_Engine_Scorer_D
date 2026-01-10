"""Event scorer (ML) for M5 candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from .events import EventFamily
from .labels import StateLabels


@dataclass(frozen=True)
class EventScorerConfig:
    num_leaves: int = 31
    learning_rate: float = 0.05
    n_estimators: int = 300
    max_depth: int = -1
    random_state: int = 42
    calibration_method: str = "sigmoid"


class FeatureBuilder:
    """Build M5 features for event scoring."""

    def __init__(self, atr_window: int = 14, micro_window: int = 6) -> None:
        self.atr_window = atr_window
        self.micro_window = micro_window

    def build(self, df_m5_ctx: pd.DataFrame) -> pd.DataFrame:
        required = {"open", "high", "low", "close", "state_hat_H1", "margin_H1"}
        missing = required - set(df_m5_ctx.columns)
        if missing:
            raise ValueError(f"Missing required columns for feature building: {sorted(missing)}")

        df = df_m5_ctx.copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]
        open_ = df["open"]

        returns_1 = close.pct_change()
        returns_3 = close.pct_change(3)
        realized_vol = returns_1.rolling(self.micro_window).std()
        atr_short = _atr(high, low, close, self.atr_window)

        range_ = (high - low)
        body = (close - open_).abs()
        upper_wick = high - np.maximum(open_, close)
        lower_wick = np.minimum(open_, close) - low
        range_safe = range_.replace(0, np.nan)

        body_ratio = body / range_safe
        upper_ratio = upper_wick / range_safe
        lower_ratio = lower_wick / range_safe

        prev_high = high.shift(1)
        prev_low = low.shift(1)
        overlap = (np.minimum(high, prev_high) - np.maximum(low, prev_low)).clip(lower=0)
        overlap_ratio = overlap / range_safe

        chop_proxy = (range_.rolling(self.micro_window).sum()) / (
            high.rolling(self.micro_window).max() - low.rolling(self.micro_window).min()
        ).replace(0, np.nan)

        roc = close.diff(self.micro_window) / close.shift(self.micro_window)
        range_atr = range_ / atr_short.replace(0, np.nan)
        recent_high = high.rolling(self.micro_window).max()
        recent_low = low.rolling(self.micro_window).min()
        dist_to_high = (recent_high - close) / atr_short.replace(0, np.nan)
        dist_to_low = (close - recent_low) / atr_short.replace(0, np.nan)
        mom_atr = close.diff(self.micro_window) / atr_short.replace(0, np.nan)

        hours = df.index.hour.to_series(index=df.index)
        hour_sin = np.sin(2 * np.pi * hours / 24.0)
        hour_cos = np.cos(2 * np.pi * hours / 24.0)

        features = pd.DataFrame(
            {
                "ret_1": returns_1,
                "ret_3": returns_3,
                "realized_vol": realized_vol,
                "atr_short": atr_short,
                "range": range_,
                "body_ratio": body_ratio,
                "upper_wick_ratio": upper_ratio,
                "lower_wick_ratio": lower_ratio,
                "overlap_ratio": overlap_ratio,
                "chop_proxy": chop_proxy,
                "roc": roc,
                "range_atr": range_atr,
                "dist_to_high_atr": dist_to_high,
                "dist_to_low_atr": dist_to_low,
                "mom_atr": mom_atr,
                "hour_sin": hour_sin,
                "hour_cos": hour_cos,
                "margin_H1": df["margin_H1"],
            },
            index=df.index,
        )

        state_one_hot = pd.get_dummies(df["state_hat_H1"].map(_state_name), prefix="state")
        for state_name in ("balance", "transition", "trend"):
            col = f"state_{state_name}"
            if col not in state_one_hot.columns:
                state_one_hot[col] = 0.0
        allow_cols = [col for col in df.columns if col.startswith("ALLOW_")]
        allow_features = df[allow_cols].astype(float) if allow_cols else pd.DataFrame(index=df.index)

        features = pd.concat([features, state_one_hot, allow_features], axis=1)
        return features

    def add_family_features(self, features: pd.DataFrame, family_ids: pd.Series) -> pd.DataFrame:
        family_one_hot = pd.get_dummies(family_ids, prefix="family")
        for family in EventFamily:
            col = f"family_{family.value}"
            if col not in family_one_hot.columns:
                family_one_hot[col] = 0.0
        return pd.concat([features, family_one_hot], axis=1)


class EventScorer:
    """LightGBM scorer with probability calibration."""

    def __init__(self, config: EventScorerConfig | None = None) -> None:
        self.config = config or EventScorerConfig()
        self._model: lgb.LGBMClassifier | None = None
        self._calibrator: CalibratedClassifierCV | None = None
        self.metadata: dict[str, Any] = {}

    def fit(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        calib_features: pd.DataFrame | None = None,
        calib_labels: pd.Series | None = None,
    ) -> None:
        model = lgb.LGBMClassifier(
            objective="binary",
            num_leaves=self.config.num_leaves,
            learning_rate=self.config.learning_rate,
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            random_state=self.config.random_state,
        )
        model.fit(features, labels)
        self._model = model

        if calib_features is not None and calib_labels is not None and not calib_labels.empty:
            if len(calib_labels) < 2:
                self._calibrator = None
            else:
                cv_folds = min(5, len(calib_labels))
                calibrator = CalibratedClassifierCV(
                    FrozenEstimator(model),
                    method=self.config.calibration_method,
                    cv=cv_folds,
                )
                calibrator.fit(calib_features, calib_labels)
                self._calibrator = calibrator
        else:
            self._calibrator = None

    def predict_proba(self, features: pd.DataFrame) -> pd.Series:
        self._require_fitted()
        if self._calibrator is not None:
            proba = self._calibrator.predict_proba(features)[:, 1]
        else:
            proba = self._model.predict_proba(features)[:, 1]
        return pd.Series(proba, index=features.index, name="edge_score")

    def predict_score(self, features: pd.DataFrame) -> pd.Series:
        return self.predict_proba(features)

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self._model,
            "calibrator": self._calibrator,
            "metadata": {
                "config": asdict(self.config),
                **(metadata or {}),
            },
        }
        joblib.dump(payload, path)

    def load(self, path: str | Path) -> None:
        payload = joblib.load(Path(path))
        if not isinstance(payload, dict):
            raise ValueError("Invalid scorer payload; expected dict.")
        self._model = payload.get("model")
        self._calibrator = payload.get("calibrator")
        self.metadata = payload.get("metadata", {})

    def _require_fitted(self) -> None:
        if self._model is None:
            raise RuntimeError("EventScorer must be fit or loaded before use.")


class EventScorerBundle:
    """Family-specific scorer bundle with a compatible interface."""

    global_key = "__global__"

    def __init__(self, config: EventScorerConfig | None = None) -> None:
        self.config = config or EventScorerConfig()
        self.scorers: dict[str, EventScorer] = {}
        self.metadata: dict[str, Any] = {}

    def fit(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        family_ids: pd.Series,
        calib_features: pd.DataFrame | None = None,
        calib_labels: pd.Series | None = None,
        calib_family_ids: pd.Series | None = None,
    ) -> None:
        families = family_ids.dropna().unique()
        for family in families:
            family_mask = family_ids == family
            if family_mask.sum() == 0:
                continue
            scorer = EventScorer(self.config)
            calib_X = None
            calib_y = None
            if calib_features is not None and calib_labels is not None and calib_family_ids is not None:
                calib_mask = calib_family_ids == family
                if calib_mask.any():
                    calib_X = calib_features.loc[calib_mask]
                    calib_y = calib_labels.loc[calib_mask]
            scorer.fit(features.loc[family_mask], labels.loc[family_mask], calib_X, calib_y)
            self.scorers[str(family)] = scorer

    def predict_proba(self, features: pd.DataFrame, family_ids: pd.Series) -> pd.Series:
        scores = pd.Series(index=features.index, dtype=float, name="edge_score")
        for family, scorer in self.scorers.items():
            if family == self.global_key:
                continue
            mask = family_ids == family
            if mask.any():
                scores.loc[mask] = scorer.predict_proba(features.loc[mask]).values
        global_scorer = self.scorers.get(self.global_key)
        if global_scorer is not None:
            missing = scores.isna()
            if missing.any():
                scores.loc[missing] = global_scorer.predict_proba(features.loc[missing]).values
        return scores

    def predict_score(self, features: pd.DataFrame, family_ids: pd.Series) -> pd.Series:
        return self.predict_proba(features, family_ids)

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "scorers": self.scorers,
            "metadata": {
                "config": asdict(self.config),
                **(metadata or {}),
            },
        }
        joblib.dump(payload, path)

    def load(self, path: str | Path) -> None:
        payload = joblib.load(Path(path))
        if not isinstance(payload, dict):
            raise ValueError("Invalid scorer payload; expected dict.")
        if "scorers" in payload:
            self.scorers = payload.get("scorers", {})
            self.metadata = payload.get("metadata", {})
            return
        if "model" in payload:
            scorer = EventScorer(self.config)
            scorer._model = payload.get("model")
            scorer._calibrator = payload.get("calibrator")
            scorer.metadata = payload.get("metadata", {})
            self.scorers = {self.global_key: scorer}
            self.metadata = scorer.metadata
            return
        raise ValueError("Invalid scorer payload; expected scorers or model.")


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window).mean()


def _state_name(value: int | float) -> str:
    try:
        label = StateLabels(int(value))
    except Exception:
        return "unknown"
    if label == StateLabels.BALANCE:
        return "balance"
    if label == StateLabels.TRANSITION:
        return "transition"
    return "trend"


__all__ = [
    "EventScorerConfig",
    "FeatureBuilder",
    "EventScorer",
    "EventScorerBundle",
    "EventFamily",
]
