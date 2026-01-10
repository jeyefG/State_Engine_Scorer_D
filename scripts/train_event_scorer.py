"""Train the Event Scorer model from MT5 data."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sklearn.metrics import roc_auc_score

from state_engine.events import detect_events, label_events
from state_engine.features import FeatureConfig, FeatureEngineer
from state_engine.gating import GatingPolicy
from state_engine.model import StateEngineModel
from state_engine.mt5_connector import MT5Connector
from state_engine.scoring import EventScorer, EventScorerConfig, FeatureBuilder


def parse_args() -> argparse.Namespace:
    default_end = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    default_start = (datetime.today() - timedelta(days=180)).strftime("%Y-%m-%d")
    parser = argparse.ArgumentParser(description="Train Event Scorer model.")
    parser.add_argument("--symbol", default="EURUSD", help="Símbolo MT5 (ej. EURUSD)")
    parser.add_argument("--start", default=default_start, help="Fecha inicio (YYYY-MM-DD) para descarga M5/H1")
    parser.add_argument("--end", default=default_end, help="Fecha fin (YYYY-MM-DD) para descarga M5/H1")
    parser.add_argument("--state-model", type=Path, default=None, help="Ruta del modelo State Engine (pkl)")
    parser.add_argument("--model-out", type=Path, default=None, help="Ruta de salida para Event Scorer")
    parser.add_argument("--model-dir", type=Path, default=Path("models"), help="Directorio base para modelos")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Ratio train/calibración")
    parser.add_argument("--k-bars", type=int, default=24, help="Ventana futura K para etiquetas")
    parser.add_argument("--reward-r", type=float, default=1.0, help="R múltiplo para TP proxy")
    parser.add_argument("--sl-mult", type=float, default=1.0, help="Multiplicador de ATR para SL proxy")
    parser.add_argument("--log-level", default="INFO", help="Logging level (INFO, DEBUG, WARNING)")
    return parser.parse_args()


def setup_logging(level: str) -> logging.Logger:
    logger = logging.getLogger("event_scorer")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)-8s %(asctime)s | %(message)s")
    handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def build_h1_context(
    ohlcv_h1: pd.DataFrame,
    state_model: StateEngineModel,
    feature_engineer: FeatureEngineer,
    gating: GatingPolicy,
) -> pd.DataFrame:
    full_features = feature_engineer.compute_features(ohlcv_h1)
    features = feature_engineer.training_features(full_features)
    outputs = state_model.predict_outputs(features)
    allows = gating.apply(outputs, features=full_features)
    ctx = pd.concat([outputs[["state_hat", "margin"]], allows], axis=1)
    ctx = ctx.shift(1)
    return ctx


def merge_h1_m5(ctx_h1: pd.DataFrame, ohlcv_m5: pd.DataFrame) -> pd.DataFrame:
    h1 = ctx_h1.reset_index().rename(columns={"time": "time"}).sort_values("time")
    m5 = ohlcv_m5.reset_index().rename(columns={"time": "time"}).sort_values("time")
    merged = pd.merge_asof(m5, h1, on="time", direction="backward")
    merged = merged.set_index("time")
    merged = merged.rename(columns={"state_hat": "state_hat_H1", "margin": "margin_H1"})
    return merged


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.log_level)

    def _safe_symbol(sym: str) -> str:
        return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in sym)

    model_path = args.state_model
    if model_path is None:
        model_path = args.model_dir / f"{_safe_symbol(args.symbol)}_state_engine.pkl"

    scorer_out = args.model_out
    if scorer_out is None:
        scorer_out = args.model_dir / f"{_safe_symbol(args.symbol)}_event_scorer.pkl"

    connector = MT5Connector()
    fecha_inicio = pd.to_datetime(args.start)
    fecha_fin = pd.to_datetime(args.end)

    ohlcv_h1 = connector.obtener_h1(args.symbol, fecha_inicio, fecha_fin)
    ohlcv_m5 = connector.obtener_m5(args.symbol, fecha_inicio, fecha_fin)
    server_now = connector.server_now(args.symbol).tz_localize(None)

    h1_cutoff = server_now.floor("H")
    m5_cutoff = server_now.floor("5min")
    ohlcv_h1 = ohlcv_h1[ohlcv_h1.index < h1_cutoff]
    ohlcv_m5 = ohlcv_m5[ohlcv_m5.index < m5_cutoff]

    if not model_path.exists():
        raise FileNotFoundError(f"State model not found: {model_path}")

    state_model = StateEngineModel()
    state_model.load(model_path)

    feature_engineer = FeatureEngineer(FeatureConfig())
    gating = GatingPolicy()
    ctx_h1 = build_h1_context(ohlcv_h1, state_model, feature_engineer, gating)

    df_m5_ctx = merge_h1_m5(ctx_h1, ohlcv_m5)
    df_m5_ctx = df_m5_ctx.dropna(subset=["state_hat_H1", "margin_H1"])

    events = detect_events(df_m5_ctx)
    if events.empty:
        logger.warning("No events detected; exiting.")
        return

    events = label_events(events, ohlcv_m5, args.k_bars, args.reward_r, args.sl_mult)
    events = events.dropna(subset=["label"])
    events = events.sort_index()

    if events.empty:
        logger.warning("No labeled events after filtering.")
        return

    feature_builder = FeatureBuilder()
    features_all = feature_builder.build(df_m5_ctx)
    event_features = features_all.reindex(events.index)
    event_features = feature_builder.add_family_features(event_features, events["family_id"])

    labels = events["label"].astype(int)
    split_idx = int(len(events) * args.train_ratio)
    X_train = event_features.iloc[:split_idx]
    y_train = labels.iloc[:split_idx]
    X_calib = event_features.iloc[split_idx:]
    y_calib = labels.iloc[split_idx:]

    scorer = EventScorer(EventScorerConfig())
    scorer.fit(X_train, y_train, calib_features=X_calib, calib_labels=y_calib)

    if not y_calib.empty:
        preds = scorer.predict_proba(X_calib)
        auc = roc_auc_score(y_calib, preds)
        logger.info("AUC_calib=%.4f samples=%s", auc, len(y_calib))

    metadata = {
        "symbol": args.symbol,
        "train_ratio": args.train_ratio,
        "k_bars": args.k_bars,
        "reward_r": args.reward_r,
        "sl_mult": args.sl_mult,
        "feature_count": event_features.shape[1],
    }
    scorer.save(scorer_out, metadata=metadata)

    logger.info("events_total=%s labeled=%s", len(events), len(labels))
    logger.info("label_distribution=%s", labels.value_counts(normalize=True).to_dict())
    logger.info("model_out=%s", scorer_out)


if __name__ == "__main__":
    main()
