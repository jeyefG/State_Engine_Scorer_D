"""Train the Event Scorer model from MT5 data.

The scorer uses a triple-barrier continuous outcome (r_outcome) and reports
ranking metrics like lift@K to gauge whether top-ranked events outperform the
base rate. lift@K = precision@K / base_rate.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import numpy as np

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
from state_engine.scoring import EventScorer, EventScorerBundle, EventScorerConfig, FeatureBuilder


def parse_args() -> argparse.Namespace:
    default_end = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    default_start = (datetime.today() - timedelta(days=180)).strftime("%Y-%m-%d")
    parser = argparse.ArgumentParser(description="Train Event Scorer model.")
    parser.add_argument("--symbol", default="EURUSD", help="Símbolo MT5 (ej. EURUSD)")
    parser.add_argument("--start", default=default_start, help="Fecha inicio (YYYY-MM-DD) para descarga M5/H1")
    parser.add_argument("--end", default=default_end, help="Fecha fin (YYYY-MM-DD) para descarga M5/H1")
    parser.add_argument("--state-model", type=Path, default=None, help="Ruta del modelo State Engine (pkl)")
    parser.add_argument("--model-out", type=Path, default=None, help="Ruta de salida para Event Scorer")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(PROJECT_ROOT / "state_engine" / "models"),
        help="Directorio base para modelos",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Ratio train/calibración")
    parser.add_argument("--k-bars", type=int, default=24, help="Ventana futura K para etiquetas")
    parser.add_argument("--reward-r", type=float, default=1.0, help="R múltiplo para TP proxy")
    parser.add_argument("--sl-mult", type=float, default=1.0, help="Multiplicador de ATR para SL proxy")
    parser.add_argument("--r-thr", type=float, default=0.0, help="Umbral para label binario basado en r_outcome")
    parser.add_argument("--tie-break", default="distance", choices=["distance", "worst"], help="Tie-break TP/SL")
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
    logger = logging.getLogger("event_scorer")
    h1 = ctx_h1.copy().sort_index()
    m5 = ohlcv_m5.copy().sort_index()
    if getattr(h1.index, "tz", None) is not None:
        h1.index = h1.index.tz_localize(None)
    if getattr(m5.index, "tz", None) is not None:
        m5.index = m5.index.tz_localize(None)
    h1 = h1.reset_index().rename(columns={h1.index.name or "index": "time"})
    m5 = m5.reset_index().rename(columns={m5.index.name or "index": "time"})
    merged = pd.merge_asof(m5, h1, on="time", direction="backward")
    merged = merged.set_index("time")
    merged = merged.rename(columns={"state_hat": "state_hat_H1", "margin": "margin_H1"})
    missing_ctx = merged[["state_hat_H1", "margin_H1"]].isna().mean()
    if (missing_ctx > 0.25).any():
        logger.warning("High missing context after merge: %s", missing_ctx.to_dict())
    return merged


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.log_level)
    min_samples_train = 200

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

    h1_cutoff = server_now.floor("h")
    m5_cutoff = server_now.floor("5min")
    ohlcv_h1 = ohlcv_h1[ohlcv_h1.index < h1_cutoff]
    ohlcv_m5 = ohlcv_m5[ohlcv_m5.index < m5_cutoff]
    logger.info(
        "Period: %s -> %s | h1_cutoff=%s m5_cutoff=%s",
        fecha_inicio,
        fecha_fin,
        h1_cutoff,
        m5_cutoff,
    )
    logger.info("Rows: H1=%s M5=%s", len(ohlcv_h1), len(ohlcv_m5))

    if not model_path.exists():
        raise FileNotFoundError(f"State model not found: {model_path}")

    state_model = StateEngineModel()
    state_model.load(model_path)

    feature_engineer = FeatureEngineer(FeatureConfig())
    gating = GatingPolicy()
    ctx_h1 = build_h1_context(ohlcv_h1, state_model, feature_engineer, gating)

    df_m5_ctx = merge_h1_m5(ctx_h1, ohlcv_m5)
    logger.info("Rows after merge: M5_ctx=%s", len(df_m5_ctx))
    df_m5_ctx = df_m5_ctx.dropna(subset=["state_hat_H1", "margin_H1"])
    logger.info("Rows after dropna ctx: M5_ctx=%s", len(df_m5_ctx))

    events = detect_events(df_m5_ctx)
    if events.empty:
        logger.warning("No events detected; exiting.")
        return
    logger.info("Detected events by family:\n%s", events["family_id"].value_counts().to_string())

    events = label_events(
        events,
        ohlcv_m5,
        args.k_bars,
        args.reward_r,
        args.sl_mult,
        r_thr=args.r_thr,
        tie_break=args.tie_break,
    )
    events = events.dropna(subset=["label", "r_outcome"])
    events = events.sort_index()

    if events.empty:
        logger.warning("No labeled events after filtering.")
        return
    logger.info("Labeled events by family:\n%s", events["family_id"].value_counts().to_string())

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
    r_calib = events["r_outcome"].iloc[split_idx:]
    fam_train = events["family_id"].iloc[:split_idx]
    fam_calib = events["family_id"].iloc[split_idx:]

    family_summary = pd.DataFrame(
        {
            "family_id": sorted(set(events["family_id"])),
        }
    )
    family_summary["samples_train"] = family_summary["family_id"].map(fam_train.value_counts()).fillna(0).astype(int)
    family_summary["samples_calib"] = family_summary["family_id"].map(fam_calib.value_counts()).fillna(0).astype(int)
    family_summary["base_rate_train"] = family_summary["family_id"].map(y_train.groupby(fam_train).mean())
    family_summary["base_rate_calib"] = family_summary["family_id"].map(y_calib.groupby(fam_calib).mean())
    logger.info("Family counts:\n%s", family_summary.to_string(index=False))

    if y_train.nunique() < 2:
        logger.error("Global training labels have a single class; cannot train scorer.")
        return

    scorer = EventScorerBundle(EventScorerConfig())
    global_scorer = EventScorer(scorer.config)
    global_scorer.fit(X_train, y_train, X_calib, y_calib)
    scorer.scorers[scorer.global_key] = global_scorer

    family_summary["train_unique"] = family_summary["family_id"].map(y_train.groupby(fam_train).nunique())
    family_summary["calib_unique"] = family_summary["family_id"].map(y_calib.groupby(fam_calib).nunique())
    family_summary["status"] = "TRAINED"

    for family_id in family_summary["family_id"]:
        train_mask = fam_train == family_id
        calib_mask = fam_calib == family_id
        train_count = int(train_mask.sum())
        calib_count = int(calib_mask.sum())
        train_unique = int(family_summary.loc[family_summary["family_id"] == family_id, "train_unique"].fillna(0).iloc[0])
        calib_unique = int(family_summary.loc[family_summary["family_id"] == family_id, "calib_unique"].fillna(0).iloc[0])

        if train_count < min_samples_train:
            family_summary.loc[family_summary["family_id"] == family_id, "status"] = "SKIP_FAMILY_LOW_SAMPLES"
            logger.warning("Skip %s: low samples (%s)", family_id, train_count)
            continue
        if train_unique < 2 or calib_unique < 2:
            family_summary.loc[family_summary["family_id"] == family_id, "status"] = "SKIP_FAMILY_SINGLE_CLASS"
            logger.warning(
                "Skip %s: missing class (train_unique=%s calib_unique=%s)",
                family_id,
                train_unique,
                calib_unique,
            )
            continue

        scorer_family = EventScorer(scorer.config)
        calib_X = X_calib.loc[calib_mask] if calib_count else None
        calib_y = y_calib.loc[calib_mask] if calib_count else None
        scorer_family.fit(X_train.loc[train_mask], y_train.loc[train_mask], calib_X, calib_y)
        scorer.scorers[str(family_id)] = scorer_family

    logger.info("Family training status:\n%s", family_summary.to_string(index=False))

    def _topk_indices(scores: pd.Series, labels_: pd.Series, k: int, seed: int = 7) -> pd.Index:
        if scores.empty:
            return pd.Index([])
        k_eff = min(k, len(scores))
        if k_eff == 0:
            return pd.Index([])
        if scores.nunique(dropna=False) <= 1:
            rng = np.random.default_rng(seed)
            chosen = rng.choice(labels_.index.to_numpy(), size=k_eff, replace=False)
            return pd.Index(chosen)
        return scores.nlargest(k_eff).index

    def precision_at_k(scores: pd.Series, labels_: pd.Series, k: int, seed: int = 7) -> float:
        top_idx = _topk_indices(scores, labels_, k, seed=seed)
        if top_idx.empty:
            return float("nan")
        return float(labels_.loc[top_idx].mean())

    def lift_at_k(scores: pd.Series, labels_: pd.Series, k: int, seed: int = 7) -> float:
        base_rate = float(labels_.mean())
        if base_rate == 0 or scores.empty:
            return float("nan")
        return precision_at_k(scores, labels_, k, seed=seed) / base_rate

    def mean_r_topk(scores: pd.Series, r_outcome: pd.Series, k: int, seed: int = 7) -> float:
        top_idx = _topk_indices(scores, r_outcome, k, seed=seed)
        if top_idx.empty:
            return float("nan")
        return float(r_outcome.loc[top_idx].mean())

    def summarize_metrics(
        scope: str,
        scores: pd.Series,
        labels_: pd.Series,
        r_outcome: pd.Series,
        seed: int = 7,
    ) -> dict[str, float]:
        base_rate = float(labels_.mean()) if not labels_.empty else float("nan")
        r_mean_all = float(r_outcome.mean()) if not r_outcome.empty else float("nan")
        metrics = {
            "scope": scope,
            "samples": len(labels_),
            "base_rate": base_rate,
            "auc": float("nan"),
            "lift@10": lift_at_k(scores, labels_, 10, seed=seed),
            "lift@20": lift_at_k(scores, labels_, 20, seed=seed),
            "lift@50": lift_at_k(scores, labels_, 50, seed=seed),
            "r_mean@10": mean_r_topk(scores, r_outcome, 10, seed=seed),
            "r_mean@20": mean_r_topk(scores, r_outcome, 20, seed=seed),
            "r_mean@50": mean_r_topk(scores, r_outcome, 50, seed=seed),
            "r_mean_all": r_mean_all,
        }
        if len(labels_.unique()) > 1:
            metrics["auc"] = roc_auc_score(labels_, scores)
        return metrics

    metrics_rows: list[dict[str, float]] = []
    baseline_rows: list[dict[str, float]] = []

    if not y_calib.empty:
        preds = scorer.predict_proba(X_calib, fam_calib)
        metrics_rows.append(summarize_metrics("global", preds, y_calib, r_calib))
        rng = np.random.default_rng(7)
        baseline_scores = pd.Series(rng.random(len(y_calib)), index=y_calib.index)
        logger.debug(
            "baseline_scores stats: nunique=%s min=%s max=%s",
            baseline_scores.nunique(dropna=False),
            baseline_scores.min(),
            baseline_scores.max(),
        )
        if baseline_scores.nunique(dropna=False) <= 1:
            logger.warning("Baseline scores are constant; using random sampling for topK metrics.")
        baseline_rows.append(summarize_metrics("global_baseline", baseline_scores, y_calib, r_calib))

        for family_id, fam_labels in y_calib.groupby(fam_calib):
            fam_scores = preds.loc[fam_labels.index]
            fam_r = r_calib.loc[fam_labels.index]
            metrics_rows.append(summarize_metrics(family_id, fam_scores, fam_labels, fam_r))
            baseline_rows.append(
                summarize_metrics(
                    f"{family_id}_baseline",
                    baseline_scores.loc[fam_labels.index],
                    fam_labels,
                    fam_r,
                )
            )

        margin_series = df_m5_ctx["margin_H1"].reindex(y_calib.index)
        bins = pd.Series(index=margin_series.index, dtype="object")
        margin_non_na = margin_series.dropna()
        if not margin_non_na.empty:
            try:
                bins.loc[margin_non_na.index] = pd.qcut(margin_non_na, q=3, duplicates="drop")
            except ValueError:
                bins = pd.Series(index=margin_series.index, dtype="object")
        if not bins.dropna().empty:
            for bin_label, bin_labels in y_calib.groupby(bins, observed=True):
                bin_scores = preds.loc[bin_labels.index]
                bin_r = r_calib.loc[bin_labels.index]
                metrics_rows.append(summarize_metrics(f"margin_bin_{bin_label}", bin_scores, bin_labels, bin_r))
                baseline_rows.append(
                    summarize_metrics(
                        f"margin_bin_{bin_label}_baseline",
                        baseline_scores.loc[bin_labels.index],
                        bin_labels,
                        bin_r,
                    )
                )

    metrics_df = pd.DataFrame(metrics_rows)
    baseline_df = pd.DataFrame(baseline_rows)
    if not metrics_df.empty:
        logger.info("Event scorer metrics (calib):\n%s", metrics_df.to_string(index=False))
    if not baseline_df.empty:
        logger.info("Baseline metrics (calib):\n%s", baseline_df.to_string(index=False))

    if not metrics_df.empty and not baseline_df.empty:
        lift_cols = ["lift@10", "lift@20", "lift@50"]
        merged = metrics_df.set_index("scope")[lift_cols].fillna(0)
        baseline = baseline_df.set_index("scope")[lift_cols].fillna(0)
        improved = False
        for scope in merged.index:
            if scope.endswith("_baseline"):
                continue
            base_scope = f"{scope}_baseline"
            if base_scope in baseline.index:
                if (merged.loc[scope] > baseline.loc[base_scope]).any():
                    improved = True
                    break
        if not improved:
            logger.warning("No family improved lift@K vs baseline; review signal quality.")

    metrics_summary = {}
    if not metrics_df.empty:
        global_row = metrics_df[metrics_df["scope"] == "global"]
        if not global_row.empty:
            metrics_summary = global_row.iloc[0].to_dict()

    metadata = {
        "symbol": args.symbol,
        "train_ratio": args.train_ratio,
        "k_bars": args.k_bars,
        "reward_r": args.reward_r,
        "sl_mult": args.sl_mult,
        "r_thr": args.r_thr,
        "tie_break": args.tie_break,
        "feature_count": event_features.shape[1],
        "train_date": datetime.now(timezone.utc).isoformat(),
        "metrics_summary": metrics_summary,
    }
    scorer.save(scorer_out, metadata=metadata)

    summary_table = pd.DataFrame(
        [
            {
                "events_total": len(events),
                "labeled": len(labels),
                "feature_count": event_features.shape[1],
            }
        ]
    )
    logger.info("Summary:\n%s", summary_table.to_string(index=False))
    logger.info("label_distribution=%s", labels.value_counts(normalize=True).to_dict())
    logger.info("model_out=%s", scorer_out)

    args.model_dir.mkdir(parents=True, exist_ok=True)
    if not metrics_df.empty:
        metrics_path = args.model_dir / f"metrics_{_safe_symbol(args.symbol)}_event_scorer.csv"
        metrics_df.to_csv(metrics_path, index=False)
        logger.info("metrics_out=%s", metrics_path)
    family_path = args.model_dir / f"family_summary_{_safe_symbol(args.symbol)}_event_scorer.csv"
    family_summary.to_csv(family_path, index=False)
    logger.info("family_summary_out=%s", family_path)

    if not y_calib.empty:
        sample_cols = ["family_id", "side", "label", "r_outcome"]
        sample_df = events.loc[y_calib.index, sample_cols].copy()
        sample_df["score"] = preds
        sample_df["margin_H1"] = df_m5_ctx["margin_H1"].reindex(sample_df.index)
        sample_df = sample_df.sort_values("score", ascending=False).head(10)
        sample_df = sample_df.reset_index().rename(columns={sample_df.index.name or "index": "time"})
        sample_path = args.model_dir / f"calib_top_scored_{_safe_symbol(args.symbol)}_event_scorer.csv"
        sample_df.to_csv(sample_path, index=False)
        logger.info("calib_top_scored_out=%s", sample_path)

    if global_scorer._model is not None and hasattr(global_scorer._model, "feature_importances_"):
        importances = pd.Series(global_scorer._model.feature_importances_, index=event_features.columns)
        top_features = importances.sort_values(ascending=False).head(10)
        logger.info("Top-10 feature importances:\n%s", top_features.to_string())


if __name__ == "__main__":
    main()
