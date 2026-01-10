# AGENTS.md — State Engine (PA-first)

## Propósito
Este repositorio implementa un State Engine para trading discrecional/semisistemático basado en Price Action (PA).
El objetivo es clasificar el estado del mercado y habilitar/prohibir familias de setups mediante reglas explícitas (ALLOW_*),
reduciendo errores estructurales y sosteniendo estabilidad cognitiva.

El sistema NO predice dirección next-bar.
Gobierna cuándo existen condiciones estructurales para que un trade direccional sea considerado.

================================================================
Decisiones de diseño (no negociables)
================================================================

Separación de capas:
- ML = percepción (clasificar estado / medir edge).
- Reglas = decisión (gating / habilitación).
- Ejecución = determinista y externa al ML.

Métrica principal:
- Expectancy y drawdown condicionados por estado y familia de setup.
- Accuracy / F1 son métricas diagnósticas secundarias.

TRANSICIÓN es crítica:
- Su función es PROHIBIR swing direccional por defecto.
- Solo habilita tácticos de tipo failure / reclaim.

Complejidad mínima:
- Máx. 10–12 features H1.
- Si se requieren más, la definición está mal.

================================================================
Scope reutilizable del repo previo
================================================================

Se recicla SOLO infraestructura, no lógica:
- MT5Connector (OHLCV).
- Utilidades limpias de calendario/sesión.
- Infra mínima de dataset (resampling, timezone, NaN).
- Persistencia de modelos.

Todo lo demás se considera legado.

================================================================
Definición de estado (H1)
================================================================

Timeframe: H1  
Ventanas fijas:
- W = 24 velas H1 (contexto)
- N = 8 velas H1 (reciente)

Estados:
- BALANCE: rotación; baja direccionalidad neta.
- TRANSICIÓN: intento de salida con aceptación incompleta o fallo; alto riesgo direccional.
- TENDENCIA: migración sostenida con aceptación; retrocesos ordenados.

Regla maestra:
El estado se define por comportamiento agregado en ventanas fijas,
no por velas individuales.

================================================================
Normalización por volatilidad
================================================================

ATR_W = ATR(t-W..t)
ATR_N = ATR(t-N..t)

- Métricas de contexto → normalizar por ATR_W
- Métricas recientes / ruptura → normalizar por ATR_N

================================================================
Variables PA-first (≤ t)
================================================================

NetMove (diagnóstico):
|C_t − C_{t-W}| / ATR_W

Path:
∑ |C_i − C_{i−1}| / ATR_W

Efficiency Ratio (ER):
|C_t − C_{t-W}| / ∑ |C_i − C_{i−1}|

Range_W:
(H_W − L_W) / ATR_W

CloseLocation:
(C_t − L_W) / (H_W − L_W) ∈ [0,1]
(si H_W == L_W → 0.5)

BreakMag:
max(0, |C_t − clamp(C_t, L_W, H_W)|) / ATR_N

ReentryCount:
# de transiciones outside→inside en ventana N

InsideBarsRatio:
(# inside bars en N) / N

SwingCount:
# pivots confirmados (high + low) en W
(pivots usan solo info hasta t−1)

Pendientes (opcionales):
ERSlope, RangeSlope

================================================================
Bootstrap inicial (NO verdad de mercado)
================================================================

El bootstrap existe solo para arrancar.
NO se optimiza por accuracy.

Regla crítica:
Las variables usadas para definir el bootstrap NO deben ser reutilizadas
como features core del modelo sin modificación.

Política explícita:
- ER y NetMove son métricas diagnósticas.
- Si se usan en bootstrap, NO se usan como features core.
- Si se incluyen como features, deben salir del bootstrap
  o degradarse explícitamente.

Prioridad conservadora:
TRANSICIÓN > TENDENCIA > BALANCE

================================================================
Modelo principal — StateEngine (H1)
================================================================

Modelo:
- LightGBM multiclass (BALANCE / TRANSICIÓN / TENDENCIA)

Outputs:
- state_hat
- margin = P(top1) − P(top2)
- probas (solo reporting)

Por defecto:
- No se asume calibración.
- El gating usa state_hat + margin.

================================================================
Gating determinista (ALLOW_*)
================================================================

ALLOW no se entrena.

Ejemplos:
- ALLOW_trend_pullback: state_hat == TENDENCIA y margin ≥ 0.15
- ALLOW_trend_continuation: state_hat == TENDENCIA y margin ≥ 0.15
- ALLOW_balance_fade: state_hat == BALANCE y margin ≥ 0.10
- ALLOW_transition_failure:
  state_hat == TRANSICIÓN
  margin ≥ 0.10
  BreakMag ≥ 0.25
  ReentryCount ≥ 1

Regla explícita:
Si state_hat == TRANSICIÓN → swing direccional prohibido por defecto.

================================================================
Event Scorer (M5) — Edge, no señal
================================================================

Rol:
Medir edge condicionado por:
- evento M5
- contexto H1 (state_hat, margin, ALLOW_*)

Output:
- edge_score ∈ [0,1]

El Event Scorer:
- NO genera señales
- NO ejecuta trades
- NO define SL/TP

================================================================
Puente H1 → M5 (causal)
================================================================

- El contexto H1 se aplica a M5 usando SOLO el último H1 CERRADO.
- Implementación: shift(1) + merge_asof(backward).
- Está prohibido usar la vela H1 en formación.

================================================================
Backtesting
================================================================

Backtesting determinista (M5):
- Entry: next_open
- SL/TP evaluado con OHLC
- Si SL y TP ocurren en la misma vela → SL primero (conservador)
- Fees y slippage configurables
- allow_overlap = False por defecto
- Walk-forward mensual

================================================================
Resultado esperado
================================================================

Un sistema que:
- Clasifica estado con robustez.
- Habilita/prohíbe familias explícitamente.
- Reduce errores estructurales.
- Prioriza supervivencia y disciplina sobre actividad.

================================================================
Notas operativas (Event Scorer M5)
================================================================

- El etiquetado del Event Scorer usa triple-barrier con `r_outcome` continuo.
  La etiqueta binaria deriva de `r_outcome` con umbral configurable.
- La evaluación enfatiza ranking (precision@K / lift@K) por familia y por bins de `margin_H1`.
- El scorer opera por familia (`EventScorerBundle`), manteniendo telemetría; no genera señales.
- Los scripts de entrenamiento/backtest usan por defecto `state_engine/models` como base de modelos.
