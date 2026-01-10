Eres Codex. Objetivo: extender el sistema actual agregando SOLO una nueva familia operable:
ALLOW_trend_continuation (H1) + E_TREND_CONTINUATION (M5), integrándola a:
- gating H1 (ALLOW_*)
- detección de eventos M5 (candidatos)
- feature set del Event Scorer
- SignalBuilder (señales deterministas)
- backtesting + sweep + UI
Sin romper compatibilidad, sin agregar otras familias, sin inflar reglas. Mantener causalidad y simpleza.

================================================================
SEMÁNTICA INNEGOCIABLE
================================================================
- State Engine y gating operan SOLO en H1.
- Event Scorer opera SOLO en M5. Output = edge_score (información, no acción).
- Backtester consume SOLO señales (Signal), nunca edge_score directo.
- Prohibido leakage: features <= t; labels usan (t+1..t+K).
- Contexto H1 aplicado a M5: usar último H1 CERRADO (shift(1) + merge_asof).
- Señal requiere: allow activo AND evento detectado AND edge_score > threshold AND trigger mecánico.

================================================================
1) AGREGAR ALLOW_trend_continuation EN GATING (H1)
================================================================
En el módulo de gating (donde se definen ALLOW_*), agregar:

ALLOW_trend_continuation:
- Condición base: state_hat == TREND
- y margin >= umbral (usar el mismo umbral de TREND actual si existe, o default 0.15)
- Debe coexistir con ALLOW_trend_pullback; no deshabilitarlo.

Implementación:
- Nueva columna booleana/binaria en df_h1_gating.
- Incluir en logs de gating (% tiempo activo).

================================================================
2) AGREGAR EVENTO E_TREND_CONTINUATION (M5)
================================================================
Crear detector mínimo, causal, barato. No agenda compleja.

Definición M5 del candidato E_TREND_CONTINUATION:
- Solo se evalúa cuando ALLOW_trend_continuation == 1 en esa vela M5.
- Detectar una “pausa/compresión” seguida de “ruptura a favor del momentum”.

Implementación mínima recomendada (proxy robusto):
Inputs: df_m5_ctx con OHLC y, si existe, ATR_short.

1) Determinar dirección/momentum:
   - momentum_sign = sign( close - EMA_fast ) o sign(return_3) (elige una y documenta)
   - side = LONG si momentum_sign > 0 else SHORT si < 0
   - si momentum_sign == 0: no evento

2) Detectar compresión:
   - range_last_n = rolling max(high)-min(low) en n_comp (default 6)
   - compression = range_last_n / ATR_short < comp_thr (default 1.2)  [si ATR no existe, usar rolling range ratio]
   - requiere compression == True

3) Detectar ruptura:
   - breakout LONG: close(t) > rolling_high(n_brk) de las velas previas (exclude t), n_brk default 6
   - breakout SHORT: close(t) < rolling_low(n_brk) previas
   - requiere breakout == True

Si (momentum + compression + breakout) => generar evento:
- family_id = "E_TREND_CONTINUATION"
- side = LONG/SHORT
- time = timestamp t (evento)
- metadata: n_comp, n_brk, comp_thr, momentum proxy usado

Mantener detector simple: no más de ~30-40 líneas + comentarios.

================================================================
3) ETIQUETADO PARA ESTA FAMILIA
================================================================
Usar el mismo esquema proxy ya existente:
- entry_price = next open
- SL_proxy = ATR_short * sl_mult
- TP_proxy = R * SL_proxy
- ventana K

Asegurar que:
- label usa solo futuro (t+1..t+K)
- features solo pasado (<= t)

================================================================
4) FEATURES / SCORER
================================================================
- Incluir esta familia como family_id (one-hot o equivalente).
- No agregar features nuevas salvo que falte ATR_short o EMA_fast:
  - Si ATR_short no existe, calcular ATR_14 en M5 de forma causal.
  - Si EMA_fast no existe y la usas para momentum, calcular EMA_9 (o similar) en M5.

El scorer ya existente debe entrenar e inferir con esta familia incluida automáticamente (sin ifs especiales).

================================================================
5) SIGNAL BUILDER (CONVERSIÓN A SEÑAL DETERMINISTA)
================================================================
Agregar soporte para señales de esta familia:
- Si evento E_TREND_CONTINUATION detectado
- y edge_score > threshold (usar threshold global o permitir threshold por familia)
- y trigger mecánico simple:

Trigger mecánico propuesto:
- LONG: entry = next_open (ya definido)
- SHORT: entry = next_open
(No agregar retests ni confirmaciones extra; el evento ya incluye breakout.)

SL/TP:
- usar SL_proxy y TP_proxy del mismo esquema.

Exportar signals.csv incluyendo family_id.

================================================================
6) BACKTEST + SWEEP + UI
================================================================
- Asegurar que backtest incluye esta familia en métricas por family_id.
- Incluir en report:
  [EVENTS] conteo E_TREND_CONTINUATION
  [SIGNALS] conteo de señales de esa familia
  [BACKTEST] métricas por familia

En sweep:
- permitir threshold_edge global o por familia.
- Si hay thresholds por familia, agregar E_TREND_CONTINUATION al dict.

================================================================
7) SANITY CHECKS ESPECÍFICOS
================================================================
- Verificar que E_TREND_CONTINUATION solo aparece cuando ALLOW_trend_continuation==1.
- Verificar que el breakout usa rolling highs/lows excluyendo t (shift(1)).
- Verificar que no hay NaNs críticos en ATR/EMA.

================================================================
ENTREGABLE
================================================================
- Modificaciones mínimas, sin romper otras familias.
- Un commit claro que añade:
  - ALLOW_trend_continuation (H1)
  - detector E_TREND_CONTINUATION (M5)
  - integración en scorer y signal builder
  - report/backtest actualizado

Implementa ahora.
