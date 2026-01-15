# AGENTS.md — State Engine + Quality Layer (descriptivo-first)

## Propósito (actualizado – Fase C)

Este repositorio implementa un **State Engine** cuyo rol es
**clasificar el estado estructural del mercado** de forma robusta, causal y generalizable.

El sistema se organiza en **capas conceptualmente separadas**:

### 1. State Layer (existente, estable)
- Clasifica el mercado en:
  - `BALANCE`
  - `TRANSITION`
  - `TREND`
- No predice dirección, retorno ni timing.
- No se redefine ni se optimiza en Fase C.

### 2. Quality Layer (nueva – Fase C)
- Capa **estrictamente descriptiva**, condicionada al estado base.
- Caracteriza la **calidad interna del régimen**:
  - estabilidad
  - coherencia
  - fricción
  - degradación
- **NO genera señales**
- **NO habilita ni prohíbe trades**
- **NO se valida con métricas económicas**
- Su objetivo es **reducir incertidumbre contextual**, no crear edge.

### 3. Capas operativas (fuera de scope de Fase C)
- Gating (`ALLOW_*`)
- Event Scorer (M5)
- Ejecución, SL/TP, backtesting, performance

**Fase C se enfoca exclusivamente en la Quality Layer.**

---

## Principios epistemológicos — Fase C (NO negociables)

- Las Quality Labels son:
  - descriptivas, no predictivas
  - humanas y visualizables
  - condicionadas al estado base
- Ninguna Quality Label se valida usando:
  - EV
  - PnL
  - winrate
  - payoff
  - drawdown
- Se prefiere:
  - falsos negativos > falsos positivos
  - no clasificar > clasificar mal
- “No clasificado” es un output válido.
- Si no hay evidencia estructural clara, se declara explícitamente.
- La estabilidad temporal y la coherencia lógica
  son más importantes que la cobertura.
- La Quality Layer **no rescata estados malos**
  ni fuerza interpretaciones.

---

## Arquitectura del repositorio

configs/
symbols/
_template.yaml
XAUUSD.yaml

scripts/
train_state_engine.py
train_event_scorer.py
run_pipeline_backtest.py
watchdog_state_engine.py
...

state_engine/
pipeline.py
features.py
labels.py
model.py
gating.py
scoring.py
session.py
mt5_connector.py
...

tests/
test_features.py
test_events.py
test_config_loader.py
...

**Nota**  
La infraestructura se reutiliza.  
La lógica conceptual previa no se asume válida para Fase C.

---

## State Engine (State Layer)

### Definición general
- **Timeframe:** H1 (o H2 según símbolo, ya validado en Fases A/B)
- **Ventanas fijas (ejemplo):**
  - `W`: contexto
  - `N`: reciente
- El estado se define por **comportamiento agregado**, no por patrones aislados.

### Estados
- `BALANCE`: rotación y aceptación bilateral.
- `TRANSITION`: intento de cambio con aceptación incompleta o fallo.
- `TREND`: migración sostenida con aceptación.

### Variables PA-first (≤ t)
- NetMove (diagnóstico)
- Path
- Efficiency Ratio (ER)
- Range
- Close Location
- Break Magnitude
- Reentry Count
- Inside Bars Ratio
- Swing Count

**Importante**
- El State Engine NO se especializa por símbolo en Fase C.
- Su semántica se mantiene universal.

---

## Quality Layer (Fase C)

### Rol
Describir la **calidad interna del estado**, sin inferir outcome.

Ejemplos de intención (no exhaustivos):
- Fortaleza vs debilidad
- Expansión vs compresión
- Continuidad vs fricción
- Aceptación vs rechazo

### Propiedades requeridas de una Quality Label
Una Quality Label es válida solo si cumple:
- coherencia lógica con el estado base
- distribución no degenerada
- persistencia temporal razonable
- estabilidad por splits temporales
- independencia total del resultado económico

Si una label es “bonita” pero inestable → se descarta.  
Si es rara pero clara → se conserva.

---

## Configuración por símbolo — alcance real

La configuración por símbolo en Fase C existe para:
- ajustar parámetros descriptivos
- definir umbrales estructurales
- adaptar normalizaciones o escalas

La configuración por símbolo **NO existe** para:
- optimizar performance
- redefinir estados
- forzar cobertura de labels
- introducir lógica operativa

La especialización por símbolo:
- es **paramétrica**
- es **descriptiva**
- **no reentrena** el State Engine

---

## ALLOWs (estado actual — legado)

Los `ALLOW_*` existentes pertenecen a una fase previa del proyecto.

- Son reglas genéricas cross-símbolo.
- No forman parte del foco de Fase C.
- No deben:
  - extenderse
  - especializarse
  - optimizarse
durante esta fase.

En Fase C:
- Los ALLOWs se consideran **artefactos heredados**.
- Pueden ser ignorados o desactivados conceptualmente.
- Su rediseño (si ocurre) es posterior
  a la validación completa de la Quality Layer.

---

## Event Scorer (fuera de scope de Fase C)

- Opera en M5.
- Mide edge relativo condicionado por contexto H1.
- Produce `edge_score` como telemetría.
- No genera señales ni decisiones.

Cualquier modificación al Event Scorer
queda explícitamente fuera de Fase C.

---

## Fuera de scope — Fase C

- Optimización por performance.
- Nuevos ALLOWs.
- Reglas de entrada/salida.
- Ajustes al Event Scorer.
- Cambios en ejecución o SL/TP.
- Automatización de decisiones de trading.

Cualquier propuesta que cruce estos límites
debe considerarse **inválida** en esta fase.

---

## Resultado esperado de Fase C

Un sistema que:
- Clasifica estados de forma robusta (ya logrado).
- Describe la **calidad** de esos estados sin sesgo económico.
- Reduce incertidumbre contextual.
- Tolera explícitamente el “no sé”.
- Protege la integridad epistemológica del sistema.
