# Phase D → Phase E Context Contract
1. Phase D builds a single canonical context bundle (ctx_df + df_score_ctx).
2. ALLOW_* columns are materialized only in Phase D (gating policy).
3. Every allow_context_filters.*.enabled:true must exist in ctx_df and be binary.
4. Missing required context columns raise errors in Phase D (no silent fallbacks).
5. Phase E consumes df_score_ctx as-is: no allow recomputation or skipping.
6. required_allow_by_family defines the only allow_id allowed in Phase E.
7. Events exist iff ctx[required_allow]==1 for their family.
8. Phase E validates the contract at startup and logs allow coverage.
9. Contract validation lives in state_engine/pipeline_phase_d.py and train_event_scorer.py.
