# LPulse — v1 Architecture
**Hamza (DE) + Hugo (MLE)**
_June 2026_

---

## Changelog

| Version | Date | Summary |
|---|---|---|
| v0 | 2026-06-12 | Architecture doc drafted. Feature engineering scope defined. Role split agreed. |
| v1 | 2026-06-12 | Phases 1–5 shipped. Backlog items 1, 4, 5 complete. Local pandas pipeline validated on CELO pool (108k positions, 228k sequence rows, 41 tests passing). Ingestion gap documented — LPulse reads from FDP Postgres; ingestion module deferred to next phase. V4 pools (Monad) and multi-pool ingestion scoped but not started. |

---

## What LPulse is

A multi-chain LP behavioral analysis system. It ingests on-chain events from Uniswap V3-compatible concentrated liquidity pools (Uniswap V3, HyperSwap, and any other CLAMM), enriches them with Merkl incentive campaign data, and runs survival analysis to answer one question:

**Do incentive campaigns retain LPs, or just attract mercenary liquidity that exits when rewards end?**

The answer is delivered in two layers:
- **Statistical layer (KM + Cox)** — interpretable, explainable to protocol teams without ML background. The KM curve *is* the pool health metric.
- **DL layer (DeepSurv → DeepHit → DRSA)** — learns LP behavioral archetypes and predicts exit probability from event sequences. Audience: campaign designers and protocol researchers who want leading indicators, not just retrospective curves.

Neither layer is a dashboard for end-users. The primary consumer is a **protocol team or campaign designer** asking: *is our incentive structure attracting sticky liquidity or mercenaries, and what are the early warning signs?*

---

## Pool selection strategy

Target pools must satisfy all three:
1. **V3-compatible CLAMM** — identical event ABI (Mint/Burn/Swap/Collect/Initialize). Confirmed for Uniswap V3 and HyperSwap (V3 fork on HyperEVM). Any Merkl CLAMM opportunity qualifies.
2. **Has Merkl campaign history** — required for the treated/control split. Prefer pools with at least one ended campaign (full lifecycle observable) + optionally one live campaign (censored positions = real right-censoring signal).
3. **Sufficient LP count** — target ≥ 500 observed exits per pool. HyperSwap WHYPE/USDC has 1,707 campaign participants → ~3,500–7,000 total positions estimated. One such pool is enough for DeepSurv; 10–50 pools justify Spark and make the DRSA generalizable.

Start with: 1 pool (CELO/USD₮/WETH, already ingested) for Day-1 DeepSurv smoke test. Expand to 10–50 pools for full training run.

---

## System architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  INGESTION                                                       │
│                                                                  │
│  Merkl API ──► pool list (chain, address, campaign dates)       │
│  HyperSync API (per chain) ──► raw event logs + block headers   │
│       │                                                          │
│       ▼                                                          │
│  GCS: gs://lpulse-raw/                                          │
│       pool_address={addr}/date={YYYY-MM-DD}/                    │
│       ├── logs.parquet                                           │
│       └── blocks.parquet                                         │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│  FEATURE ENGINEERING  [Hamza]                                   │
│                                                                  │
│  Dataproc (PySpark) — one Spark job, N pools in parallel        │
│  Each pool is an independent partition → embarrassingly parallel │
│                                                                  │
│  Per pool:                                                       │
│  ├── decode_events()      → typed Mint/Burn/Swap/Collect DFs    │
│  ├── verify_lp_exit()     → lp_summary (duration, status,       │
│  │                           censoring, position_id)            │
│  ├── tvl() + Chainlink    → tvl_usd, delta_tvl_pct              │
│  ├── collected_fees()     → fees per LP position (not pool)     │
│  ├── exit_type()          → voluntary_exit / range_exit /       │
│  │                           censored                           │
│  └── event_sequence()     → long table per position_id          │
│       │                                                          │
│       ▼                                                          │
│  BigQuery: lpulse_features (dataset)                            │
│  ├── lp_features           (one row per position_id)            │
│  ├── lp_survival_labels    (duration, status, exit_type)        │
│  ├── lp_event_sequences    (long table, keyed position_id)      │
│  └── lp_training_view      (joined view for Hugo's jobs)        │
└─────────────────────────────────────────────────────────────────┘
          │
          ├──────────────────────────────────────────────────────┐
          │                                                       │
          ▼                                                       ▼
┌─────────────────────┐                    ┌──────────────────────────────┐
│  STATISTICAL LAYER  │                    │  DL TRAINING  [Hugo]         │
│  [Hamza]            │                    │                              │
│                     │                    │  Vertex AI Training Job      │
│  lifelines (Python) │                    │  (custom PyTorch container)  │
│  ├── KM curves      │                    │  ├── DeepSurv  (Day 1)       │
│  │   (all + segmented                    │  ├── DeepHit   (Day 1–2)     │
│  │    by campaign)  │                    │  └── DRSA/LSTM (Day 2)       │
│  ├── Cox constant   │                    │       │                      │
│  └── Cox time-      │                    │  Vertex AI Experiments       │
│      varying        │                    │  (metric logging)            │
│                     │                    │       │                      │
│  Interpretability   │                    │  Vertex AI Model Registry    │
│  layer — explainable│                    │  (versioned artifacts)       │
│  to non-ML audience │                    └──────────────────────────────┘
└─────────────────────┘                              │
          │                                          │
          └──────────────┬───────────────────────────┘
                         ▼
              ┌──────────────────────┐
              │  SERVING (batch)     │
              │                      │
              │  BQ scheduled query  │
              │  → lp_exit_predictions
              │    (daily refresh)   │
              │                      │
              │  Stretch: Vertex AI  │
              │  Endpoint (online)   │
              └──────────────────────┘
                         │
              ┌──────────────────────┐
              │  ORCHESTRATION       │
              │                      │
              │  Cloud Composer      │
              │  (Airflow)           │
              │  ├── daily feature   │
              │  │   extraction DAG  │
              │  └── weekly retrain  │
              │       trigger        │
              └──────────────────────┘
```

---

## Why Spark (not pandas)

Single pool = 1–10 GB parquet at most. Pandas handles it fine.

The Spark justification is **pool-level parallelism**. Each pool's feature extraction is completely independent — no cross-pool joins, no shared state. Dataproc fans 50–200 pools out as parallel tasks. Processing 100 pools takes ~3 minutes on a 4-worker cluster; sequentially in pandas it would take 100× longer.

Beyond throughput: the ML model trained on 50 pools across different chains, fee tiers, and TVL profiles generalizes. A survival model trained on one pool is a curiosity; one trained on dozens of pools is an actual predictive tool. Spark is what makes that training set honest — it's not an architectural flex.

**Cost:** PySpark feature run across 100 pools on Dataproc ≈ $0.40. Full pipeline end-to-end under $5. Weekly retrain for a month still under $20.

---

## Why Chainlink (not Flare FTSO, not CoinGecko)

- **CoinGecko**: daily granularity only. LP durations are hours to days — daily prices introduce too much error for `tvl_at_entry_usd`.
- **Flare FTSO**: on-chain oracle on the Flare network. Unusable on Celo/Arbitrum/HyperEVM without bridging. Voting rounds are ~90s regardless — worse than Chainlink heartbeat.
- **Chainlink**: feeds deployed as contracts on the same chains as the pools. Emits `AnswerUpdated` events on heartbeat (~1h) or ±0.5% price deviation. HyperSync already fetches contract events — adding the Chainlink aggregator address to the ingestion config is the only change required. `merge_asof` by block order attaches the last known price to each LP event, same pattern already implemented in `tvl()` for `sqrtPriceX96`.

One feed (WETH/USD) unblocks all USD-denominated features for non-stablecoin pools. The pool's own `sqrtPriceX96` handles the cross-rate to the second token.

---

## Statistical models — role and covariates

### Kaplan-Meier (built, running)
Non-parametric survival curve. No covariates. Two cohorts: entered during campaign vs. before. Log-rank test for statistical significance.
**Output:** $S(t)$ — fraction of LPs still active at time $t$.

### Cox Proportional Hazards — constant covariates (wired, commented out)
$$h(t) = h_0(t) \cdot \exp(\beta_1 \cdot \text{campaign\_flag} + \beta_2 \cdot \text{tick\_range\_width} + \beta_3 \cdot \text{fee\_tier})$$
**Output:** hazard ratio per covariate. "LPs who entered during a campaign exit 1.4× faster (or slower)."

### Cox Time-Varying (wired, blocked on oracle)
$$h_i(t) = h_0(t) \cdot \exp(\beta_1 \cdot \text{num\_active\_campaigns}(t) + \beta_2 \cdot \Delta\text{TVL\%}(t) + \beta_3 \cdot \text{apr}(t))$$
Covariates change over the LP's lifetime. The honest model — captures what happens *during* the campaign, not just at entry. Unblocked once Chainlink oracle is integrated.

---

## DL models — role and archetypes

All three answer a different version of the core question:

| Model | What it learns | Output |
|---|---|---|
| DeepSurv | Nonlinear interactions between at-entry features | Hazard score per LP at entry |
| DeepHit | Two competing exit mechanisms (voluntary vs range-exit) | P(exit by type T before time t) |
| DRSA/LSTM | Trajectory of events leading to exit | Real-time exit probability updated as events arrive |

**The behavioral archetypes DeepHit separates:**
- `voluntary_exit` — price was in range at burn time. LP chose to leave. Likely took profits, responded to campaign end, or rebalanced.
- `range_exit` — price had left the LP's tick range before the burn. Position became dead capital. LP burned eventually to reclaim tokens.

The mercenary hypothesis makes a specific prediction: campaign LPs have a higher rate of `voluntary_exit` clustered around campaign end dates. Non-campaign LPs should have a higher baseline rate of `range_exit` (passive, don't rebalance). If the data confirms this, it tells campaign designers that rewards are not building sticky liquidity — they're renting it.

**The DRSA sequence model** goes further: it observes the event trajectory `[Mint → Swap×N → Collect → Burn]` and learns to detect the signature of imminent exit *before* it happens. E.g. fees stop accumulating (price left range) + no rebalancing action = exit within N days. This is a leading indicator, not a lagging one.

---

## Feature table

| Feature | Type | Availability | Blocked on |
|---|---|---|---|
| `tick_range_width` | at_entry | ✅ now | — |
| `fee_tier` | at_entry | ✅ now | — |
| `campaign_flag` | at_entry | ✅ now | — |
| `pool_address` | at_entry | ✅ now | — |
| `tvl_at_entry_usd` | at_entry | ❌ | Chainlink oracle |
| `collected_fees_usd` | realized_over_life | ❌ | oracle + per-LP fee rewrite |
| `time_in_range_pct` | realized_over_life | ❌ | exit_type function |
| `event_count` | realized_over_life | derivable now | — |
| `delta_tvl_pct(t)` | time_varying | ⚠️ noisy | oracle (clean USD anchor) |
| `apr_at_t` | time_varying | ❌ | oracle + per-LP fee rewrite |
| `exit_type` | label | ❌ | exit_type function (new) |

`at_entry` features only go into DeepSurv/DeepHit. `realized_over_life` and `time_varying` features are DRSA/Cox time-varying only — using them in static models leaks outcome information.

---

## Implementation backlog (Hamza, ordered by dependency)

| # | Task | Status | Unblocks |
|---|---|---|---|
| 1 | `position_id` fix (add `first_mint_tx_hash` to key) | ✅ done (Phase 2) | Hugo's Day-1 slice, stable joins |
| 2 | `collected_fees()` per LP position (groupby owner+ticks) | deferred | APR covariate |
| 3 | Chainlink oracle ingestion via HyperSync + `merge_asof` attach | deferred | All USD features, Cox time-varying |
| 4 | `exit_type` classification function in `metrics.py` | ✅ done (Phase 3) | DeepHit, competing risks label |
| 5 | Event sequence table function in `metrics.py` | ✅ done (Phase 4) | DRSA/LSTM |
| 5a | `pipeline.run()` end-to-end + 3 Parquet output files | ✅ done (Phase 5) | Hugo Day-1 load |
| 6 | LPulse `ingestion/` module + `pools_registry.yaml` | next | Removes FDP dependency, enables multi-pool |
| 7 | Multi-pool ingestion (list of chain+address pairs) | blocked on 6 | Spark justification, training set size |
| 8 | PySpark port of feature pipeline | blocked on 7 | Dataproc fan-out |
| 9 | BigQuery output (`lp_features`, `lp_survival_labels`, `lp_event_sequences`) | blocked on 8 | Hugo's training job |
| 10 | Cloud Composer DAG (daily feature run + weekly retrain trigger) | blocked on 9 | Production pipeline |

Items 1–5a are local Python (complete). Item 6 is the next local step before GCP. Items 7–10 are GCP.

### Known constraints (v1)
- **Ingestion gap**: LPulse currently reads from FDP Postgres. No HyperSync indexing inside LPulse. Adding a new pool requires running FDP ingestion separately first.
- **V3 only**: Decoder is V3-specific (topic0 hashes, Mint/Burn/Swap ABI). V4 pools (e.g. Monad UNISWAP_V4) require a separate `ModifyLiquidity` decoder and pool identity changes (`poolId` bytes32 vs address).
- **Pool quality criteria**: target pools with ≥1 ended Merkl campaign and ≥500 observed exits. Live-campaign pools are valid (right-censored) but don't contribute to the post-campaign stickiness comparison.

---

## Training cost (GCP)

| Component | Hardware | Est. time | Cost |
|---|---|---|---|
| DeepSurv | CPU | 2–5 min | ~$0.01 |
| DRSA/LSTM | T4 GPU | 15–30 min | ~$0.15 |
| SurvTRACE (stretch) | T4 GPU | 30–60 min | ~$0.35 |
| PySpark feature run (100 pools) | 4-worker Dataproc | 10–20 min | ~$0.40 |
| **Full pipeline end-to-end** | | | **< $5** |

Weekly retrain for one month: still under $20.

---

## Role split

| Component | Owner |
|---|---|
| HyperSync ingestion → GCS | Hamza |
| Event decoding, metrics, survival labels | Hamza |
| Chainlink oracle integration | Hamza |
| PySpark feature pipeline → BigQuery | Hamza |
| KM + Cox models (statistical layer) | Hamza |
| Cloud Composer DAG | Hamza |
| DeepSurv baseline | Hugo |
| DeepHit competing risks | Hugo |
| DRSA/LSTM headline model | Hugo |
| Vertex AI training jobs + experiment tracking | Hugo |
| Batch serving (BigQuery predictions) | Shared |
| Architecture doc + README | Shared |
