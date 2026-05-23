# Deployment lifecycle (V1 spec)

Status: Phase 1 in progress. This doc is the locked V1 design — review feedback from two Codex passes is baked in.

## Problem

Today the registry's `active` status means "passed all promotion gates." It does NOT mean "currently trading on OKX."
The two freqtrade trading containers (`ft-momentum`, `ft-sweep`) hardcode their strategies in `docker-compose.yml` and
have never run anything the strategy factory produced. We've been celebrating "promotions" that don't deploy.

## V1 goal

Make "deployed" actually mean "trading on OKX demo right now." Separate "research-approved" from
"currently-running-a-container" cleanly.

## V1 policy (locked)

```
MAX_DEPLOYED               = 3
STAKE_AMOUNT (per strategy) = 100 USDT
max_open_trades (per strategy) = 3
Maximum allocated exposure  ≈ 900 USDT  (MAX_DEPLOYED × STAKE × max_open_trades)
```

### Eligibility (hard filters, all must hold)

| Filter | Rationale |
|--------|-----------|
| `total_trades >= 20` | statistical significance |
| `profit_total_pct > 0` | basic edge |
| `sharpe > 0` | risk-adjusted edge |
| worst `max_drawdown_pct` across validation bundle `< 15%` | avoid fragile strategies |
| `last_backtest_at` within 30 days | recent run |
| `backtest_data_end_at` within 30 days | recent **data** (a recent re-run on stale candles doesn't count) |
| correlation `< 0.7` with already-selected | diversification (enforced greedily, see Selection) |

### Selection (greedy with correlation skip)

1. Eligible = strategies passing all hard filters.
2. Sort eligible by Sharpe descending.
3. `selected = []`
4. For each candidate in sorted order:
   - If `max(corr(candidate, s) for s in selected) >= 0.7`: skip.
   - Else: append to `selected`.
   - If `len(selected) == MAX_DEPLOYED`: stop.
5. Deploy `selected` (typically 3).

### Replacement (hysteresis — both conditions must hold to swap)

```
challenger_sharpe >= current_sharpe * 1.20
AND
challenger_sharpe >= current_sharpe + 0.10
```

Why both: relative threshold (×1.20) is too easy when current sharpe is tiny (0.05 × 1.20 = 0.06, basically the same).
Absolute floor (+0.10) prevents churn around weak scores.

### Cooldowns

```
DEPLOYMENT_MIN_DURATION_HOURS    = 24   (deployed for at least this long before eligible for eviction)
DEPLOYMENT_COOLDOWN_HOURS        = 12   (after being stopped, wait at least this long before redeploying)
```

Risk-stopped strategies additionally get:
- `deployment_status = stopped`
- `deployment_blocked_until = now + cooldown`
- `last_deployment_error = <reason>`

This prevents the reconciler from instantly redeploying a strategy that just hit a risk limit.

## Schema

Additive to existing tables — `status` is preserved as a compatibility shim through Phase 4, then deprecated.

```sql
ALTER TABLE strategies ADD COLUMN research_status TEXT;
   -- candidate / approved / rejected / retired
ALTER TABLE strategies ADD COLUMN deployment_status TEXT;
   -- not_deployed / deploying / deployed / stopping / stopped / failed
ALTER TABLE strategies ADD COLUMN deployed_at TEXT;
ALTER TABLE strategies ADD COLUMN last_deployment_error TEXT;
ALTER TABLE strategies ADD COLUMN deployment_blocked_until TEXT;

ALTER TABLE backtest_results ADD COLUMN backtest_data_end_at TEXT;
   -- the END of the data the backtest ran on, NOT when the backtest was invoked
```

Backfill on migration: existing `status='active'` rows become
`research_status='approved'`, `deployment_status='not_deployed'`. They become candidates for the reconciler to deploy
once it starts acting in Phase 3.

## Container topology

One Freqtrade container per deployed strategy. Identifiable by Docker labels:

```
first_duck.role                  = "deployed-strategy"
first_duck.strategy_id           = "<integer>"
first_duck.strategy_name         = "<class-name>"
first_duck.deployment_generation = "<integer>"   # bumps on every redeploy
```

**Reconciler will only ever start, stop, or remove containers carrying `first_duck.role=deployed-strategy`.** This
label scope is the primary safety mechanism against accidental container destruction. The orchestrator itself,
`ft-monitor`, and the (still-running, during migration) `ft-momentum`/`ft-sweep` carry no such label and are invisible
to the reconciler.

### Networking

- Internal Docker network only — **no host port mappings**.
- Orchestrator talks to deployed containers by container name (`http://ft-deployed-<slug>:8080`).
- This avoids the port-collision problem that would hit immediately if 3 containers all tried to bind 8080 on the host.

### Per-strategy paths

```
container name: ft-deployed-<strategy-name-kebab>
db url:         sqlite:////freqtrade/user_data/tradesv3-deployed-<slug>.sqlite
logfile:        /freqtrade/user_data/logs/ft-deployed-<slug>.log
config:         /tmp/config-deployed-<slug>.json   (rendered at startup, chmod 0600, not host-mounted)
```

## Why hybrid Compose + Docker SDK

- **Compose** owns static infrastructure: orchestrator, ft-monitor, the shared network, mounted user_data, and
  (during migration) the legacy ft-momentum / ft-sweep services. Anything human ops needs to inspect with
  `docker compose ps` lives here.
- **Docker SDK** (called from the orchestrator container, via the already-mounted docker socket) owns the
  dynamic `ft-deployed-*` containers. Creating/destroying these via Compose would require either pre-declaring
  every possible strategy as a service (impossible, names are LLM-generated) or generating YAML at runtime
  (operationally awful).

## Rollout flags

| Flag | Default | Phases |
|------|---------|--------|
| `RECONCILER_ACTING` | `false` | Phases 1-2: reconciler exists but observes only. Phase 3+: turns on |
| `LEGACY_CONTAINERS_ENABLED` | `true` | Phases 1-3.5: legacy ft-momentum / ft-sweep keep running. Phase 4: off |
| `STRICT_PROMOTION_GATES` | `true` | already shipped in PR #39 |

## Phased rollout

| Phase | What | Behavior change | Reversibility |
|-------|------|-----------------|---------------|
| 1 (this PR) | Design doc + schema migration + config template + Docker SDK wrapper (dry-run capable, label-scoped) + tests | None | Pure addition |
| 2 | Reconciler in **observe-only** mode: cron logs intended start/stop actions, writes drift table | None | Flip `RECONCILER_ACTING=true` to graduate |
| 3 | Single-strategy live shakedown — manually deploy ONE strategy (by ID) via SDK while legacy keeps running | One new container starts trading | Stop the one container; `RECONCILER_ACTING=false` |
| 3.5 | Minimal per-strategy risk stop + pool exposure cap + drift alarm | Additional safety | — |
| 4 | Full reconciler active + retire legacy ft-momentum / ft-sweep | Major cutover | `LEGACY_CONTAINERS_ENABLED=true` brings legacy back; `RECONCILER_ACTING=false` freezes reconciler |
| 5 | Full per-strategy drawdown attribution + portfolio risk model + dashboard | Observability | — |

Each phase is a separate PR. Each phase is reversible via env-flag flip or, for schema, additive rollback.

## What Phase 1 contains (this PR)

- `docs/deployment-lifecycle.md` — this document
- `user_data/scripts/strategy_registry.py` — schema migration + helpers (`get_deployment_eligible`, `compute_selection`)
- `user_data/configs/config-deployed.json.template` — generic per-strategy config with `${STRATEGY_NAME}`-style placeholders
- `user_data/scripts/deployment_manager.py` — Docker SDK wrapper with `dry_run` mode and label-scoped operations
- Tests for everything

No production effect. The reconciler doesn't exist yet (Phase 2). The SDK wrapper isn't called from anywhere except
tests. Schema changes are additive; existing code paths read `status` as before.

## What's explicitly NOT in V1

- Sharpe-weighted capital allocation (V2 candidate)
- Multi-strategy-per-container (Freqtrade doesn't support live)
- Cross-strategy capital pooling beyond the per-strategy stake
- Per-strategy hyperopt re-runs before redeployment
- A weighted deployment-score formula (we'd be inventing weights without live data)
- Live host-side REST port mapping per deployed container

These can be added later without changing the lifecycle architecture.
