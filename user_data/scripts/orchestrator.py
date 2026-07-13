"""
Strategy Factory Orchestrator

APScheduler-based job manager that coordinates all components:
- Daily: fetch macro data, classify regime, update strategy instance configs
- Weekly: run reflector agent, generate new strategies, backtest candidates
- Continuous: monitor strategy instances, enforce risk limits

The orchestrator talks to Freqtrade instances via their REST API and manages
Docker containers for starting/stopping strategies based on regime.
"""

import json
import logging
import os
import subprocess

# Notification helper (graceful if not configured)
try:
    from notifier import (
        notify_regime_change, notify_kill_switch, notify_strategy_promoted,
        notify_factory_summary, notify_instance_down, notify_reflector_summary,
    )
except ImportError:
    # Stub out if notifier not available
    def notify_regime_change(*a, **kw): pass
    def notify_kill_switch(*a, **kw): pass
    def notify_strategy_promoted(*a, **kw): pass
    def notify_factory_summary(*a, **kw): pass
    def notify_instance_down(*a, **kw): pass
    def notify_reflector_summary(*a, **kw): pass
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.events import EVENT_JOB_ERROR

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# In Docker: script is at /app/scripts/, user_data is mounted at /app/user_data/
# Locally: script is at user_data/scripts/, user_data is the parent
_script_dir = Path(__file__).resolve().parent
_candidate_base = _script_dir.parent  # user_data/ when running locally
if (_candidate_base / "data").exists():
    BASE_DIR = _candidate_base
elif Path("/app/user_data").exists():
    BASE_DIR = Path("/app/user_data")
else:
    BASE_DIR = _candidate_base
DATA_DIR = BASE_DIR / "data"
REGIME_STATE_FILE = BASE_DIR / "data" / "regime_state.json"
RISK_STATE_FILE = BASE_DIR / "data" / "risk_state.json"

# Freqtrade instance endpoints (container names resolve via Docker networking).
# REST API credentials live in env vars rather than committed source — the
# values here must match what the corresponding Freqtrade containers see
# via their config templates (FT_{NAME}_API_PASSWORD). Defaults preserve
# the prior literals so a developer can still run the orchestrator without
# wiring env vars first, but production deploys must override.
def _instance_password(env_var: str, default: str) -> str:
    return os.environ.get(env_var, default)


INSTANCES = {
    "sweep": {
        "url": "http://ft-sweep:8080",
        "username": os.environ.get("FT_SWEEP_API_USERNAME", "freqtrader"),
        "password": _instance_password("FT_SWEEP_API_PASSWORD", "CHANGE_ME_sweep_password"),
        "strategy": "LiquiditySweepStrategy",
        "regimes": ["ranging"],  # Active in these regimes
    },
    "momentum": {
        "url": "http://ft-momentum:8080",
        "username": os.environ.get("FT_MOMENTUM_API_USERNAME", "freqtrader"),
        "password": _instance_password("FT_MOMENTUM_API_PASSWORD", "CHANGE_ME_momentum_password"),
        "strategy": "MomentumTrendStrategy",
        "regimes": ["trending", "breakout"],  # Active in these regimes
    },
}

# Risk limits (NOT tunable by LLM, human-set only)
RISK_LIMITS = {
    "max_drawdown_daily_pct": 3.0,    # -3% daily max
    "max_drawdown_total_pct": 10.0,   # -10% total kill switch
    "crisis_regime_action": "stop_all",  # Stop all trading in crisis
}

MODE = os.environ.get("ORCHESTRATOR_MODE", "dry-run")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log_dir = BASE_DIR / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_dir / "orchestrator.log"),
    ],
)
log = logging.getLogger("orchestrator")


# ---------------------------------------------------------------------------
# Freqtrade API Client
# ---------------------------------------------------------------------------
class FreqtradeClient:
    """Simple client for Freqtrade REST API."""

    def __init__(self, base_url: str, username: str, password: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self._token = None
        self._username = username
        self._password = password

    def _login(self):
        try:
            resp = self.session.post(
                f"{self.base_url}/api/v1/token/login",
                auth=(self._username, self._password),
                timeout=10,
            )
            resp.raise_for_status()
            self._token = resp.json().get("access_token")
            self.session.headers["Authorization"] = f"Bearer {self._token}"
        except Exception as e:
            log.warning(f"Login failed for {self.base_url}: {e}")

    def _request(self, method: str, endpoint: str, **kwargs):
        if not self._token:
            self._login()
        try:
            resp = self.session.request(
                method, f"{self.base_url}/api/v1/{endpoint}", timeout=15, **kwargs
            )
            if resp.status_code == 401:
                self._login()
                resp = self.session.request(
                    method, f"{self.base_url}/api/v1/{endpoint}", timeout=15, **kwargs
                )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            log.warning(f"API request failed: {method} {endpoint} -> {e}")
            return None

    def get_status(self):
        return self._request("GET", "status")

    def get_profit(self):
        return self._request("GET", "profit")

    def get_balance(self):
        return self._request("GET", "balance")

    def start(self):
        return self._request("POST", "start")

    def stop(self):
        return self._request("POST", "stop")

    def force_exit_all(self):
        """Force exit all open trades."""
        status = self.get_status()
        if status:
            for trade in status:
                trade_id = trade.get("trade_id")
                if trade_id:
                    self._request("POST", "forceexit", json={"tradeid": trade_id})

    def is_alive(self) -> bool:
        try:
            resp = self.session.get(f"{self.base_url}/api/v1/ping", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Regime State
# ---------------------------------------------------------------------------
def load_regime_state() -> dict:
    if REGIME_STATE_FILE.exists():
        with open(REGIME_STATE_FILE) as f:
            return json.load(f)
    return {"regime": "ranging", "confidence": 0.5, "source": "default", "timestamp": None}


def save_regime_state(state: dict):
    REGIME_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    state["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(REGIME_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)
    log.info(f"Regime state saved: {state['regime']} (confidence: {state['confidence']})")


# ---------------------------------------------------------------------------
# Risk State
# ---------------------------------------------------------------------------
def load_risk_state() -> dict:
    if RISK_STATE_FILE.exists():
        with open(RISK_STATE_FILE) as f:
            return json.load(f)
    return {
        "kill_switch_active": False,
        "daily_pnl": 0.0,
        "total_pnl": 0.0,
        "last_check": None,
    }


def save_risk_state(state: dict):
    RISK_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    state["last_check"] = datetime.now(timezone.utc).isoformat()
    with open(RISK_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

def job_fetch_macro_data():
    """Daily: fetch macro (Yahoo) + BTC perp (Binance Futures) + ETH/BTC (Binance spot)."""
    log.info("=== Job: Fetch macro data ===")
    macro_script = BASE_DIR / "scripts" / "fetch_extra_data.py"
    perp_script = BASE_DIR / "scripts" / "fetch_perp_data.py"
    eth_btc_script = BASE_DIR / "scripts" / "fetch_eth_btc.py"
    # Run each fetcher independently so one outage doesn't block the others.
    # Errors are logged but don't crash the orchestrator.
    for label, script in (
        ("Macro (Yahoo)", macro_script),
        ("Perp (Binance)", perp_script),
        ("ETH/BTC (Binance spot)", eth_btc_script),
    ):
        try:
            subprocess.run([sys.executable, str(script)], check=True, timeout=120)
            log.info(f"{label} fetch completed.")
        except Exception as e:
            log.error(f"{label} fetch failed: {e}")


def job_fetch_ohlcv():
    """Weekly: refresh OKX OHLCV feathers used by mini-/full-backtests.

    Walk-forward gate defaults to 3 splits × 60 days = 180 days; we pull 200
    days as a safety buffer. Freqtrade incrementally appends — re-running is
    idempotent and only fetches what's new.

    Runs Saturday 19:30 UTC, 30 min before generation, so the Saturday
    mini-backtests and Sunday full backtests both see fresh data. Without
    this, backtest windows silently truncate to whatever the feathers cover
    (last seen: 5 weeks stale, all generated strategies returned 0 trades).
    """
    log.info("=== Job: Fetch OKX OHLCV data ===")
    try:
        import json
        with open(BASE_DIR / "config.json") as fh:
            cfg = json.load(fh)
        pairs = cfg.get("exchange", {}).get("pair_whitelist", [])
        timeframe = cfg.get("timeframe", "1h")
    except Exception as e:
        log.error(f"OHLCV fetch: could not read pair_whitelist from config: {e}")
        return
    if not pairs:
        log.warning("OHLCV fetch: no pairs configured, skipping")
        return

    host_project_dir = os.environ.get("HOST_PROJECT_DIR", str(BASE_DIR.parent))
    compose_file = str(BASE_DIR.parent / "docker-compose.yml")
    cmd = [
        "docker", "compose", "-f", compose_file,
        "--project-directory", host_project_dir,
        "--profile", "backtest", "run", "--rm",
        "freqtrade-backtest", "download-data",
        "--config", "/freqtrade/user_data/config.json",
        "--pairs", *pairs,
        "--timeframes", timeframe,
        "--days", "200",
    ]
    log.info(f"OHLCV fetch: {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode == 0:
            log.info(f"OHLCV fetch completed for {len(pairs)} pairs.")
        else:
            log.error(f"OHLCV fetch exited {proc.returncode}: {proc.stderr[-500:]}")
    except subprocess.TimeoutExpired:
        log.error("OHLCV fetch timed out after 600s")
    except Exception as e:
        log.error(f"OHLCV fetch failed: {e}")


def job_classify_regime():
    """Daily: classify current market regime using indicators.

    Phase 1: indicator-based only.
    Phase 2 will add LLM layer on top.
    """
    log.info("=== Job: Classify regime ===")
    try:
        # Load the most recent OHLCV data for BTC (primary signal)
        import pandas as pd
        import pandas_ta as ta

        # Try to load BTC 1h data
        btc_file = DATA_DIR / "okx" / "BTC_USDT-1h.feather"
        if not btc_file.exists():
            btc_file = DATA_DIR / "okx" / "futures" / "BTC_USDT_USDT-1h-futures.feather"
        if not btc_file.exists():
            btc_file = DATA_DIR / "binance" / "BTC_USDT-1h.feather"

        if not btc_file.exists():
            log.warning("No BTC data found for regime classification. Using default.")
            save_regime_state({"regime": "ranging", "confidence": 0.3, "source": "no-data"})
            return

        df = pd.read_feather(btc_file)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], utc=True)

        # Use the last 200 candles for regime detection
        df = df.tail(250).reset_index(drop=True)

        # Import and run regime detector
        sys.path.insert(0, str(BASE_DIR))
        from indicators.regime_detector import add_regime_detection

        df = add_regime_detection(df)

        # Take the most recent regime
        latest = df.iloc[-1]
        regime = latest["regime"]
        confidence = float(latest["regime_confidence"])

        save_regime_state({
            "regime": regime,
            "confidence": confidence,
            "source": "indicator",
            "adx": float(latest.get("regime_adx", 0)),
            "vol_pct": float(latest.get("regime_vol_pct", 0)),
        })

    except Exception as e:
        log.error(f"Regime classification failed: {e}", exc_info=True)
        save_regime_state({"regime": "ranging", "confidence": 0.3, "source": "error"})


def job_apply_regime():
    """Daily (after classification): start/stop strategy instances based on regime.

    - Each strategy has a list of regimes it's active in.
    - In 'crisis' regime, stop all trading and force-exit positions.
    - In other regimes, start matching strategies and stop non-matching ones.
    """
    log.info("=== Job: Apply regime to instances ===")
    state = load_regime_state()
    regime = state.get("regime", "ranging")
    risk = load_risk_state()

    if risk.get("kill_switch_active"):
        log.warning("KILL SWITCH ACTIVE. All trading stopped.")
        for name, cfg in INSTANCES.items():
            client = FreqtradeClient(cfg["url"], cfg["username"], cfg["password"])
            if client.is_alive():
                client.stop()
        return

    # Detect regime change and notify
    prev_regime = risk.get("last_regime", "unknown")
    if regime != prev_regime and prev_regime != "unknown":
        notify_regime_change(prev_regime, regime, state.get("confidence", 0), state.get("source", ""))
    risk["last_regime"] = regime
    save_risk_state(risk)

    log.info(f"Current regime: {regime}")

    # Crisis = stop everything
    if regime == "crisis":
        log.warning("CRISIS regime detected. Stopping all instances and force-exiting.")
        for name, cfg in INSTANCES.items():
            client = FreqtradeClient(cfg["url"], cfg["username"], cfg["password"])
            if client.is_alive():
                client.force_exit_all()
                client.stop()
                log.info(f"  {name}: STOPPED (crisis)")
        return

    # Normal regime routing
    for name, cfg in INSTANCES.items():
        client = FreqtradeClient(cfg["url"], cfg["username"], cfg["password"])
        if not client.is_alive():
            log.warning(f"  {name}: NOT REACHABLE")
            continue

        if regime in cfg["regimes"]:
            client.start()
            log.info(f"  {name}: ACTIVE (regime={regime} matches {cfg['regimes']})")
        else:
            # Don't force-exit, just stop new entries. Existing positions managed by strategy.
            client.stop()
            log.info(f"  {name}: PAUSED (regime={regime} not in {cfg['regimes']})")


def job_check_risk():
    """Frequent: check drawdown limits across all instances.

    If daily drawdown exceeds limit or total drawdown hits kill switch,
    stop everything immediately.
    """
    risk = load_risk_state()

    if risk.get("kill_switch_active"):
        return  # Already dead

    total_pnl = 0.0
    for name, cfg in INSTANCES.items():
        client = FreqtradeClient(cfg["url"], cfg["username"], cfg["password"])
        profit = client.get_profit()
        if profit and "profit_all_coin" in profit:
            total_pnl += float(profit["profit_all_coin"])

    # Check total drawdown kill switch
    # Assuming 1000 USDT total capital (sum of both instances)
    total_capital = 1000.0
    drawdown_pct = abs(min(total_pnl, 0)) / total_capital * 100

    if drawdown_pct >= RISK_LIMITS["max_drawdown_total_pct"]:
        log.critical(
            f"KILL SWITCH TRIGGERED! Drawdown: {drawdown_pct:.1f}% >= "
            f"{RISK_LIMITS['max_drawdown_total_pct']}%. Stopping all trading."
        )
        for name, cfg in INSTANCES.items():
            client = FreqtradeClient(cfg["url"], cfg["username"], cfg["password"])
            if client.is_alive():
                client.force_exit_all()
                client.stop()

        risk["kill_switch_active"] = True
        risk["total_pnl"] = total_pnl
        risk["trigger_reason"] = f"Drawdown {drawdown_pct:.1f}%"
        save_risk_state(risk)
        notify_kill_switch(risk["trigger_reason"], total_pnl)
        return

    risk["total_pnl"] = total_pnl
    save_risk_state(risk)


def job_health_check():
    """Frequent: log the status of all instances."""
    for name, cfg in INSTANCES.items():
        client = FreqtradeClient(cfg["url"], cfg["username"], cfg["password"])
        alive = client.is_alive()
        status = "UP" if alive else "DOWN"
        log.info(f"  Health: {name} = {status}")
        if not alive:
            notify_instance_down(name)


def _parse_allowlist(raw: str) -> set[int]:
    """Parse RECONCILER_ALLOWLIST = "127,131" → {127, 131}. Empty / malformed
    entries silently drop so a typo can't accidentally enable action on the
    wrong strategy."""
    out: set[int] = set()
    for token in (raw or "").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.add(int(token))
        except ValueError:
            log.warning(f"  RECONCILER_ALLOWLIST: ignoring non-integer token {token!r}")
    return out


def _build_deployed_env(strategy_name: str, strategy_slug: str) -> dict[str, str]:
    """Env vars the deployed-strategy container needs at startup.

    render_config.py substitutes ${VAR} placeholders in the deployed
    template from these. Any required var that's empty/unset will fail
    the render fast and loud — see render_config.py for the strict check.

    PYTHONPATH=/freqtrade/user_data is non-negotiable: generated
    strategies do `from indicators.external_data import ...` and that
    package lives under the bind-mounted user_data/. Without this env
    var freqtrade dies at strategy-import time with "No module named
    'indicators'" (same setup as ft-momentum / ft-sweep in
    docker-compose). Caught in the Phase 3 shakedown.
    """
    return {
        "PYTHONPATH":               "/freqtrade/user_data",
        "OKX_API_KEY":              os.environ.get("OKX_API_KEY", ""),
        "OKX_API_SECRET":           os.environ.get("OKX_API_SECRET", ""),
        "OKX_API_PASSPHRASE":       os.environ.get("OKX_API_PASSPHRASE", ""),
        "FT_DEPLOYED_JWT_SECRET":   os.environ.get("FT_DEPLOYED_JWT_SECRET", ""),
        "FT_DEPLOYED_API_PASSWORD": os.environ.get("FT_DEPLOYED_API_PASSWORD", ""),
        "STRATEGY_NAME":            strategy_name,
        "STRATEGY_SLUG":            strategy_slug,
    }


def job_reconcile_deployments():
    """Deployment reconciler — see ``docs/deployment-lifecycle.md``.

    Computes:

      desired  =  greedy correlation-aware selection from get_deployment_eligible()
      running  =  containers carrying first_duck.role=deployed-strategy

    and logs the diff (starts + stops) plus persists the full snapshot to
    deployment_drift_log for forensics.

    Acting policy (Phase 3):

      * RECONCILER_ACTING=false (default) → fully observe-only, no Docker
        mutation. Same behavior as Phase 2.
      * RECONCILER_ACTING=true AND RECONCILER_ALLOWLIST="<id>,<id>,..."
        → the reconciler will start/stop containers ONLY for strategies
        whose id appears in the allowlist. Everything outside the
        allowlist is still observed and logged but not actually touched.
      * RECONCILER_ACTING=true AND empty allowlist → still observe-only.
        The empty allowlist is the deliberate safety net for Phase 3
        shakedown — operator must explicitly opt one ID in at a time
        before any container action happens.

    Safe to run on every cron tick (every 5 min).
    """
    log.info("=== Job: Reconcile deployments ===")
    reconciler_acting = os.environ.get(
        "RECONCILER_ACTING", "false"
    ).lower() in ("1", "true", "yes")
    allowlist = _parse_allowlist(os.environ.get("RECONCILER_ALLOWLIST", ""))
    actually_acting = reconciler_acting and bool(allowlist)
    if reconciler_acting and not allowlist:
        log.info("  RECONCILER_ACTING=true but RECONCILER_ALLOWLIST empty — "
                 "shakedown safety net: behaving as observe-only until an id "
                 "is explicitly opted in")
    elif actually_acting:
        log.info(f"  ACTING for strategy ids in allowlist: {sorted(allowlist)}; "
                 f"all other intents stay observe-only")

    try:
        from strategy_registry import (
            get_deployment_eligible, get_currently_deployed, record_drift_log,
            mark_deployment_status,
        )
        from deployment_selection import (
            compute_desired_deployments, DEFAULT_MAX_DEPLOY, DEFAULT_CORR_THRESHOLD,
        )
        from deployment_manager import (
            DeploymentManager, DeployedContainerSpec, container_name_for,
            strategy_slug,
        )

        max_deploy = int(os.environ.get("MAX_DEPLOYED", str(DEFAULT_MAX_DEPLOY)))
        corr_threshold = float(os.environ.get(
            "DEPLOYMENT_CORR_THRESHOLD", str(DEFAULT_CORR_THRESHOLD)))

        eligible = get_deployment_eligible()
        log.info(f"  eligible pool: {len(eligible)} strategies")

        selection = compute_desired_deployments(
            eligible, max_deploy=max_deploy, corr_threshold=corr_threshold,
        )
        desired_rows = selection["desired"]
        skipped_rows = selection["skipped"]

        # `running` is the GROUND TRUTH from docker — what's actually there.
        # `registry_deployed` is what the registry BELIEVES is deployed.
        # The Phase 4 drift alarm compares the two; here we just log both.
        try:
            mgr = DeploymentManager()
            running = mgr.list_deployed()
        except Exception as e:
            log.warning(f"  Docker SDK list_deployed failed: {type(e).__name__}: {e}")
            running = []
        registry_deployed = get_currently_deployed()

        log.info(f"  desired: {[r.get('name') for r in desired_rows]}")
        log.info(f"  running (docker): "
                 f"{[(r['name'], r['status']) for r in running]}")
        log.info(f"  registry says deployed: "
                 f"{[r['name'] for r in registry_deployed]}")
        for s in skipped_rows:
            row = s["row"]
            log.info(f"  [skipped] {row.get('name')!r}: {s['reason']}")

        # Compute intent. Container-name match is the canonical join key —
        # the registry's row.id may differ from what Docker remembers, but
        # the name derives from the strategy class name and is stable.
        running_names = {r["name"] for r in running}
        desired_container_names = {
            container_name_for(r["name"]): r for r in desired_rows
        }
        intended_starts = [
            {"strategy_id": r.get("id"), "name": r.get("name"),
             "container_name": cname, "sharpe": r.get("sharpe")}
            for cname, r in desired_container_names.items()
            if cname not in running_names
        ]
        intended_stops = [
            {"container_name": rn,
             "strategy_name": next(
                 (r["strategy_name"] for r in running if r["name"] == rn), ""),
             "strategy_id": next(
                 (r["strategy_id"] for r in running if r["name"] == rn), -1)}
            for rn in running_names if rn not in desired_container_names
        ]

        if intended_starts:
            log.info(f"  WOULD START ({len(intended_starts)}): "
                     f"{[a['container_name'] for a in intended_starts]}")
        if intended_stops:
            log.info(f"  WOULD STOP  ({len(intended_stops)}): "
                     f"{[a['container_name'] for a in intended_stops]}")
        if not intended_starts and not intended_stops:
            log.info("  reconciler converged: desired == running")

        # Acting path. Only fires when RECONCILER_ACTING=true AND the
        # strategy id is in RECONCILER_ALLOWLIST. Everything else is
        # observe-only — Phase 3 shakedown safety: operator opts each
        # strategy in by ID, one at a time, and watches it for ~24h
        # before adding more.
        if actually_acting:
            host_user_data = os.environ.get("HOST_PROJECT_DIR", "") + "/user_data"
            for start in intended_starts:
                sid = start["strategy_id"]
                if sid not in allowlist:
                    log.info(f"  [skip-action] start {start['container_name']!r} "
                             f"(id={sid}) — not in allowlist")
                    continue
                mark_deployment_status(sid, "deploying")
                try:
                    spec = DeployedContainerSpec(
                        strategy_id=sid,
                        strategy_name=start["name"],
                        deployment_generation=1,  # Phase 5 bumps on redeploy
                        env=_build_deployed_env(
                            start["name"], strategy_slug(start["name"])),
                        volumes={
                            host_user_data: {
                                "bind": "/freqtrade/user_data", "mode": "rw"
                            },
                        },
                    )
                    mgr.start(spec, dry_run=False)
                    mark_deployment_status(sid, "deployed")
                    log.info(f"  STARTED {start['container_name']!r} "
                             f"(strategy_id={sid})")
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"
                    log.error(f"  start FAILED for {start['container_name']!r}: {err}",
                              exc_info=True)
                    # cooldown so the next tick doesn't immediately retry
                    mark_deployment_status(sid, "failed", error=err,
                                           block_for_hours=1.0)

            for stop in intended_stops:
                sid = stop.get("strategy_id", -1)
                if sid not in allowlist:
                    log.info(f"  [skip-action] stop {stop['container_name']!r} "
                             f"(id={sid}) — not in allowlist")
                    continue
                strat_name = stop["strategy_name"] or stop["container_name"].removeprefix("ft-deployed-")
                if sid >= 0:
                    mark_deployment_status(sid, "stopping")
                try:
                    mgr.stop_graceful(strat_name, dry_run=False)
                    mgr.remove(strat_name, dry_run=False)
                    if sid >= 0:
                        mark_deployment_status(sid, "stopped")
                    log.info(f"  STOPPED {stop['container_name']!r}")
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"
                    log.error(f"  stop FAILED for {stop['container_name']!r}: {err}",
                              exc_info=True)
                    if sid >= 0:
                        mark_deployment_status(sid, "failed", error=err)

        record_drift_log(
            desired=[{k: v for k, v in r.items()} for r in desired_rows],
            running=running,
            intended_starts=intended_starts,
            intended_stops=intended_stops,
            skipped_eligible=[
                {"name": s["row"].get("name"),
                 "id": s["row"].get("id"),
                 "reason": s["reason"]}
                for s in skipped_rows
            ],
            reconciler_acting=reconciler_acting,
            notes=("phase-3 acting (allowlist active)" if actually_acting
                   else "observe-only"),
        )

    except Exception as e:
        log.error(f"reconcile job error: {type(e).__name__}: {e}", exc_info=True)


# ---------------------------------------------------------------------------
# Weekly Jobs: Strategy Factory Loop
# ---------------------------------------------------------------------------

def job_generate_strategies():
    """Weekly: generate new candidate strategies via LLM.

    Generates strategies across different regimes, validates them,
    and registers passing candidates in the registry.
    """
    log.info("=== Job: Generate strategies ===")

    if not (os.environ.get("DEEPSEEK_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")):
        log.warning("No LLM API key set (DEEPSEEK_API_KEY or ANTHROPIC_API_KEY). "
                    "Skipping strategy generation.")
        return

    try:
        sys.path.insert(0, str(BASE_DIR / "scripts"))
        from strategy_generator import generate_batch, _format_failure_examples, dedupe_class_name
        from strategy_registry import (
            register_strategy, get_registry_stats, get_strategy_by_name,
            get_recent_failures, load_recent_reflections, get_recent_attributions,
        )
        from trade_attribution import (
            aggregate_attributions_by_bucket, format_aggregate_for_generator,
        )

        # Get current regime for context
        state = load_regime_state()
        context = f"Current market regime: {state.get('regime', 'unknown')} (confidence: {state.get('confidence', 0)})"

        # Get existing strategy results for context
        stats = get_registry_stats()
        existing_results = (
            f"Registry stats: {stats['active']} active, {stats['candidate']} candidates, "
            f"{stats['retired']} retired, {stats['total_backtests']} backtests run."
        )

        # Close the feedback loop: reflector insights + per-regime failure memory
        reflector_insights = load_recent_reflections(n=2)
        log.info(f"  Reflector context: {len(reflector_insights)} chars from latest reflections")

        def failures_for(regime: str) -> str:
            rows = get_recent_failures(k=8, regime=regime)
            log.info(f"  Failure memory for regime={regime}: {len(rows)} prior failures")
            return _format_failure_examples(rows)

        # Build the attribution patterns block per regime, with pool-wide
        # fallback when a regime has fewer than 3 strategies with attribution
        # (a single-strategy aggregate is just that strategy's attribution
        # restated — no cross-strategy signal). The pool-wide rollup
        # always uses the full set of attributed strategies so the LLM still
        # gets evidence-based guidance even on rare regimes.
        all_attributions = get_recent_attributions(n=20, min_trades=10)

        def attribution_for(regime: str) -> str:
            per_regime = aggregate_attributions_by_bucket(all_attributions, regime=regime)
            if per_regime["n_strategies"] >= 3:
                log.info(
                    f"  Attribution patterns for regime={regime}: "
                    f"{per_regime['n_strategies']} strategies, "
                    f"{len(per_regime['top_consistent_winners'])} winners, "
                    f"{len(per_regime['top_consistent_losers'])} losers"
                )
                return format_aggregate_for_generator(per_regime, regime)
            # Fallback to pool-wide
            pool = aggregate_attributions_by_bucket(all_attributions, regime=None)
            log.info(
                f"  Attribution patterns for regime={regime}: pool-wide fallback "
                f"({per_regime['n_strategies']} regime-specific < 3); "
                f"{pool['n_strategies']} pool strategies, "
                f"{len(pool['top_consistent_winners'])} winners, "
                f"{len(pool['top_consistent_losers'])} losers"
            )
            return format_aggregate_for_generator(pool, regime)

        # Phase 6 — iterate the 20-cell coherence matrix instead of cycling
        # 5 strategies through regimes. Each cell is one (archetype, regime)
        # pair the spec validator enforces. Diversity is structural, not
        # prompt-suggested.
        from archetypes import coherence_matrix
        cells = coherence_matrix()
        log.info(f"  Coherence matrix: {len(cells)} (archetype, regime) cells to generate")

        results = generate_batch(
            cells=cells,
            context=context,
            existing_results=existing_results,
            reflector_insights=reflector_insights,
            get_failures_for_regime=failures_for,
            get_attribution_for_regime=attribution_for,
            # R6 — let each candidate iterate up to 2 turns based on its own
            # mini-backtest result. Single-shot generation was producing too
            # many 0-trade strategies even with R3+R5 in place.
            iterative=True,
            max_turns=2,
        )

        # Register successful strategies
        for r in results:
            if r.get("success"):
                filepath = r.get("filepath", "")
                gen_id = r.get("generation_id", "")

                # Extract class name and metadata from the file
                try:
                    import ast
                    source = Path(filepath).read_text()
                    tree = ast.parse(source)
                    class_name = ""
                    thesis = ""
                    target_regime = r.get("target_regime") or "all"
                    archetype = r.get("archetype") or ""
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            class_name = node.name
                            for item in node.body:
                                if isinstance(item, ast.Assign):
                                    for target in item.targets:
                                        if isinstance(target, ast.Name):
                                            if target.id == "STRATEGY_THESIS" and isinstance(item.value, ast.Constant):
                                                thesis = item.value.value
                                            elif target.id == "TARGET_REGIME" and isinstance(item.value, ast.Constant):
                                                target_regime = item.value.value
                                            elif target.id == "STRATEGY_ARCHETYPE" and isinstance(item.value, ast.Constant):
                                                archetype = item.value.value
                            break

                    if class_name:
                        # Avoid UNIQUE collision when LLM reuses a class name
                        # that matches an already-retired strategy.
                        class_name = dedupe_class_name(
                            filepath,
                            class_name,
                            lambda n: get_strategy_by_name(n) is not None,
                        )
                        register_strategy(
                            name=class_name,
                            filepath=str(filepath),
                            thesis=thesis,
                            target_regime=target_regime,
                            generation_id=gen_id,
                            archetype=archetype,
                        )
                        log.info(f"  Registered: {class_name} "
                                 f"(archetype={archetype or 'legacy'}, regime={target_regime})")
                except Exception as e:
                    log.warning(f"  Failed to register {filepath}: {e}")

        passed = sum(1 for r in results if r.get("success"))
        log.info(f"Generation complete: {passed}/{len(results)} strategies passed validation")

    except Exception as e:
        log.error(f"Strategy generation failed: {e}", exc_info=True)


def _find_btc_data_file() -> Path | None:
    """Best-effort lookup of a BTC OHLCV feather file for buyhold + regime."""
    candidates = [
        DATA_DIR / "okx" / "BTC_USDT-1h.feather",
        DATA_DIR / "okx" / "futures" / "BTC_USDT_USDT-1h-futures.feather",
        DATA_DIR / "binance" / "BTC_USDT-1h.feather",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def job_backtest_candidates(only_name: str | None = None):
    """Weekly (after generation): backtest all uneval'd candidates.

    For each candidate:
      1. Run full backtest
      2. Run R7 pipeline gates (regime-conditional floor, beat-buy-and-hold,
         optionally walk-forward) against the result
      3. Promote only if all gates pass — otherwise retire with a gate-aware
         verdict so the failure memory captures *which* gate killed it

    FreqAI candidates (spec_type='freqai', issue #47) run through the SAME
    lifecycle with three differences:
      - backtests use the freqtrade-freqai service, the candidate's rendered
        config (configs/freqai/<name>.json), an explicit timerange, and a
        much larger timeout (model training dominates the runtime)
      - walk-forward is MANDATORY: it runs even when R7_WALK_FORWARD is off,
        and a skipped/failed walk-forward blocks promotion even when
        STRICT_PROMOTION_GATES is off — an ML candidate that only looks good
        on one window is the overfitting failure mode this exists to catch
      - model artifacts are purged after evaluation (win or lose)

    `only_name` restricts the run to a single candidate — used by targeted
    manual evaluations (e.g. shaking down one FreqAI candidate) without
    touching the rest of the pool.
    """
    log.info("=== Job: Backtest candidates ===")
    try:
        sys.path.insert(0, str(BASE_DIR / "scripts"))
        from strategy_registry import (
            get_candidates, record_backtest, promote_strategy, retire_strategy,
            get_active_strategies_with_trade_paths,
        )
        from backtest_runner import run_backtest
        from pipeline_gates import (
            compute_regime_fractions, compute_btc_buyhold,
            gate_regime_conditional_floor, gate_beat_buyhold,
            gate_walk_forward, run_walk_forward, gate_correlation,
        )
        from trade_attribution import (
            build_macro_snapshots, load_trades_from_zip,
            attribute_trades, summarize_attribution,
        )

        candidates = get_candidates()
        if only_name:
            candidates = [c for c in candidates if c["name"] == only_name]
        if not candidates:
            log.info("No candidates to backtest.")
            return

        # R7: precompute reference data once per job run (shared across all
        # candidates), not once per candidate. BTC feather + regime fractions
        # only depend on the lookback window, not the strategy.
        btc_path = _find_btc_data_file()
        regime_fractions = None
        if btc_path:
            try:
                import pandas as pd
                btc_df = pd.read_feather(btc_path)
                regime_fractions = compute_regime_fractions(btc_df, lookback_days=180)
                log.info(f"  Regime fractions (180d): {regime_fractions}")
            except Exception as e:
                log.warning(f"  Regime fraction computation failed: {e}")

        # Walk-forward is expensive (N x backtest), but it is part of the
        # production promotion story. With STRICT_PROMOTION_GATES=true, a
        # disabled walk-forward gate records SKIP_WF; strict mode treats that
        # as "not enough evidence" and blocks promotion. Keep R7_WALK_FORWARD
        # enabled when strict gates are enabled, or explicitly turn strict mode
        # off for exploratory runs where skipped gates should be non-blocking.
        enable_wf = os.environ.get("R7_WALK_FORWARD", "").lower() in ("1", "true", "yes")
        wf_splits = int(os.environ.get("R7_WF_SPLITS", "3"))
        wf_days = int(os.environ.get("R7_WF_DAYS", "60"))

        # R2d: macro snapshot dataframe is the same for every candidate in
        # this job run — build it once, share across the loop.
        try:
            macro_df = build_macro_snapshots()
        except Exception as e:
            log.warning(f"  macro snapshot build failed; attribution will be skipped: {e}")
            macro_df = None

        # Phase 6 can produce up to 20 strategies per generation (10
        # archetypes × 20-cell coherence matrix). 25 covers the matrix
        # output plus a small buffer.
        #
        # get_candidates() returns FIFO (oldest first). If a run produces
        # more than 25 ACCEPTED candidates, the overflow waits in the
        # pool and gets evaluated the FOLLOWING Sunday — never skipped
        # outright. This bounds worst-case latency to ~2 weeks rather
        # than letting older candidates starve forever.
        for cand in candidates[:25]:
            name = cand["name"]
            target_regime = cand.get("target_regime", "all")
            is_freqai = (cand.get("spec_type") or "rule") == "freqai"
            log.info(f"  Backtesting: {name} (regime={target_regime}"
                     f"{', freqai' if is_freqai else ''})")

            # FreqAI candidates carry their backtest wiring in the artifacts
            # the renderer wrote: per-candidate config + spec sidecar (model
            # family). Shared kwargs so the full backtest and every
            # walk-forward window run identically.
            freqai_kwargs = {}
            if is_freqai:
                from freqai_spec import container_config_path, load_spec_sidecar
                sidecar = load_spec_sidecar(cand.get("filepath", "")) or {}
                freqai_kwargs = {
                    "config_path": container_config_path(name),
                    "freqai_model": sidecar.get("model", {}).get(
                        "family", "LightGBMRegressor"),
                    "timeout_seconds": int(
                        os.environ.get("FREQAI_BACKTEST_TIMEOUT", "5400")),
                }

            try:
                if is_freqai:
                    # FreqAI requires an explicit timerange — mirror the
                    # 6-month window run_backtest defaults to for rule
                    # candidates when given all available data.
                    bt_end = datetime.now(timezone.utc)
                    bt_start = bt_end - timedelta(days=180)
                    result = run_backtest(
                        strategy_name=name,
                        timerange=(f"{bt_start.strftime('%Y%m%d')}-"
                                   f"{bt_end.strftime('%Y%m%d')}"),
                        use_sandbox=True,
                        export_trades=True,  # R2d
                        **freqai_kwargs,
                    )
                else:
                    result = run_backtest(
                        strategy_name=name,
                        use_sandbox=True,
                        timeout_seconds=600,
                        export_trades=True,  # R2d
                    )

                if not result.get("success"):
                    err = result.get("error", "unknown")
                    log.warning(f"  {name}: backtest failed — {err}")
                    retire_strategy(
                        cand["id"],
                        reason=f"Backtest failed: {err}",
                        verdict="FAIL_BACKTEST",
                    )
                    continue

                # R2d: per-trade attribution. Best-effort — if anything goes
                # wrong (missing export, malformed trades, etc.) we log and
                # carry on with the rest of the pipeline.
                attribution = None
                trades_path = result.get("trades_export_path")
                if trades_path and macro_df is not None and not macro_df.empty:
                    try:
                        trades = load_trades_from_zip(trades_path, name)
                        attribution = attribute_trades(trades, macro_df)
                        if attribution and attribution["total_trades"] > 0:
                            log.info(f"  {name} [attribution]\n{summarize_attribution(attribution)}")
                    except Exception as e:
                        log.warning(f"  {name}: attribution failed — {e}")

                # Record results before running gates so we always have the
                # full-backtest row even if a gate later fails.
                record_backtest(cand["id"], result, attribution=attribution)

                total_trades = result.get("total_trades", 0)
                profit_pct = result.get("profit_total_pct", 0)
                sharpe = result.get("sharpe", 0)
                log.info(f"  {name}: {total_trades} trades, {profit_pct}% profit, Sharpe={sharpe}")

                # R7 gates. Every expected gate produces a verdict in
                # gate_verdicts even when the data it needs is missing.
                # In strict mode (default), skipped verdicts count as fail:
                # promotion requires evidence from every gate, not an implicit
                # pass because a gate was disabled or missing data. This means
                # STRICT_PROMOTION_GATES=true should normally be paired with
                # R7_WALK_FORWARD=true, otherwise SKIP_WF blocks otherwise
                # profitable candidates.
                from pipeline_gates import _skip, _fail, is_strict_pass
                strict_mode = os.environ.get(
                    "STRICT_PROMOTION_GATES", "true"
                ).lower() not in ("0", "false", "no")
                gate_verdicts = []

                if regime_fractions is not None:
                    v = gate_regime_conditional_floor(
                        result, target_regime, regime_fractions, base_min_trades=20,
                    )
                else:
                    v = _skip("SKIP_REGIME", "regime_fractions not available — "
                              "btc data file missing or regime detector failed")
                gate_verdicts.append(v)
                log.info(f"  {name} [regime]: {v['verdict']} — {v['reason']}")

                if btc_path:
                    bh = compute_btc_buyhold(btc_path, timerange=result.get("timerange"))
                    v = gate_beat_buyhold(result, bh)
                else:
                    v = _skip("SKIP_BH", "btc_path not available — "
                              "buyhold reference cannot be computed")
                gate_verdicts.append(v)
                log.info(f"  {name} [buyhold]: {v['verdict']} — {v['reason']}")

                # Walk-forward is opt-in for rule candidates but MANDATORY
                # for freqai ones — overfitting to a single window is the
                # canonical ML failure mode (issue #47).
                if enable_wf or is_freqai:
                    log.info(f"  {name} [walk-forward]: running {wf_splits} windows × {wf_days}d…")
                    wf_results = run_walk_forward(
                        name,
                        backtest_fn=lambda n, tr: run_backtest(
                            strategy_name=n, timerange=tr,
                            use_sandbox=True,
                            **(freqai_kwargs or {"timeout_seconds": 600}),
                        ),
                        n_splits=wf_splits, days_per_split=wf_days,
                    )
                    v = gate_walk_forward(wf_results)
                else:
                    v = _skip("SKIP_WF", "walk-forward disabled "
                              "(set R7_WALK_FORWARD=true to enable)")
                wf_verdict = v
                gate_verdicts.append(v)
                log.info(f"  {name} [walk-forward]: {v['verdict']} — {v['reason']}")

                # Promotion = baseline profitability AND every gate passes
                # (strictly, in strict_mode).
                baseline_ok = total_trades >= 20 and profit_pct > 0 and sharpe > 0
                if strict_mode:
                    all_gates_passed = all(is_strict_pass(v) for v in gate_verdicts)
                else:
                    all_gates_passed = all(v["passed"] for v in gate_verdicts)

                # FreqAI promotion requires REAL walk-forward evidence even
                # in non-strict mode — a skipped WF verdict carries
                # passed=True for legacy aggregation, which must never be
                # enough to promote an ML candidate (issue #47).
                if is_freqai and not is_strict_pass(wf_verdict):
                    all_gates_passed = False

                # R7.4: correlation gate is expensive (reads zip files per
                # active strategy) — only run it when the candidate is
                # otherwise promotion-bound. Cheap-rejects skip this step.
                if baseline_ok and all_gates_passed and trades_path:
                    try:
                        cand_trades = load_trades_from_zip(trades_path, name)
                        active_peers = get_active_strategies_with_trade_paths()
                        v = gate_correlation(cand_trades, active_peers)
                    except Exception as e:
                        log.warning(f"  {name}: correlation gate errored — {e}")
                        # Previously silently logged and continued; that turned
                        # an exception into an implicit pass. Now we record an
                        # explicit fail so the promotion path doesn't ride on
                        # a swallowed error.
                        v = _fail("FAIL_CORR_ERROR",
                                  f"correlation gate raised {type(e).__name__}: {e}")
                    gate_verdicts.append(v)
                    log.info(f"  {name} [correlation]: {v['verdict']} — {v['reason']}")
                    if strict_mode:
                        all_gates_passed = is_strict_pass(v)
                    else:
                        all_gates_passed = v["passed"]

                if baseline_ok and all_gates_passed:
                    promote_strategy(cand["id"])
                    log.info(f"  {name}: AUTO-PROMOTED (baseline + {len(gate_verdicts)} gates)")
                elif not baseline_ok and total_trades < 5:
                    retire_strategy(
                        cand["id"],
                        reason=f"Too few trades: {total_trades} (profit={profit_pct}%, sharpe={sharpe})",
                        verdict="FAIL_TOO_FEW",
                    )
                    log.info(f"  {name}: RETIRED (too few trades)")
                elif not baseline_ok:
                    retire_strategy(
                        cand["id"],
                        reason=f"Unprofitable: {total_trades} trades, {profit_pct}% profit, sharpe={sharpe}",
                        verdict="FAIL_UNPROFITABLE",
                    )
                    log.info(f"  {name}: RETIRED (unprofitable)")
                else:
                    # Profitable on baseline but a gate blocked. Tag with the
                    # first failing gate's verdict so the failure memory has
                    # a specific reason ("FAIL_BH" reads better than a vague
                    # FAIL_UNPROFITABLE for a strategy that did make money
                    # but lost to HODL).
                    failed = next((v for v in gate_verdicts if not v["passed"]), None)
                    if failed is None and is_freqai and not is_strict_pass(wf_verdict):
                        # The only blocker was the mandatory-WF rule: the WF
                        # verdict was a skip (passed=True), so the generic
                        # first-failing-gate scan finds nothing. Name the ML
                        # failure explicitly for the failure memory.
                        failed = {
                            "verdict": "FAIL_ML_NO_WALKFORWARD",
                            "reason": ("freqai candidates require a real "
                                       f"walk-forward pass; got: {wf_verdict['reason']}"),
                        }
                    verdict = failed["verdict"] if failed else "FAIL_GATES"
                    reason = failed["reason"] if failed else "blocked by gates"
                    retire_strategy(cand["id"], reason=reason, verdict=verdict)
                    log.info(f"  {name}: RETIRED ({verdict}: {reason})")

            except Exception as e:
                log.warning(f"  {name}: error — {e}")
            finally:
                if is_freqai:
                    # Win or lose, drop the candidate's model artifacts —
                    # a full backtest + walk-forward leaves O(100MB) per
                    # candidate under user_data/models/.
                    try:
                        from freqai_spec import purge_model_artifacts
                        purge_model_artifacts(name)
                    except Exception as e:
                        log.warning(f"  {name}: model artifact purge failed — {e}")

    except Exception as e:
        log.error(f"Candidate backtesting failed: {e}", exc_info=True)


def job_hyperopt_candidates():
    """Weekly (after backtest): try to rescue marginal failures with hyperopt.

    Picks up the top-N most-promising recently-retired candidates
    (FAIL_TOO_FEW or FAIL_UNPROFITABLE only — crashes are skipped) and runs
    Freqtrade's hyperopt to search the param space the LLM already declared
    via IntParameter/DecimalParameter. Hyperopt auto-writes
    <StrategyFile>.json next to the .py file with the winning params, so the
    re-backtest below picks them up transparently.

    Outcomes:
      HYPEROPT_PROMOTE   re-backtest met PROMOTE criteria → status='active'
      HYPEROPT_NO_EDGE   hyperopt ran but re-backtest still failed → stays retired
                         (failure_verdict updated so we don't retry forever)
      HYPEROPT_FAILED    hyperopt subprocess itself errored (timeout, OOM, etc.)
    """
    log.info("=== Job: Hyperopt rescue ===")
    try:
        sys.path.insert(0, str(BASE_DIR / "scripts"))
        from strategy_registry import get_hyperopt_candidates, mark_hyperopt_outcome
        from backtest_runner import run_hyperopt, run_backtest

        # Bound the per-cycle compute. 3 × ~3-10 min/hyperopt = ~10-30 min total.
        candidates = get_hyperopt_candidates(limit=3)
        if not candidates:
            log.info("No marginal-failure candidates to hyperopt.")
            return

        for cand in candidates:
            name = cand["name"]
            log.info(
                f"  Hyperopt rescue: {name} (had {cand['total_trades']} trades, "
                f"profit={cand['profit_total_pct']}%, prior verdict={cand['failure_verdict']})"
            )

            hopt = run_hyperopt(name, epochs=15, timeout_seconds=1800)
            if not hopt.get("success"):
                err = hopt.get("error", "unknown")
                log.warning(f"  {name}: hyperopt itself failed — {err}")
                mark_hyperopt_outcome(
                    cand["id"], verdict="HYPEROPT_FAILED",
                    reason=f"Hyperopt subprocess failed: {err}",
                )
                continue

            # Hyperopt wrote <StrategyFile>.json — re-backtest picks it up
            log.info(f"  {name}: hyperopt found {hopt.get('total_trades', 0)} trades, "
                     f"{hopt.get('profit_total_pct', 0)}% profit in best epoch. Re-backtesting...")

            bt = run_backtest(name, use_sandbox=True, timeout_seconds=600)
            if not bt.get("success"):
                err = bt.get("error", "unknown")
                log.warning(f"  {name}: re-backtest failed — {err}")
                mark_hyperopt_outcome(
                    cand["id"], verdict="HYPEROPT_NO_EDGE",
                    reason=f"Re-backtest failed after hyperopt: {err}",
                )
                continue

            total_trades = bt.get("total_trades", 0)
            profit_pct = bt.get("profit_total_pct", 0)
            sharpe = bt.get("sharpe", 0)
            log.info(f"  {name}: post-hyperopt re-backtest: {total_trades} trades, "
                     f"{profit_pct}% profit, Sharpe={sharpe}")

            # Same PROMOTE criteria as job_backtest_candidates
            if total_trades >= 20 and profit_pct > 0 and sharpe > 0:
                mark_hyperopt_outcome(
                    cand["id"], verdict="HYPEROPT_PROMOTE",
                    reason=f"Rescued: {total_trades} trades, {profit_pct}% profit, sharpe={sharpe}",
                    promote=True,
                )
            else:
                mark_hyperopt_outcome(
                    cand["id"], verdict="HYPEROPT_NO_EDGE",
                    reason=f"Still failing: {total_trades} trades, {profit_pct}% profit, sharpe={sharpe}",
                )

    except Exception as e:
        log.error(f"Hyperopt rescue job failed: {e}", exc_info=True)


def job_reflector():
    """Weekly: LLM reviews recent trades and proposes improvements.

    Reads trade logs from all instances, analyzes wins/losses against
    regime labels, and suggests regime-to-strategy mapping changes.
    """
    log.info("=== Job: Reflector agent ===")

    try:
        # llm_client handles provider selection + API key check. We still
        # gate on at least ONE provider being configured so we don't burn
        # the rest of this job on a definite failure.
        if not (os.environ.get("DEEPSEEK_API_KEY")
                or os.environ.get("ANTHROPIC_API_KEY")):
            log.warning("No LLM API key set (DEEPSEEK_API_KEY or ANTHROPIC_API_KEY). Skipping reflector.")
            return

        sys.path.insert(0, str(BASE_DIR / "scripts"))
        from llm_client import chat_completion

        # Collect trade data from all instances
        trade_summary = []
        for name, cfg in INSTANCES.items():
            client = FreqtradeClient(cfg["url"], cfg["username"], cfg["password"])
            profit = client.get_profit()
            status = client.get_status()

            instance_info = {
                "instance": name,
                "strategy": cfg["strategy"],
                "active_regimes": cfg["regimes"],
            }
            if profit:
                instance_info["profit_all"] = profit.get("profit_all_coin", 0)
                instance_info["trade_count"] = profit.get("trade_count", 0)
            if status:
                instance_info["open_trades"] = len(status)

            trade_summary.append(instance_info)

        # Get regime history
        regime_state = load_regime_state()
        risk_state = load_risk_state()

        # Get registry stats + recent attribution evidence
        sys.path.insert(0, str(BASE_DIR / "scripts"))
        from strategy_registry import (
            get_registry_stats, get_active_strategies, get_recent_attributions,
        )
        from trade_attribution import format_attributions_for_reflector
        stats = get_registry_stats()
        active = get_active_strategies()
        attributions = get_recent_attributions(n=10, min_trades=10)
        attribution_section = format_attributions_for_reflector(attributions)
        log.info(f"  Reflector context: {len(attributions)} strategies with attribution")

        # Build prompt
        prompt = f"""You are a trading system reflector. Review the following weekly trading data
and provide actionable insights.

CURRENT REGIME: {regime_state.get('regime', 'unknown')} (confidence: {regime_state.get('confidence', 0)})
RISK STATE: total_pnl={risk_state.get('total_pnl', 0)}, kill_switch={risk_state.get('kill_switch_active', False)}

INSTANCE PERFORMANCE:
{json.dumps(trade_summary, indent=2)}

REGISTRY STATS: {json.dumps(stats)}
ACTIVE STRATEGIES: {json.dumps([s['name'] for s in active])}

{attribution_section}

Provide:
1. PERFORMANCE SUMMARY: One paragraph on how the system performed this week.
2. REGIME ACCURACY: Was the regime classification correct? Did strategies match?
3. ATTRIBUTION PATTERNS: From the per-strategy attribution above, what macro
   conditions consistently favor wins or losses across the pool? Call out
   buckets that appear in 2+ strategies' top-lift lists — those are the most
   reliable signals for the next round of strategy generation.
4. RECOMMENDATIONS: 2-3 specific, actionable suggestions. Examples:
   - "Generate more ranging strategies — current ranging strategy underperforms"
   - "Tighten stoploss on momentum strategy — large drawdowns on trend reversals"
   - "Add fgi_fear macro_confidence filter to next batch — 3/4 strategies show
     +0.05 to +0.10 lift in that bucket"
5. RISK FLAGS: Any concerns about drawdown, exposure, or system health.

Be specific. Reference actual numbers from the data above. The attribution
section is the strongest signal you have — use it."""

        # The reflector is a small, summarization-style task — provider
        # default model is fine. Falls back automatically if primary fails.
        # Note: 2048 instead of 1024 because reasoning-capable models
        # (DeepSeek V4 Pro) consume tokens for internal reasoning_content
        # before emitting the visible answer — a tight budget gets eaten
        # by the preamble and returns an empty reply.
        reflection = chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
        )

        # Save reflection to file
        reflections_dir = BASE_DIR / "data" / "reflections"
        reflections_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        reflection_file = reflections_dir / f"reflection-{timestamp}.md"
        reflection_file.write_text(f"# Weekly Reflection — {timestamp}\n\n{reflection}\n")

        log.info(f"Reflection saved to: {reflection_file}")
        log.info(f"Reflection preview: {reflection[:200]}...")

    except Exception as e:
        log.error(f"Reflector failed: {e}", exc_info=True)


def job_llm_regime_override():
    """Daily (after indicator classification): LLM layer for regime.

    Reads the indicator-based regime, macro data, and recent news
    to potentially override or adjust the regime classification.
    """
    log.info("=== Job: LLM regime override ===")

    if not (os.environ.get("DEEPSEEK_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")):
        log.info("No LLM API key set. Using indicator-only regime.")
        return

    try:
        sys.path.insert(0, str(BASE_DIR / "scripts"))
        from llm_client import chat_completion

        state = load_regime_state()
        indicator_regime = state.get("regime", "ranging")
        confidence = state.get("confidence", 0.5)

        # Load macro data
        macro_data = {}
        for pair_name in ["VIX/USDT", "GOLD/USDT", "SPX/USDT", "DXY/USDT"]:
            filename = pair_name.replace("/", "_")
            filepath = DATA_DIR / "binance" / f"{filename}-1d.json"
            if filepath.exists():
                with open(filepath) as f:
                    data = json.load(f)
                if data:
                    last = data[-1]
                    macro_data[pair_name.split("/")[0]] = {
                        "close": last[4],
                        "change_1d": round((last[4] - data[-2][4]) / data[-2][4] * 100, 2) if len(data) > 1 else 0,
                    }

        prompt = f"""You are a crypto market regime classifier. Based on the data below,
determine if the indicator-based regime classification is correct or should be overridden.

INDICATOR REGIME: {indicator_regime} (confidence: {confidence})
ADX: {state.get('adx', 'N/A')}
Volatility percentile: {state.get('vol_pct', 'N/A')}

MACRO DATA:
{json.dumps(macro_data, indent=2)}

REGIMES: trending, ranging, breakout, crisis

Respond with EXACTLY one line in this format:
REGIME: <regime> CONFIDENCE: <0.0-1.0> REASON: <one sentence>

Only override if you have strong reason. The indicator regime is usually correct."""

        # 512 instead of 100: reasoning-capable models (DeepSeek V4 Pro)
        # spend 30-80 tokens on internal reasoning_content before producing
        # the one-line REGIME: ... response. 100 tokens leaves nothing for
        # the visible output.
        text = chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
        ).strip()
        log.info(f"LLM regime response: {text}")

        # Parse response
        if "REGIME:" in text and "CONFIDENCE:" in text:
            parts = text.split("CONFIDENCE:")
            regime_part = parts[0].replace("REGIME:", "").strip().lower()
            conf_part = parts[1].split("REASON:")[0].strip()

            valid_regimes = {"trending", "ranging", "breakout", "crisis"}
            if regime_part in valid_regimes:
                llm_confidence = float(conf_part)

                # Only override if LLM confidence is higher than indicator
                if llm_confidence > confidence and regime_part != indicator_regime:
                    log.info(f"LLM OVERRIDE: {indicator_regime} -> {regime_part} (conf: {llm_confidence})")
                    save_regime_state({
                        "regime": regime_part,
                        "confidence": llm_confidence,
                        "source": "llm",
                        "indicator_regime": indicator_regime,
                        "llm_reason": text.split("REASON:")[-1].strip() if "REASON:" in text else "",
                    })
                else:
                    log.info(f"LLM agrees with indicator regime: {indicator_regime}")
                    state["source"] = "indicator+llm"
                    save_regime_state(state)

    except Exception as e:
        log.error(f"LLM regime override failed: {e}", exc_info=True)
        log.info("Falling back to indicator-only regime.")


# ---------------------------------------------------------------------------
# Error Handler
# ---------------------------------------------------------------------------
def on_job_error(event):
    log.error(f"Job {event.job_id} failed: {event.exception}", exc_info=event.exception)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log.info("=" * 60)
    log.info(f"Strategy Factory Orchestrator starting (mode={MODE})")
    log.info("=" * 60)

    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_listener(on_job_error, EVENT_JOB_ERROR)

    # --- Daily jobs (chained: fetch -> classify -> LLM override -> apply) ---
    # Run at 00:05 UTC daily (after daily candle close)
    scheduler.add_job(job_fetch_macro_data, "cron", hour=0, minute=5, id="fetch_macro")
    scheduler.add_job(job_classify_regime, "cron", hour=0, minute=10, id="classify_regime")
    scheduler.add_job(job_llm_regime_override, "cron", hour=0, minute=12, id="llm_regime")
    scheduler.add_job(job_apply_regime, "cron", hour=0, minute=15, id="apply_regime")

    # --- Weekly jobs: Strategy Factory Loop ---
    # OHLCV refresh fires 30 min before generation so both Saturday
    # mini-backtests and Sunday full backtests use fresh OKX data.
    scheduler.add_job(job_fetch_ohlcv, "cron", day_of_week="sat", hour=19, minute=30, id="fetch_ohlcv")
    # Phase 6: generation moved Sun 02:00 → Sat 20:00 UTC to accommodate the
    # 20-cell coherence matrix (≈3-4 hr generation + 20-30 min backtest at
    # ~3 min/DeepSeek-call). Sunday 02:00 left only ~2 hours before daily
    # macro fetch and risked timeline collision. Saturday 20:00 → Sunday 09:00
    # gives a 13-hour runway.
    scheduler.add_job(job_generate_strategies, "cron", day_of_week="sat", hour=20, minute=0, id="generate_strategies")
    # Backtest fires at Sun 00:30 — gives generation up to 4.5 hours
    scheduler.add_job(job_backtest_candidates, "cron", day_of_week="sun", hour=0, minute=30, id="backtest_candidates")
    # Reflector at Sun 09:00 — backtests should be done by then
    scheduler.add_job(job_reflector, "cron", day_of_week="sun", hour=9, minute=0, id="reflector")
    # Hyperopt last — slowest stage, can run while we sleep
    scheduler.add_job(job_hyperopt_candidates, "cron", day_of_week="sun", hour=12, minute=0, id="hyperopt_candidates")

    # --- Risk monitoring (every 5 minutes) ---
    scheduler.add_job(job_check_risk, "interval", minutes=5, id="check_risk")

    # --- Health check (every 2 minutes) ---
    scheduler.add_job(job_health_check, "interval", minutes=2, id="health_check")

    # --- Deployment reconciler (every 5 minutes, observe-only in Phase 2) ---
    # Logs desired vs running diff and records each tick to
    # deployment_drift_log. RECONCILER_ACTING is plumbed through but the
    # job ignores it until Phase 3 — flipping the flag today has no
    # effect on live containers. See docs/deployment-lifecycle.md.
    scheduler.add_job(job_reconcile_deployments, "interval", minutes=5,
                      id="reconcile_deployments")

    # --- Wait for instances to be ready ---
    log.info("Waiting 15s for Freqtrade instances to start...")
    time.sleep(15)

    # --- Run initial jobs on startup ---
    log.info("Running initial regime classification...")
    try:
        job_fetch_macro_data()
        job_classify_regime()
        job_apply_regime()
        job_health_check()
    except Exception as e:
        log.error(f"Initial job run failed: {e}", exc_info=True)

    log.info("Scheduler started. Jobs registered:")
    for job in scheduler.get_jobs():
        log.info(f"  - {job.id}: {job.trigger}")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("Orchestrator shutting down.")
        scheduler.shutdown()


if __name__ == "__main__":
    main()
