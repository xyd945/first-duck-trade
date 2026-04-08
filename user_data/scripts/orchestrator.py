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
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.events import EVENT_JOB_ERROR

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # user_data/
DATA_DIR = BASE_DIR / "data"
REGIME_STATE_FILE = BASE_DIR / "data" / "regime_state.json"
RISK_STATE_FILE = BASE_DIR / "data" / "risk_state.json"

# Freqtrade instance endpoints (container names resolve via Docker networking)
INSTANCES = {
    "sweep": {
        "url": "http://ft-sweep:8080",
        "username": "freqtrader",
        "password": "CHANGE_ME_sweep_password",
        "strategy": "LiquiditySweepStrategy",
        "regimes": ["ranging"],  # Active in these regimes
    },
    "momentum": {
        "url": "http://ft-momentum:8080",
        "username": "freqtrader",
        "password": "CHANGE_ME_momentum_password",
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
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(BASE_DIR / "logs" / "orchestrator.log"),
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
                data={"username": self._username, "password": self._password},
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
    """Daily: fetch VIX, Gold, DXY, SPX from Yahoo Finance."""
    log.info("=== Job: Fetch macro data ===")
    try:
        # Import and run the existing fetch script
        script = BASE_DIR / "scripts" / "fetch_extra_data.py"
        subprocess.run([sys.executable, str(script)], check=True, timeout=120)
        log.info("Macro data fetch completed.")
    except Exception as e:
        log.error(f"Macro data fetch failed: {e}")


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

    # --- Daily jobs (chained: fetch -> classify -> apply) ---
    # Run at 00:05 UTC daily (after daily candle close)
    scheduler.add_job(job_fetch_macro_data, "cron", hour=0, minute=5, id="fetch_macro")
    scheduler.add_job(job_classify_regime, "cron", hour=0, minute=10, id="classify_regime")
    scheduler.add_job(job_apply_regime, "cron", hour=0, minute=15, id="apply_regime")

    # --- Risk monitoring (every 5 minutes) ---
    scheduler.add_job(job_check_risk, "interval", minutes=5, id="check_risk")

    # --- Health check (every 2 minutes) ---
    scheduler.add_job(job_health_check, "interval", minutes=2, id="health_check")

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
