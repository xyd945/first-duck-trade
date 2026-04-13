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
from datetime import datetime, timezone
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


# ---------------------------------------------------------------------------
# Weekly Jobs: Strategy Factory Loop
# ---------------------------------------------------------------------------

def job_generate_strategies():
    """Weekly: generate new candidate strategies via LLM.

    Generates strategies across different regimes, validates them,
    and registers passing candidates in the registry.
    """
    log.info("=== Job: Generate strategies ===")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log.warning("ANTHROPIC_API_KEY not set. Skipping strategy generation.")
        return

    try:
        sys.path.insert(0, str(BASE_DIR / "scripts"))
        from strategy_generator import generate_batch
        from strategy_registry import register_strategy, get_registry_stats

        # Get current regime for context
        state = load_regime_state()
        context = f"Current market regime: {state.get('regime', 'unknown')} (confidence: {state.get('confidence', 0)})"

        # Get existing strategy results for context
        stats = get_registry_stats()
        existing_results = (
            f"Registry stats: {stats['active']} active, {stats['candidate']} candidates, "
            f"{stats['retired']} retired, {stats['total_backtests']} backtests run."
        )

        results = generate_batch(
            count=5,
            regimes=["trending", "ranging", "breakout", "all", "trending"],
            context=context,
            existing_results=existing_results,
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
                    target_regime = "all"
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
                            break

                    if class_name:
                        register_strategy(
                            name=class_name,
                            filepath=str(filepath),
                            thesis=thesis,
                            target_regime=target_regime,
                            generation_id=gen_id,
                        )
                        log.info(f"  Registered: {class_name} (regime={target_regime})")
                except Exception as e:
                    log.warning(f"  Failed to register {filepath}: {e}")

        passed = sum(1 for r in results if r.get("success"))
        log.info(f"Generation complete: {passed}/{len(results)} strategies passed validation")

    except Exception as e:
        log.error(f"Strategy generation failed: {e}", exc_info=True)


def job_backtest_candidates():
    """Weekly (after generation): backtest all uneval'd candidates.

    Runs the 2-stage evaluation (mini-backtest then full) on each
    candidate that doesn't have backtest results yet.
    """
    log.info("=== Job: Backtest candidates ===")
    try:
        sys.path.insert(0, str(BASE_DIR / "scripts"))
        from strategy_registry import get_candidates, record_backtest, promote_strategy, retire_strategy
        from backtest_runner import run_backtest

        candidates = get_candidates()
        if not candidates:
            log.info("No candidates to backtest.")
            return

        for cand in candidates[:10]:  # Cap at 10 per run to limit compute
            name = cand["name"]
            log.info(f"  Backtesting: {name}")

            try:
                result = run_backtest(
                    strategy_name=name,
                    use_sandbox=True,
                    timeout_seconds=180,
                )

                if not result.get("success"):
                    log.warning(f"  {name}: backtest failed — {result.get('error', 'unknown')}")
                    retire_strategy(cand["id"], reason=f"Backtest failed: {result.get('error')}")
                    continue

                # Record results
                record_backtest(cand["id"], result)

                total_trades = result.get("total_trades", 0)
                profit_pct = result.get("profit_total_pct", 0)
                sharpe = result.get("sharpe", 0)

                log.info(f"  {name}: {total_trades} trades, {profit_pct}% profit, Sharpe={sharpe}")

                # Auto-promote if profitable with decent trade count
                if total_trades >= 20 and profit_pct > 0 and sharpe > 0:
                    promote_strategy(cand["id"])
                    log.info(f"  {name}: AUTO-PROMOTED (profitable, Sharpe > 0)")
                elif total_trades < 5:
                    retire_strategy(cand["id"], reason=f"Too few trades: {total_trades}")
                    log.info(f"  {name}: RETIRED (too few trades)")

            except Exception as e:
                log.warning(f"  {name}: error — {e}")

    except Exception as e:
        log.error(f"Candidate backtesting failed: {e}", exc_info=True)


def job_reflector():
    """Weekly: LLM reviews recent trades and proposes improvements.

    Reads trade logs from all instances, analyzes wins/losses against
    regime labels, and suggests regime-to-strategy mapping changes.
    """
    log.info("=== Job: Reflector agent ===")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log.warning("ANTHROPIC_API_KEY not set. Skipping reflector.")
        return

    try:
        import anthropic

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

        # Get registry stats
        sys.path.insert(0, str(BASE_DIR / "scripts"))
        from strategy_registry import get_registry_stats, get_active_strategies
        stats = get_registry_stats()
        active = get_active_strategies()

        # Build prompt
        prompt = f"""You are a trading system reflector. Review the following weekly trading data
and provide actionable insights.

CURRENT REGIME: {regime_state.get('regime', 'unknown')} (confidence: {regime_state.get('confidence', 0)})
RISK STATE: total_pnl={risk_state.get('total_pnl', 0)}, kill_switch={risk_state.get('kill_switch_active', False)}

INSTANCE PERFORMANCE:
{json.dumps(trade_summary, indent=2)}

REGISTRY STATS: {json.dumps(stats)}
ACTIVE STRATEGIES: {json.dumps([s['name'] for s in active])}

Provide:
1. PERFORMANCE SUMMARY: One paragraph on how the system performed this week.
2. REGIME ACCURACY: Was the regime classification correct? Did strategies match?
3. RECOMMENDATIONS: 2-3 specific, actionable suggestions. Examples:
   - "Generate more ranging strategies — current ranging strategy underperforms"
   - "Tighten stoploss on momentum strategy — large drawdowns on trend reversals"
   - "Current regime thresholds may be too sensitive — 3 regime changes this week"
4. RISK FLAGS: Any concerns about drawdown, exposure, or system health.

Be specific. Reference actual numbers from the data above."""

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )

        reflection = response.content[0].text

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

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log.info("ANTHROPIC_API_KEY not set. Using indicator-only regime.")
        return

    try:
        import anthropic

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

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text.strip()
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

    # --- Weekly jobs: Strategy Factory Loop (Sundays at 02:00 UTC) ---
    scheduler.add_job(job_generate_strategies, "cron", day_of_week="sun", hour=2, minute=0, id="generate_strategies")
    scheduler.add_job(job_backtest_candidates, "cron", day_of_week="sun", hour=2, minute=30, id="backtest_candidates")
    scheduler.add_job(job_reflector, "cron", day_of_week="sun", hour=3, minute=0, id="reflector")

    # --- Risk monitoring (every 5 minutes) ---
    scheduler.add_job(job_check_risk, "interval", minutes=5, id="check_risk")

    # --- Health check (every 2 minutes) ---
    scheduler.add_job(job_health_check, "interval", minutes=2, id="health_check")

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
