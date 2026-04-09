"""
Monitoring Dashboard — Simple HTTP server for system status.

Serves a single-page dashboard at http://localhost:8888 showing:
  - Current regime + confidence + source
  - Instance status (UP/DOWN, trading/paused, P&L)
  - Risk state (drawdown, kill switch)
  - Strategy registry stats
  - Recent orchestrator log entries
  - Last factory run results

Also provides a JSON API at /api/status for programmatic access.
"""

import json
import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import requests

log = logging.getLogger("monitor")

BASE_DIR = Path(__file__).resolve().parent.parent  # user_data/
if not (BASE_DIR / "data").exists() and Path("/app/user_data").exists():
    BASE_DIR = Path("/app/user_data")

REGIME_STATE_FILE = BASE_DIR / "data" / "regime_state.json"
RISK_STATE_FILE = BASE_DIR / "data" / "risk_state.json"
REGISTRY_DB = BASE_DIR / "data" / "strategy_registry.db"
LOG_FILE = BASE_DIR / "logs" / "orchestrator.log"

# When running in Docker, use container names. When running locally, use localhost ports.
_in_docker = Path("/app/user_data").exists()
INSTANCES = {
    "ft-sweep": {
        "url": "http://ft-sweep:8080" if _in_docker else "http://localhost:8081",
        "strategy": "LiquiditySweepStrategy",
        "regimes": ["ranging"],
    },
    "ft-momentum": {
        "url": "http://ft-momentum:8080" if _in_docker else "http://localhost:8082",
        "strategy": "MomentumTrendStrategy",
        "regimes": ["trending", "breakout"],
    },
}


def get_system_status() -> dict:
    """Collect full system status."""
    status = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "regime": {},
        "risk": {},
        "instances": {},
        "registry": {},
        "schedule": {},
        "reflections": [],
        "recent_logs": [],
    }

    # Regime
    if REGIME_STATE_FILE.exists():
        with open(REGIME_STATE_FILE) as f:
            status["regime"] = json.load(f)

    # Risk
    if RISK_STATE_FILE.exists():
        with open(RISK_STATE_FILE) as f:
            status["risk"] = json.load(f)

    # Instances
    for name, cfg in INSTANCES.items():
        inst = {"strategy": cfg["strategy"], "regimes": cfg["regimes"]}
        try:
            resp = requests.get(f"{cfg['url']}/api/v1/ping", timeout=3)
            inst["status"] = "UP" if resp.status_code == 200 else "DOWN"
        except Exception:
            inst["status"] = "DOWN"
        status["instances"][name] = inst

    # Registry
    if REGISTRY_DB.exists():
        try:
            conn = sqlite3.connect(str(REGISTRY_DB))
            for s in ("candidate", "active", "retired"):
                status["registry"][s] = conn.execute(
                    "SELECT COUNT(*) FROM strategies WHERE status = ?", (s,)
                ).fetchone()[0]
            status["registry"]["total_backtests"] = conn.execute(
                "SELECT COUNT(*) FROM backtest_results"
            ).fetchone()[0]

            # Top strategies by Sharpe
            rows = conn.execute("""
                SELECT s.name, s.target_regime, s.status, br.sharpe, br.profit_total_pct, br.total_trades
                FROM strategies s
                JOIN backtest_results br ON s.id = br.strategy_id
                ORDER BY br.sharpe DESC LIMIT 5
            """).fetchall()
            status["registry"]["top_strategies"] = [
                {"name": r[0], "regime": r[1], "status": r[2], "sharpe": r[3], "profit_pct": r[4], "trades": r[5]}
                for r in rows
            ]
            conn.close()
        except Exception as e:
            status["registry"]["error"] = str(e)

    # Macro data — latest values from Yahoo Finance JSON files
    status["macro"] = {}
    for pair_name, label in [("VIX/USDT", "VIX"), ("GOLD/USDT", "Gold"), ("SPX/USDT", "S&P 500"), ("DXY/USDT", "DXY")]:
        filename = pair_name.replace("/", "_")
        filepath = BASE_DIR / "data" / "binance" / f"{filename}-1d.json"
        if filepath.exists():
            try:
                with open(filepath) as f:
                    data_rows = json.load(f)
                if len(data_rows) >= 2:
                    last = data_rows[-1]
                    prev = data_rows[-2]
                    close = last[4]
                    prev_close = prev[4]
                    change_pct = ((close - prev_close) / prev_close * 100) if prev_close else 0
                    ts_ms = last[0]
                    status["macro"][label] = {
                        "close": round(close, 2),
                        "change_pct": round(change_pct, 2),
                        "date": datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d"),
                    }
            except Exception:
                pass

    # Schedule — static definition + last run extraction from logs
    status["schedule"] = {
        "daily": [
            {"time": "00:05 UTC", "job": "Fetch macro data", "id": "fetch_macro", "source": "Yahoo Finance (VIX, Gold, DXY, SPX)"},
            {"time": "00:10 UTC", "job": "Classify regime", "id": "classify_regime", "source": "Indicators (ADX, EMA, volatility, F&G)"},
            {"time": "00:12 UTC", "job": "LLM regime override", "id": "llm_regime", "source": "Claude Haiku + macro data"},
            {"time": "00:15 UTC", "job": "Apply regime", "id": "apply_regime", "source": "Start/stop strategy instances"},
            {"time": "Every 5min", "job": "Risk monitoring", "id": "check_risk", "source": "Drawdown check, -10% kill switch"},
            {"time": "Every 2min", "job": "Health check", "id": "health_check", "source": "Ping all instances"},
        ],
        "weekly": [
            {"time": "Sun 02:00 UTC", "job": "Generate strategies", "id": "generate_strategies", "source": "Claude Sonnet, 5 strategies across regimes"},
            {"time": "Sun 02:30 UTC", "job": "Backtest candidates", "id": "backtest_candidates", "source": "Sandboxed Docker, 2-stage evaluation"},
            {"time": "Sun 03:00 UTC", "job": "Reflector agent", "id": "reflector", "source": "Claude Haiku, weekly trade review"},
        ],
    }

    # Extract last run times from logs for each job
    if LOG_FILE.exists():
        try:
            log_text = LOG_FILE.read_text()
            for group in ("daily", "weekly"):
                for job in status["schedule"][group]:
                    job_id = job["id"]
                    # Find last "=== Job:" line for this job
                    last_run = None
                    last_status = "unknown"
                    for line in log_text.split("\n"):
                        if f"Job: {job['job']}" in line or f'job_{job_id}' in line:
                            # Extract timestamp from log line
                            if line[:19].replace("-", "").replace(":", "").replace(" ", "").isdigit() or line[:10].startswith("202"):
                                last_run = line[:19]
                        if f'job_{job_id}' in line and "executed successfully" in line:
                            last_status = "success"
                            if line[:19].replace("-", "").replace(":", "").replace(" ", "").isdigit() or line[:10].startswith("202"):
                                last_run = line[:19]
                        elif f'job_{job_id}' in line and ("ERROR" in line or "failed" in line.lower()):
                            last_status = "error"
                    job["last_run"] = last_run
                    job["last_status"] = last_status
        except Exception:
            pass

    # Reflections — load recent weekly reflections
    reflections_dir = BASE_DIR / "data" / "reflections"
    if reflections_dir.exists():
        try:
            files = sorted(reflections_dir.glob("reflection-*.md"), reverse=True)[:3]
            for f in files:
                content = f.read_text()
                status["reflections"].append({
                    "filename": f.name,
                    "date": f.name.replace("reflection-", "").replace(".md", "")[:8],
                    "preview": content[:500],
                })
        except Exception:
            pass

    # Recent logs
    if LOG_FILE.exists():
        try:
            lines = LOG_FILE.read_text().strip().split("\n")
            status["recent_logs"] = lines[-30:]
        except Exception:
            pass

    return status


DASHBOARD_HTML = """<!DOCTYPE html>
<html>
<head>
<title>Strategy Factory Dashboard</title>
<meta charset="utf-8">
<meta http-equiv="refresh" content="30">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', monospace;
         background: #0d1117; color: #c9d1d9; padding: 20px; }
  h1 { color: #58a6ff; margin-bottom: 20px; font-size: 22px; }
  h2 { color: #8b949e; margin: 20px 0 10px; font-size: 16px; text-transform: uppercase; letter-spacing: 1px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }
  .card-title { color: #8b949e; font-size: 12px; text-transform: uppercase; margin-bottom: 8px; }
  .value { font-size: 28px; font-weight: bold; }
  .regime-trending { color: #3fb950; }
  .regime-ranging { color: #d29922; }
  .regime-breakout { color: #58a6ff; }
  .regime-crisis { color: #f85149; }
  .status-up { color: #3fb950; }
  .status-down { color: #f85149; }
  .status-paused { color: #d29922; }
  .positive { color: #3fb950; }
  .negative { color: #f85149; }
  table { width: 100%; border-collapse: collapse; margin-top: 8px; }
  th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid #21262d; font-size: 13px; }
  th { color: #8b949e; }
  .logs { background: #0d1117; border: 1px solid #30363d; border-radius: 6px;
          padding: 12px; max-height: 300px; overflow-y: auto; font-size: 11px;
          line-height: 1.6; white-space: pre-wrap; word-break: break-all; }
  .log-error { color: #f85149; }
  .log-warning { color: #d29922; }
  .log-info { color: #8b949e; }
  .kill-switch { background: #f8514922; border-color: #f85149; }
  .footer { margin-top: 20px; color: #484f58; font-size: 11px; }
</style>
</head>
<body>
<h1>First Duck Trade - Strategy Factory</h1>
<div id="dashboard">Loading...</div>
<script>
async function load() {
  const resp = await fetch('/api/status');
  const data = await resp.json();
  const d = document.getElementById('dashboard');

  const regime = data.regime.regime || 'unknown';
  const regimeClass = 'regime-' + regime;
  const confidence = (data.regime.confidence * 100 || 0).toFixed(0);
  const source = data.regime.source || 'unknown';
  const killSwitch = data.risk.kill_switch_active;
  const totalPnl = data.risk.total_pnl || 0;

  // --- Build instances table ---
  let instancesHtml = '';
  for (const [name, info] of Object.entries(data.instances)) {
    const statusClass = info.status === 'UP' ? 'status-up' : 'status-down';
    const regimeMatch = data.regime.regime && info.regimes.includes(data.regime.regime);
    const trading = info.status === 'UP' && regimeMatch;
    const tradingLabel = trading ? '<span class="status-up">TRADING</span>' : '<span class="status-paused">PAUSED</span>';
    instancesHtml += '<tr><td>' + name + '</td><td>' + info.strategy + '</td><td class="' + statusClass + '">' + info.status + '</td><td>' + tradingLabel + '</td><td>' + info.regimes.join(', ') + '</td></tr>';
  }

  // --- Build registry table ---
  let registryHtml = '';
  const reg = data.registry;
  if (reg.top_strategies && reg.top_strategies.length > 0) {
    for (const s of reg.top_strategies) {
      const profitClass = s.profit_pct > 0 ? 'positive' : 'negative';
      registryHtml += '<tr><td>' + s.name + '</td><td>' + s.regime + '</td><td>' + s.status + '</td><td>' + (s.sharpe || 0).toFixed(2) + '</td><td class="' + profitClass + '">' + (s.profit_pct || 0).toFixed(1) + '%</td><td>' + (s.trades || 0) + '</td></tr>';
    }
  }

  // --- Build schedule tables ---
  function buildScheduleRows(jobs) {
    return jobs.map(j => {
      const statusIcon = j.last_status === 'success' ? '&#x2705;' : j.last_status === 'error' ? '&#x274C;' : '&#x2796;';
      const lastRun = j.last_run || 'Never';
      return '<tr><td>' + j.time + '</td><td>' + j.job + '</td><td>' + j.source + '</td><td>' + statusIcon + '</td><td style="color:#8b949e;font-size:11px">' + lastRun + '</td></tr>';
    }).join('');
  }

  // --- Build reflections ---
  let reflectionsHtml = '';
  if (data.reflections && data.reflections.length > 0) {
    reflectionsHtml = data.reflections.map(r => {
      return '<div class="card" style="margin-bottom:8px"><div class="card-title">Reflection ' + r.date + '</div><div style="font-size:12px;line-height:1.5;white-space:pre-wrap">' + r.preview.replace(/</g, '&lt;').replace(/\\n/g, '<br>') + '</div></div>';
    }).join('');
  }

  // --- Build macro data cards ---
  let macroHtml = '';
  const macro = data.macro || {};
  for (const [label, m] of Object.entries(macro)) {
    const changeClass = m.change_pct >= 0 ? 'positive' : 'negative';
    const arrow = m.change_pct >= 0 ? '&uarr;' : '&darr;';
    macroHtml += '<div class="card" style="text-align:center">'
      + '<div class="card-title">' + label + '</div>'
      + '<div style="font-size:22px;font-weight:bold">' + m.close.toLocaleString() + '</div>'
      + '<div class="' + changeClass + '" style="margin-top:4px">' + arrow + ' ' + m.change_pct.toFixed(2) + '%</div>'
      + '<div style="color:#484f58;font-size:10px;margin-top:2px">' + m.date + '</div>'
      + '</div>';
  }

  // --- Build regime detail ---
  const adx = data.regime.adx ? data.regime.adx.toFixed(1) : 'N/A';
  const volPct = data.regime.vol_pct ? data.regime.vol_pct.toFixed(1) : 'N/A';
  const indicatorRegime = data.regime.indicator_regime || '';
  const llmReason = data.regime.llm_reason || '';
  let regimeDetail = 'ADX: ' + adx + ' &middot; Vol %ile: ' + volPct;
  if (indicatorRegime) regimeDetail += '<br>Indicator said: ' + indicatorRegime + ', LLM overrode';
  if (llmReason) regimeDetail += '<br>LLM reason: ' + llmReason;

  // --- Build logs ---
  let logsHtml = '';
  if (data.recent_logs) {
    logsHtml = data.recent_logs.map(l => {
      let cls = 'log-info';
      if (l.includes('ERROR') || l.includes('CRITICAL')) cls = 'log-error';
      else if (l.includes('WARNING')) cls = 'log-warning';
      return '<span class="' + cls + '">' + l.replace(/</g, '&lt;') + '</span>';
    }).join('\\n');
  }

  // --- Render ---
  d.innerHTML = `
    <div class="grid">
      <div class="card ${killSwitch ? 'kill-switch' : ''}">
        <div class="card-title">Current Regime</div>
        <div class="value ${regimeClass}">${regime.toUpperCase()}</div>
        <div style="margin-top:6px;color:#8b949e">Confidence: ${confidence}% &middot; Source: ${source}</div>
        <div style="margin-top:4px;color:#484f58;font-size:11px">${regimeDetail}</div>
      </div>
      <div class="card ${killSwitch ? 'kill-switch' : ''}">
        <div class="card-title">Risk</div>
        <div class="value ${totalPnl >= 0 ? 'positive' : 'negative'}">${totalPnl >= 0 ? '+' : ''}${totalPnl.toFixed(2)} USDT</div>
        <div style="margin-top:6px;color:#8b949e">${killSwitch ? '<span style="color:#f85149">KILL SWITCH ACTIVE</span>' : '<span style="color:#3fb950">Normal</span>'}</div>
      </div>
      <div class="card">
        <div class="card-title">Strategy Registry</div>
        <div style="font-size:18px">${reg.active || 0} active &middot; ${reg.candidate || 0} candidates &middot; ${reg.retired || 0} retired</div>
        <div style="margin-top:6px;color:#8b949e">${reg.total_backtests || 0} backtests run</div>
      </div>
    </div>

    ${macroHtml ? `
    <h2>Macro Data</h2>
    <div class="grid" style="grid-template-columns: repeat(auto-fit, minmax(140px, 1fr))">
      ${macroHtml}
    </div>` : ''}

    <h2>Strategy Instances</h2>
    <div class="card">
      <table>
        <tr><th>Container</th><th>Strategy</th><th>Status</th><th>Trading</th><th>Active Regimes</th></tr>
        ${instancesHtml}
      </table>
    </div>

    <div class="grid" style="margin-top:16px">
      <div>
        <h2>Daily Schedule</h2>
        <div class="card">
          <table>
            <tr><th>Time</th><th>Job</th><th>Source</th><th></th><th>Last Run</th></tr>
            ${buildScheduleRows(data.schedule.daily || [])}
          </table>
        </div>
      </div>
      <div>
        <h2>Weekly Schedule (Sunday)</h2>
        <div class="card">
          <table>
            <tr><th>Time</th><th>Job</th><th>Source</th><th></th><th>Last Run</th></tr>
            ${buildScheduleRows(data.schedule.weekly || [])}
          </table>
        </div>
      </div>
    </div>

    ${registryHtml ? `
    <h2>Top Strategies (by Sharpe)</h2>
    <div class="card">
      <table>
        <tr><th>Name</th><th>Regime</th><th>Status</th><th>Sharpe</th><th>Profit</th><th>Trades</th></tr>
        ${registryHtml}
      </table>
    </div>` : ''}

    ${reflectionsHtml ? `
    <h2>Recent Reflections</h2>
    ${reflectionsHtml}` : ''}

    <h2>Orchestrator Logs</h2>
    <div class="logs">${logsHtml || 'No logs available'}</div>

    <div class="footer">Auto-refreshes every 30 seconds &middot; ${new Date(data.timestamp).toLocaleString()}</div>
  `;
}
load();
</script>
</body>
</html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/status":
            status = get_system_status()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(status, indent=2).encode())
        elif self.path == "/" or self.path == "/dashboard":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress request logging


def main():
    port = int(os.environ.get("MONITOR_PORT", 8888))
    server = HTTPServer(("0.0.0.0", port), DashboardHandler)
    print(f"Dashboard running at http://localhost:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
