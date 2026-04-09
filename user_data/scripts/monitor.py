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

INSTANCES = {
    "ft-sweep": {"url": "http://localhost:8081", "strategy": "LiquiditySweepStrategy", "regimes": ["ranging"]},
    "ft-momentum": {"url": "http://localhost:8082", "strategy": "MomentumTrendStrategy", "regimes": ["trending", "breakout"]},
}


def get_system_status() -> dict:
    """Collect full system status."""
    status = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "regime": {},
        "risk": {},
        "instances": {},
        "registry": {},
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

    # Recent logs
    if LOG_FILE.exists():
        try:
            lines = LOG_FILE.read_text().strip().split("\n")
            status["recent_logs"] = lines[-30:]
        except Exception:
            pass

    return status


DASHBOARD_HTML = r"""<!DOCTYPE html>
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

  let instancesHtml = '';
  for (const [name, info] of Object.entries(data.instances)) {
    const statusClass = info.status === 'UP' ? 'status-up' : 'status-down';
    const regimeMatch = data.regime.regime && info.regimes.includes(data.regime.regime);
    const trading = info.status === 'UP' && regimeMatch;
    const tradingLabel = trading ? '<span class="status-up">TRADING</span>' : '<span class="status-paused">PAUSED</span>';
    instancesHtml += '<tr><td>' + name + '</td><td>' + info.strategy + '</td><td class="' + statusClass + '">' + info.status + '</td><td>' + tradingLabel + '</td><td>' + info.regimes.join(', ') + '</td></tr>';
  }

  let registryHtml = '';
  const reg = data.registry;
  if (reg.top_strategies) {
    for (const s of reg.top_strategies) {
      const profitClass = s.profit_pct > 0 ? 'positive' : 'negative';
      registryHtml += '<tr><td>' + s.name + '</td><td>' + s.regime + '</td><td>' + s.status + '</td><td>' + (s.sharpe || 0).toFixed(2) + '</td><td class="' + profitClass + '">' + (s.profit_pct || 0).toFixed(1) + '%</td><td>' + (s.trades || 0) + '</td></tr>';
    }
  }

  let logsHtml = '';
  if (data.recent_logs) {
    logsHtml = data.recent_logs.map(l => {
      let cls = 'log-info';
      if (l.includes('ERROR') || l.includes('CRITICAL')) cls = 'log-error';
      else if (l.includes('WARNING')) cls = 'log-warning';
      return '<span class="' + cls + '">' + l.replace(/</g, '&lt;') + '</span>';
    }).join('\\n');
  }

  d.innerHTML = \`
    <div class="grid">
      <div class="card \${killSwitch ? 'kill-switch' : ''}">
        <div class="card-title">Current Regime</div>
        <div class="value \${regimeClass}">\${regime.toUpperCase()}</div>
        <div style="margin-top:6px;color:#8b949e">Confidence: \${confidence}% &middot; Source: \${source}</div>
      </div>
      <div class="card \${killSwitch ? 'kill-switch' : ''}">
        <div class="card-title">Risk</div>
        <div class="value \${totalPnl >= 0 ? 'positive' : 'negative'}">\${totalPnl >= 0 ? '+' : ''}\${totalPnl.toFixed(2)} USDT</div>
        <div style="margin-top:6px;color:#8b949e">\${killSwitch ? '🔴 KILL SWITCH ACTIVE' : '🟢 Normal'}</div>
      </div>
      <div class="card">
        <div class="card-title">Registry</div>
        <div style="font-size:18px">\${reg.active || 0} active &middot; \${reg.candidate || 0} candidates &middot; \${reg.retired || 0} retired</div>
        <div style="margin-top:6px;color:#8b949e">\${reg.total_backtests || 0} backtests run</div>
      </div>
    </div>

    <h2>Instances</h2>
    <div class="card">
      <table>
        <tr><th>Container</th><th>Strategy</th><th>Status</th><th>Trading</th><th>Active Regimes</th></tr>
        \${instancesHtml}
      </table>
    </div>

    \${registryHtml ? \`
    <h2>Top Strategies (by Sharpe)</h2>
    <div class="card">
      <table>
        <tr><th>Name</th><th>Regime</th><th>Status</th><th>Sharpe</th><th>Profit</th><th>Trades</th></tr>
        \${registryHtml}
      </table>
    </div>\` : ''}

    <h2>Recent Logs</h2>
    <div class="logs">\${logsHtml || 'No logs available'}</div>

    <div class="footer">Auto-refreshes every 30 seconds &middot; \${new Date(data.timestamp).toLocaleString()}</div>
  \`;
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
