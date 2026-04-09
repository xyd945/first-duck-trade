"""
Telegram Notifier — sends alerts for critical system events.

Events that trigger notifications:
  - Regime change
  - Kill switch triggered
  - Strategy promoted or retired
  - Weekly factory run summary
  - Instance down

Setup:
  1. Create a Telegram bot via @BotFather, get the token
  2. Get your chat_id (message @userinfobot)
  3. Set in .env: TELEGRAM_TOKEN and TELEGRAM_CHAT_ID
"""

import logging
import os

import requests

log = logging.getLogger("notifier")

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


def _is_configured() -> bool:
    return bool(TELEGRAM_TOKEN and TELEGRAM_CHAT_ID)


def send_telegram(message: str):
    """Send a message via Telegram bot."""
    if not _is_configured():
        log.debug("Telegram not configured. Skipping notification.")
        return

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        resp = requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "Markdown",
        }, timeout=10)

        if resp.status_code != 200:
            log.warning(f"Telegram send failed: {resp.status_code} {resp.text}")
    except Exception as e:
        log.warning(f"Telegram notification failed: {e}")


# ---------------------------------------------------------------------------
# Pre-formatted alert messages
# ---------------------------------------------------------------------------

def notify_regime_change(old_regime: str, new_regime: str, confidence: float, source: str):
    emoji = {"trending": "📈", "ranging": "↔️", "breakout": "🚀", "crisis": "🚨"}.get(new_regime, "❓")
    send_telegram(
        f"{emoji} *Regime Change*\n"
        f"`{old_regime}` → `{new_regime}`\n"
        f"Confidence: {confidence:.0%} | Source: {source}"
    )


def notify_kill_switch(reason: str, total_pnl: float):
    send_telegram(
        f"🔴 *KILL SWITCH TRIGGERED*\n"
        f"Reason: {reason}\n"
        f"Total P&L: {total_pnl:.2f} USDT\n"
        f"All trading STOPPED."
    )


def notify_strategy_promoted(name: str, regime: str, sharpe: float, profit_pct: float):
    send_telegram(
        f"✅ *Strategy Promoted*\n"
        f"Name: `{name}`\n"
        f"Regime: {regime} | Sharpe: {sharpe:.2f} | Profit: {profit_pct:.1f}%"
    )


def notify_factory_summary(generated: int, passed: int, promoted: int, retired: int):
    send_telegram(
        f"🏭 *Weekly Factory Run*\n"
        f"Generated: {generated} | Validated: {passed}\n"
        f"Promoted: {promoted} | Retired: {retired}"
    )


def notify_instance_down(name: str):
    send_telegram(f"⚠️ *Instance Down*: `{name}`")


def notify_reflector_summary(summary: str):
    # Truncate to Telegram's 4096 char limit
    if len(summary) > 3900:
        summary = summary[:3900] + "\n...(truncated)"
    send_telegram(f"📊 *Weekly Reflection*\n{summary}")
