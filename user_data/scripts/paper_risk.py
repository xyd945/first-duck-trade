"""Deterministic paper-account checks. This module never executes an order."""
from datetime import datetime, timezone

from accounting import number

# No environment flag may enable execution during this research phase.
EXECUTION_ENABLED = False


def fresh(value, now, max_age=300):
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is None or now.tzinfo is None or not 0 <= (now - stamp).total_seconds() <= max_age:
        raise ValueError("Missing, stale or future timestamp")
    return stamp.astimezone(timezone.utc)


def quote(snapshot, pair, now):
    if not isinstance(pair, str) or not pair.endswith("/USDT") or pair.count("/") != 1:
        raise ValueError("Only USDT spot pairs are supported")
    price = snapshot["prices"][pair]
    fresh(price["asof"], now)
    return number(price["price"], "price", positive=True)


def mark_account(snapshot, expected_instances, now):
    fresh(snapshot["asof"], now)
    if snapshot["quote_currency"] != "USDT":
        raise ValueError("Only USDT spot paper accounts are supported")
    cash = 0.
    positions, pending, available_cash, instance_accounts = {}, {}, {}, {}
    accounts, instances = set(), set()
    for account in snapshot["accounts"]:
        if account["mode"] != "spot":
            raise ValueError("Only spot accounts are supported")
        if account["id"] in accounts or not account["id"]:
            raise ValueError("Duplicate or missing account identity")
        accounts.add(account["id"])
        for instance in account["instances"]:
            if instance in instances or not instance:
                raise ValueError("Duplicate or missing instance identity")
            instances.add(instance)
            instance_accounts[instance] = account["id"]
        cash += number(account["cash"], "cash")
        available_cash[account["id"]] = account["cash"]
        if not isinstance(account["positions"], list) or not isinstance(account["pending"], list):
            raise ValueError("Explicit positions and pending orders are required")
        ids = set()
        for position in account["positions"]:
            if position["id"] in ids or position["instance"] not in account["instances"]:
                raise ValueError("Duplicate or unowned position")
            ids.add(position["id"])
            pair = position["pair"]
            if position.get("leverage", 1) != 1 or position.get("is_short", False):
                raise ValueError("Only unlevered long spot positions are supported")
            value = number(position["quantity"], "quantity") * quote(snapshot, pair, now)
            positions[pair] = positions.get(pair, 0.) + value
        ids = set()
        for order in account["pending"]:
            if order["id"] in ids or order["instance"] not in account["instances"]:
                raise ValueError("Duplicate or unowned pending order")
            ids.add(order["id"])
            pair = order["pair"]
            quote(snapshot, pair, now)
            amount = number(order["notional"], "pending notional", positive=True)
            if order["side"] == "buy":
                pending[pair] = pending.get(pair, 0.) + amount
                available_cash[account["id"]] -= amount
            elif order["side"] != "sell":
                raise ValueError("Invalid pending order side")
    if not accounts or not expected_instances or instances != set(expected_instances):
        raise ValueError("Snapshot must cover every expected instance exactly once")
    gross = sum(positions.values())
    equity = number(cash + gross, "equity", positive=True)
    return {"cash": cash, "equity": equity, "gross_exposure": gross, "positions": positions,
            "pending": pending, "available_cash": available_cash, "instance_accounts": instance_accounts}


def evaluate(snapshot, state, expected_instances, now, *, proposals=()):
    result = dict(state) if isinstance(state, dict) else {}
    result["allow_new_exposure"] = False
    try:
        if not isinstance(state, dict):
            raise ValueError("Risk state must be an object")
        # Allow scheduler jitter around the five-minute job, not stale market inputs.
        stamp = fresh(state["asof"], now, max_age=600)
        if (state["day"] != stamp.date().isoformat() or
                not isinstance(state["kill_switch_active"], bool) or not isinstance(state["daily_halt"], bool)):
            raise ValueError("Invalid persisted risk state")
        for key in ("capital", "peak_equity", "day_start_equity", "last_equity"):
            number(state[key], key, positive=True)
        if state["peak_equity"] < max(state["capital"], state["day_start_equity"], state["last_equity"]):
            raise ValueError("Invalid equity high-water mark")

        result.update(mark_account(snapshot, expected_instances, now))
        day = now.astimezone(timezone.utc).date().isoformat()
        rollover = state["day"] != day
        opening = state["last_equity"] if rollover else state["day_start_equity"]
        peak = max(state["peak_equity"], result["equity"])
        daily_loss = max(0., (opening - result["equity"]) / opening)
        drawdown = (peak - result["equity"]) / peak
        daily_halt = (state["daily_halt"] and not rollover) or daily_loss >= .005
        kill = state["kill_switch_active"] or drawdown >= .05
        result.update(day=day, day_start_equity=opening, peak_equity=peak, last_equity=result["equity"],
                      daily_loss=daily_loss, drawdown=drawdown, daily_halt=daily_halt,
                      total_pnl=result["equity"] - state["capital"],
                      kill_switch_active=kill, allow_new_exposure=not (kill or daily_halt),
                      reason="kill_latched" if state["kill_switch_active"] else
                      "drawdown" if kill else "daily_loss" if daily_halt else "ok")
        projected = dict(result["positions"])
        for pair, value in result["pending"].items():
            projected[pair] = projected.get(pair, 0.) + value
        for proposal in proposals:
            account = result["instance_accounts"][proposal["instance"]]
            pair = proposal["pair"]
            quote(snapshot, pair, now)
            amount = number(proposal["notional"], "proposed notional", positive=True)
            projected[pair] = projected.get(pair, 0.) + amount
            result["available_cash"][account] -= amount
        result["projected_gross"] = sum(projected.values())
        reason = None
        if result["projected_gross"] > .05 * result["equity"]:
            reason = "gross_limit"
        elif any(projected[pair] > .01 * result["equity"]
                 for pair in set(result["pending"]) | {p["pair"] for p in proposals}):
            reason = "position_limit"
        elif any(value < 0 for value in result["available_cash"].values()):
            reason = "cash_limit"
        if reason:
            result.update(allow_new_exposure=False, reason=reason)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        result.update(kill_switch_active=True, allow_new_exposure=False, reason=f"Risk input rejected: {exc}")
    result["asof"] = now.isoformat()
    return result
