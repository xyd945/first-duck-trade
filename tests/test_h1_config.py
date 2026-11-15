"""The committed paper config is keyless, inactive and tied to G0 defaults."""
import json
from pathlib import Path


def test_h1_dryrun_config_is_inactive_and_matches_g0():
    root = Path(__file__).resolve().parents[1] / "user_data/configs"
    cfg = json.loads((root / "h1_dryrun.json").read_text())
    policy = json.loads((root / "research-h1.example.json").read_text())
    assert cfg["dry_run"] is True
    assert cfg["initial_state"] == "stopped"
    assert cfg["force_entry_enable"] is False
    assert cfg["api_server"]["enabled"] is cfg["telegram"]["enabled"] is False
    assert cfg["exchange"]["name"] == "okx"
    assert all(cfg["exchange"][key] == "" for key in ("key", "secret", "password"))
    assert set(cfg["exchange"]["pair_whitelist"]) == set(policy["h1"]["eligibility"])
    assert cfg["research_h1"] == policy["h1"]
    assert cfg["startup_candle_count"] == cfg["research_h1"]["paper_startup"] == 400
    assert cfg["dry_run_wallet"] == cfg["research_h1"]["initial_wallet"] == 10000
    assert (cfg["timeframe"], cfg["stake_currency"], cfg["stake_amount"], cfg["max_open_trades"]) == ("1d", "USDT", "unlimited", 4)
    assert cfg["research_params"] == {**policy["params"], "cost_rate": 0.0015}
    assert cfg["research_costs"] == {**policy["costs"], "stress_multiplier": 1}
    assert cfg["fee"] == cfg["research_params"]["cost_rate"]
    assert "placeholder, not selected" in cfg["h1_activation_status"]
    assert cfg["research_h1"]["paper_mapping_status"] == "pending Yudi's OK"


def test_h1_dryrun_strict_engine_schema():
    import pytest
    pytest.importorskip("freqtrade")
    from freqtrade.configuration.config_validation import validate_config_schema
    from freqtrade.enums import RunMode
    path = Path(__file__).resolve().parents[1] / "user_data/configs/h1_dryrun.json"
    cfg = json.loads(path.read_text())
    cfg["runmode"] = RunMode.DRY_RUN
    validate_config_schema(cfg)
