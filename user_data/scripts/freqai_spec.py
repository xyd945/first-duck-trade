"""
FreqAI candidate specs — validation, rendering, registration (issue #47).

The FreqAI path mirrors the rule-based factory's safety posture: candidates
are DECLARATIVE specs, never free-form ML code. A spec picks features from
the whitelisted library (indicators/freqai_features.py), a target horizon,
a model family from the whitelist, bounded training params, and entry/exit
thresholds. Rendering produces:

  1. A strategy .py in user_data/strategies/candidates/ — a thin subclass
     of BaseFreqaiStrategy containing ONLY declarations (no computation).
  2. A per-candidate backtest config in user_data/configs/freqai/ — the
     committed config-freqai-base.json with the spec's freqai block merged
     in and a unique model identifier (the class name) so model artifacts
     never collide between candidates.
  3. A spec sidecar (<Strategy>.freqai.json) next to the .py so the
     orchestrator can recover model family / spec metadata at backtest
     time, and the failure memory can show WHAT experiment failed.

Spec shape (all fields required unless noted):

  {
    "spec_type": "freqai",
    "name": "FreqaiMomentumLgbm",          # Python class name
    "thesis": "...",
    "target_regime": "trending",            # trending|ranging|breakout|all
    "features": ["rsi", "funding", ...],   # >= 3 keys from the library
    "target": {"type": "future_return", "horizon_candles": 24},
    "model": {"family": "LightGBMRegressor",
               "params": {"n_estimators": 400, ...}},   # params optional
    "thresholds": {"entry": 0.005, "exit": 0.0},
    "freqai": {"train_period_days": 60,     # optional block, bounded
                "backtest_period_days": 7,
                "include_shifted_candles": 2,
                "indicator_periods_candles": [14, 50],
                "test_size": 0.25},
    "risk": {"stoploss": -0.06,
              "minimal_roi": {"0": 0.15, ...}},
    "generation_id": "..."                  # optional, stamped by caller
  }

CLI:
  python freqai_spec.py validate <spec.json>
  python freqai_spec.py register <spec.json>   # render + register candidate
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("freqai_spec")

BASE_DIR = Path(__file__).resolve().parent.parent  # user_data/
CANDIDATES_DIR = BASE_DIR / "strategies" / "candidates"
FREQAI_CONFIG_DIR = BASE_DIR / "configs" / "freqai"
BASE_CONFIG_PATH = BASE_DIR / "configs" / "config-freqai-base.json"

SPEC_TYPE = "freqai"

# Model families the renderer may emit. Extend deliberately, one at a time —
# each family is a different --freqaimodel and a different params surface.
MODEL_FAMILIES = ("LightGBMRegressor",)

# Bounded LightGBM params the spec may set. (name -> (type, min, max))
MODEL_PARAM_BOUNDS = {
    "n_estimators": (int, 50, 1000),
    "learning_rate": (float, 0.005, 0.3),
    "max_depth": (int, 2, 12),
    "num_leaves": (int, 8, 256),
    "min_child_samples": (int, 5, 200),
    "subsample": (float, 0.5, 1.0),
    "colsample_bytree": (float, 0.5, 1.0),
    "reg_alpha": (float, 0.0, 10.0),
    "reg_lambda": (float, 0.0, 10.0),
}

FREQAI_PARAM_BOUNDS = {
    "train_period_days": (int, 30, 180),
    "backtest_period_days": (int, 2, 30),
    "include_shifted_candles": (int, 0, 5),
    "test_size": (float, 0.1, 0.4),
}

VALID_REGIMES = ("trending", "ranging", "breakout", "all")

# Entry gates (issue #47): hard market-state preconditions on buying.
# Types live in the feature library (indicators/freqai_features.GATE_TYPES);
# bounds for their parameters live here with the other spec bounds.
EMA_GATE_PERIOD_BOUNDS = (50, 400)     # hours, on the 1h timeframe
DI_THRESHOLD_BOUNDS = (0.5, 5.0)       # FreqAI Dissimilarity Index cutoff

REQUIRED_FIELDS = (
    "spec_type", "name", "thesis", "target_regime",
    "features", "target", "model", "thresholds", "risk",
)

MIN_FEATURES = 3
HORIZON_BOUNDS = (4, 72)           # candles; 4h..3d on the 1h timeframe
INDICATOR_PERIOD_BOUNDS = (5, 100)
ENTRY_THRESHOLD_BOUNDS = (0.0005, 0.10)


class FreqaiSpecError(ValueError):
    """Raised when a FreqAI spec fails validation."""


def _require(cond: bool, msg: str):
    if not cond:
        raise FreqaiSpecError(msg)


def _check_bounded(params: dict, bounds: dict, where: str):
    for key, value in params.items():
        _require(key in bounds, f"{where}: unknown param {key!r} "
                 f"(allowed: {sorted(bounds)})")
        typ, lo, hi = bounds[key]
        _require(isinstance(value, (int, float)) and not isinstance(value, bool),
                 f"{where}.{key}: must be a number")
        if typ is int:
            _require(float(value).is_integer(), f"{where}.{key}: must be an integer")
        _require(lo <= value <= hi,
                 f"{where}.{key}: {value} outside allowed range [{lo}, {hi}]")


def validate_freqai_spec(spec: dict) -> None:
    """Strict validation; raises FreqaiSpecError on the first violation."""
    from indicators.freqai_features import ALL_FEATURE_KEYS

    _require(isinstance(spec, dict), "spec must be a JSON object")
    missing = [f for f in REQUIRED_FIELDS if f not in spec]
    _require(not missing, f"missing required fields: {missing}")
    _require(spec["spec_type"] == SPEC_TYPE,
             f"spec_type must be {SPEC_TYPE!r}, got {spec['spec_type']!r}")

    import re
    _require(bool(re.match(r"^[A-Z][A-Za-z0-9_]*$", spec["name"])),
             f"name {spec['name']!r} must be a PascalCase Python identifier")
    _require(isinstance(spec["thesis"], str) and spec["thesis"].strip(),
             "thesis must be a non-empty string")
    _require(spec["target_regime"] in VALID_REGIMES,
             f"target_regime must be one of {VALID_REGIMES}")

    # generation_id reaches the rendered file — treat it like `name`, not
    # like free text. Optional, but when present it must be a plain token
    # (the renderer additionally json.dumps-es it as defense in depth).
    gen_id = spec.get("generation_id", "manual")
    _require(isinstance(gen_id, str)
             and bool(re.match(r"^[A-Za-z0-9._-]{1,64}$", gen_id)),
             "generation_id must match ^[A-Za-z0-9._-]{1,64}$")

    # Features: whitelisted keys only, no duplicates, enough signal to learn
    features = spec["features"]
    _require(isinstance(features, list) and len(features) >= MIN_FEATURES,
             f"features must be a list of >= {MIN_FEATURES} keys")
    unknown = [f for f in features if f not in ALL_FEATURE_KEYS]
    _require(not unknown, f"unknown feature keys: {unknown} "
             f"(allowed: {sorted(ALL_FEATURE_KEYS)})")
    _require(len(set(features)) == len(features), "duplicate feature keys")

    # Target: fixed-horizon future return is the only supported type
    target = spec["target"]
    _require(isinstance(target, dict) and target.get("type") == "future_return",
             "target.type must be 'future_return'")
    horizon = target.get("horizon_candles")
    _require(isinstance(horizon, int) and not isinstance(horizon, bool)
             and HORIZON_BOUNDS[0] <= horizon <= HORIZON_BOUNDS[1],
             f"target.horizon_candles must be an int in {HORIZON_BOUNDS}")

    # Model: whitelisted family, bounded params
    model = spec["model"]
    _require(isinstance(model, dict) and model.get("family") in MODEL_FAMILIES,
             f"model.family must be one of {MODEL_FAMILIES}")
    model_params = model.get("params") or {}
    _require(isinstance(model_params, dict), "model.params must be an object")
    _check_bounded(model_params, MODEL_PARAM_BOUNDS, "model.params")

    # Thresholds: entry strictly above exit, both sane
    thresholds = spec["thresholds"]
    _require(isinstance(thresholds, dict), "thresholds must be an object")
    entry = thresholds.get("entry")
    exit_ = thresholds.get("exit")
    _require(isinstance(entry, (int, float)) and
             ENTRY_THRESHOLD_BOUNDS[0] <= entry <= ENTRY_THRESHOLD_BOUNDS[1],
             f"thresholds.entry must be a number in {ENTRY_THRESHOLD_BOUNDS}")
    _require(isinstance(exit_, (int, float)) and -0.05 <= exit_ < entry,
             "thresholds.exit must be a number in [-0.05, entry)")

    # Optional freqai block: bounded scalars + indicator periods list
    freqai = spec.get("freqai", {})
    _require(isinstance(freqai, dict), "freqai must be an object")
    periods = freqai.get("indicator_periods_candles")
    scalar_params = {k: v for k, v in freqai.items()
                     if k != "indicator_periods_candles"}
    _check_bounded(scalar_params, FREQAI_PARAM_BOUNDS, "freqai")
    if periods is not None:
        _require(isinstance(periods, list) and 1 <= len(periods) <= 4
                 and all(isinstance(p, int) and not isinstance(p, bool)
                         and INDICATOR_PERIOD_BOUNDS[0] <= p <= INDICATOR_PERIOD_BOUNDS[1]
                         for p in periods),
                 f"freqai.indicator_periods_candles must be 1-4 ints "
                 f"in {INDICATOR_PERIOD_BOUNDS}")

    # Entry gate: optional; each type has exactly its own params
    gate = spec.get("entry_gate", {"type": "none"})
    _require(isinstance(gate, dict), "entry_gate must be an object")
    from indicators.freqai_features import GATE_TYPES
    gate_type = gate.get("type")
    _require(gate_type in GATE_TYPES,
             f"entry_gate.type must be one of {GATE_TYPES}")
    allowed_keys = {"type"}
    if gate_type == "ema_trend":
        allowed_keys.add("period")
        p = gate.get("period")
        _require(isinstance(p, int) and not isinstance(p, bool)
                 and EMA_GATE_PERIOD_BOUNDS[0] <= p <= EMA_GATE_PERIOD_BOUNDS[1],
                 f"entry_gate.period must be an int in {EMA_GATE_PERIOD_BOUNDS}")
    if gate_type == "di_confidence":
        allowed_keys.add("di_threshold")
        d = gate.get("di_threshold")
        _require(isinstance(d, (int, float)) and not isinstance(d, bool)
                 and DI_THRESHOLD_BOUNDS[0] <= d <= DI_THRESHOLD_BOUNDS[1],
                 f"entry_gate.di_threshold must be a number in {DI_THRESHOLD_BOUNDS}")
    stray = set(gate) - allowed_keys
    _require(not stray, f"entry_gate: unexpected keys {sorted(stray)} "
             f"for type {gate_type!r}")

    # Risk: same contract as rule-based specs
    risk = spec["risk"]
    _require(isinstance(risk, dict), "risk must be an object")
    stoploss = risk.get("stoploss")
    _require(isinstance(stoploss, (int, float)) and not isinstance(stoploss, bool)
             and -0.5 <= stoploss < 0,
             "risk.stoploss must be a negative number >= -0.5")
    roi = risk.get("minimal_roi")
    _require(isinstance(roi, dict) and roi,
             "risk.minimal_roi must be a non-empty dict")
    # ROI contents render into Python source via json.dumps — a bool/null
    # value would emit `true`/`null` (NameError at import), and freqtrade
    # needs minute-string keys anyway. Reject anything but digits -> numbers.
    for k, v in roi.items():
        _require(isinstance(k, str) and k.isdigit(),
                 f"minimal_roi key {k!r} must be a string of digits (minutes)")
        _require(isinstance(v, (int, float)) and not isinstance(v, bool)
                 and -1.0 <= v <= 10.0,
                 f"minimal_roi[{k!r}] must be a number in [-1, 10]")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_TEMPLATE = '''"""
{name} — FreqAI candidate rendered from a validated spec (issue #47).

AUTO-GENERATED by freqai_spec.render_freqai_strategy — DO NOT EDIT.
All executable logic lives in BaseFreqaiStrategy; this file is declarations
only. The matching backtest config is configs/freqai/{name}.json.
"""

from base_freqai import BaseFreqaiStrategy


class {name}(BaseFreqaiStrategy):
    STRATEGY_THESIS = {thesis}
    STRATEGY_ARCHETYPE = "ml_regressor"
    TARGET_REGIME = "{target_regime}"
    GENERATION_ID = {generation_id}

    FREQAI_FEATURES = {features}

    ENTRY_THRESHOLD = {entry_threshold}
    EXIT_THRESHOLD = {exit_threshold}
    ENTRY_GATE_TYPE = "{gate_type}"
    ENTRY_GATE_PERIOD = {gate_period}

    stoploss = {stoploss}
    minimal_roi = {minimal_roi}
'''


def render_freqai_strategy(spec: dict) -> str:
    """Render the declarative strategy subclass. Validates first."""
    validate_freqai_spec(spec)
    gate = spec.get("entry_gate", {"type": "none"})
    return _TEMPLATE.format(
        gate_type=gate["type"],
        gate_period=int(gate.get("period", 0)),
        name=spec["name"],
        thesis=json.dumps(spec["thesis"]),
        target_regime=spec["target_regime"],
        # json.dumps as defense in depth — the validator already restricts
        # generation_id to a plain token, but a template must never trust
        # a spec string enough to place it raw inside quotes.
        generation_id=json.dumps(spec.get("generation_id", "manual")),
        features=json.dumps(spec["features"]),
        entry_threshold=float(spec["thresholds"]["entry"]),
        exit_threshold=float(spec["thresholds"]["exit"]),
        stoploss=float(spec["risk"]["stoploss"]),
        minimal_roi=json.dumps(spec["risk"]["minimal_roi"]),
    )


def render_freqai_config(spec: dict, base_config_path: Path | str = None) -> dict:
    """Merge the spec's ML settings into the committed base config.

    Returns the config dict. The identifier is the class name — unique per
    candidate, so FreqAI's model artifacts (user_data/models/<identifier>/)
    never collide and can be purged per-candidate after evaluation.
    """
    validate_freqai_spec(spec)
    base_path = Path(base_config_path or BASE_CONFIG_PATH)
    config = json.loads(base_path.read_text())

    freqai = config["freqai"]
    spec_freqai = spec.get("freqai", {})

    freqai["identifier"] = spec["name"]
    freqai["train_period_days"] = int(
        spec_freqai.get("train_period_days", freqai["train_period_days"])
    )
    freqai["backtest_period_days"] = int(
        spec_freqai.get("backtest_period_days", freqai["backtest_period_days"])
    )

    fp = freqai["feature_parameters"]
    fp["label_period_candles"] = int(spec["target"]["horizon_candles"])
    fp["include_shifted_candles"] = int(
        spec_freqai.get("include_shifted_candles", fp["include_shifted_candles"])
    )
    if spec_freqai.get("indicator_periods_candles"):
        fp["indicator_periods_candles"] = [
            int(p) for p in spec_freqai["indicator_periods_candles"]
        ]

    if "test_size" in spec_freqai:
        freqai["data_split_parameters"]["test_size"] = float(spec_freqai["test_size"])

    # di_confidence gate: FreqAI-native — a DI_threshold > 0 makes FreqAI
    # flag predictions on data unlike the training distribution as
    # do_predict=0, which the base strategy's entry condition requires.
    gate = spec.get("entry_gate", {})
    if gate.get("type") == "di_confidence":
        fp["DI_threshold"] = float(gate["di_threshold"])

    # Model params: start from base defaults, overlay the spec's bounded set.
    freqai["model_training_parameters"].update(spec["model"].get("params", {}))

    return config


# ---------------------------------------------------------------------------
# Materialization + registration
# ---------------------------------------------------------------------------

def spec_sidecar_path(strategy_filepath: Path | str) -> Path:
    """<candidates>/<Name>.py -> <candidates>/<Name>.freqai.json"""
    p = Path(strategy_filepath)
    return p.with_suffix("").with_suffix(".freqai.json") if p.suffix == ".py" \
        else p.with_name(p.name + ".freqai.json")


def container_config_path(name: str) -> str:
    """Path of a rendered per-candidate config as seen from INSIDE the
    freqtrade containers (user_data is bind-mounted at /freqtrade)."""
    return f"/freqtrade/user_data/configs/freqai/{name}.json"


def load_spec_sidecar(strategy_filepath: Path | str) -> dict | None:
    """Best-effort read of a candidate's spec sidecar. None if absent/bad."""
    p = spec_sidecar_path(strategy_filepath)
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def materialize_freqai_candidate(spec: dict) -> dict:
    """Validate + write the three artifacts (strategy .py, config, sidecar).

    Returns {"name", "filepath", "config_path", "sidecar_path"} with HOST
    paths. Does not touch the registry — see register_freqai_candidate.
    """
    validate_freqai_spec(spec)
    name = spec["name"]

    code = render_freqai_strategy(spec)
    from validation_pipeline import validate_freqai_strategy_file

    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    FREQAI_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # Freqtrade's resolver only puts the candidate's OWN directory on
    # sys.path while importing it (iresolver PathModifier), so the base
    # class must exist inside candidates/ — same reason base_generated.py
    # is duplicated there. Refresh the copy on every materialization so
    # candidates never run against a stale base.
    base_src = BASE_DIR / "strategies" / "base_freqai.py"
    base_dst = CANDIDATES_DIR / "base_freqai.py"
    if base_src.exists() and (
        not base_dst.exists() or base_dst.read_text() != base_src.read_text()
    ):
        base_dst.write_text(base_src.read_text())

    filepath = CANDIDATES_DIR / f"{name}.py"
    filepath.write_text(code)

    # Renderer output must satisfy the freqai validator — a template bug
    # (or a hostile thesis string breaking out of its literal) should kill
    # the candidate here, not at backtest time.
    result = validate_freqai_strategy_file(filepath)
    if not result.passed:
        filepath.unlink(missing_ok=True)
        raise FreqaiSpecError(f"rendered strategy failed validation: {result}")

    config = render_freqai_config(spec)
    config_path = FREQAI_CONFIG_DIR / f"{name}.json"
    config_path.write_text(json.dumps(config, indent=2))

    sidecar = spec_sidecar_path(filepath)
    sidecar.write_text(json.dumps(spec, indent=2))

    return {
        "name": name,
        "filepath": str(filepath),
        "config_path": str(config_path),
        "sidecar_path": str(sidecar),
    }


def purge_model_artifacts(identifier: str) -> bool:
    """Delete user_data/models/<identifier>/ after a candidate is evaluated.

    FreqAI writes one model per sliding window per pair; a single full
    backtest leaves hundreds of MB behind. The identifier is the candidate
    class name (see render_freqai_config), so this only ever touches that
    candidate's artifacts. Returns True if something was removed.
    """
    import shutil
    models_dir = BASE_DIR / "models" / identifier
    if not models_dir.is_dir():
        return False
    shutil.rmtree(models_dir, ignore_errors=True)
    log.info(f"Purged FreqAI model artifacts: {models_dir}")
    return True


def register_freqai_candidate(spec: dict) -> int:
    """Materialize the spec and register it in the strategy registry with
    spec_type='freqai'. Returns the strategy id."""
    artifacts = materialize_freqai_candidate(spec)
    from strategy_registry import init_db, register_strategy

    init_db()
    strategy_id = register_strategy(
        name=artifacts["name"],
        filepath=artifacts["filepath"],
        thesis=spec["thesis"],
        target_regime=spec["target_regime"],
        generation_id=spec.get("generation_id", "manual"),
        archetype="ml_regressor",
        spec_type=SPEC_TYPE,
    )
    log.info(f"Registered FreqAI candidate {artifacts['name']} (id={strategy_id})")
    return strategy_id


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    if len(sys.argv) != 3 or sys.argv[1] not in ("validate", "register"):
        print("usage: freqai_spec.py {validate|register} <spec.json>",
              file=sys.stderr)
        sys.exit(2)

    action, spec_path = sys.argv[1], Path(sys.argv[2])
    spec = json.loads(spec_path.read_text())

    if action == "validate":
        validate_freqai_spec(spec)
        print(f"OK: {spec['name']} is a valid freqai spec")
    else:
        sid = register_freqai_candidate(spec)
        print(f"registered {spec['name']} as candidate id={sid}")
