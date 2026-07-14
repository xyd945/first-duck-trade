"""
BaseFreqaiStrategy — Constrained template for FreqAI (ML) candidates.

Issue #47: FreqAI candidates are rendered from declarative specs, never
free-form ML code. This base class owns ALL executable logic — feature
engineering (delegated to indicators.freqai_features), target definition,
and threshold-based entry/exit. A rendered subclass only declares data:

  - FREQAI_FEATURES: list[str]  — feature keys from the whitelisted library
  - ENTRY_THRESHOLD / EXIT_THRESHOLD — predicted-return cut-offs
  - stoploss / minimal_roi / timeframe — risk numbers from the spec
  - STRATEGY_THESIS / TARGET_REGIME / GENERATION_ID / STRATEGY_ARCHETYPE

The prediction target is the forward return over
freqai.feature_parameters.label_period_candles (set per candidate in the
rendered config). Entries are long-only (spot policy shared with the
rule-based factory): enter when the model predicts a return above
ENTRY_THRESHOLD and FreqAI flags the prediction usable (do_predict == 1);
exit when the prediction drops below EXIT_THRESHOLD or turns unusable.

The one legitimate shift(-N) in the ML path lives in
freqai_features.add_future_return_target — targets must look forward.
Rendered candidate files contain no computation at all, so the standard
look-ahead validator applies to them unchanged.
"""

from freqtrade.strategy import IStrategy
from pandas import DataFrame

from indicators.freqai_features import (
    add_basic_features,
    add_expand_features,
    add_future_return_target,
    add_standard_features,
    entry_gate_mask,
)

TARGET_COLUMN = "&-future_return"


class BaseFreqaiStrategy(IStrategy):
    """Base class for all FreqAI candidate strategies.

    Subclasses (rendered from specs, or the hand-written baseline) override
    the declarative attributes documented in the module docstring. They
    should NOT override the feature/target/entry/exit methods — that would
    reintroduce the free-form-code risk the spec path exists to prevent.
    """

    INTERFACE_VERSION = 3

    # --- Required metadata (subclass MUST override) ---
    STRATEGY_THESIS: str = "UNDEFINED — subclass must set this"
    STRATEGY_ARCHETYPE: str = "ml_regressor"
    TARGET_REGIME: str = "all"
    GENERATION_ID: str = "unknown"

    # --- Declarative ML surface (subclass overrides from its spec) ---
    FREQAI_FEATURES: list = [
        "rsi", "ema_dist", "natr", "roc", "volume_z", "pct_change",
    ]
    ENTRY_THRESHOLD: float = 0.005   # predicted fwd return to enter long
    EXIT_THRESHOLD: float = 0.0      # predicted fwd return to exit

    # Market-state entry gate (issue #47): a hard precondition on BUYING
    # the model cannot override. "none" = today's behavior. "ema_trend"
    # uses ENTRY_GATE_PERIOD; "regime_match" uses TARGET_REGIME;
    # "di_confidence" acts through the rendered config's DI_threshold
    # (surfaces as do_predict=0, already required below). Exits are
    # never gated — a gate closing does not force-sell open positions.
    ENTRY_GATE_TYPE: str = "none"
    ENTRY_GATE_PERIOD: int = 0

    # --- Safe defaults (subclass CAN override within spec bounds) ---
    minimal_roi = {"0": 0.15, "60": 0.08, "120": 0.04, "240": 0.02}
    stoploss = -0.06
    timeframe = "1h"
    can_short = False
    # FreqAI requires processing each new candle exactly once.
    process_only_new_candles = True
    startup_candle_count = 200

    # ------------------------------------------------------------------
    # FreqAI hooks — fixed logic, driven by the declarative attributes
    # ------------------------------------------------------------------

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict, **kwargs
    ) -> DataFrame:
        """Period-parameterized features; FreqAI calls this once per period
        in feature_parameters.indicator_periods_candles and handles the
        column-name expansion itself."""
        return add_expand_features(dataframe, self.FREQAI_FEATURES, period)

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        return add_basic_features(dataframe, self.FREQAI_FEATURES)

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        return add_standard_features(dataframe, self.FREQAI_FEATURES)

    def set_freqai_targets(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        horizon = int(
            self.freqai_info["feature_parameters"]["label_period_candles"]
        )
        return add_future_return_target(dataframe, horizon, TARGET_COLUMN)

    # ------------------------------------------------------------------
    # Standard strategy interface
    # ------------------------------------------------------------------

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Hands the dataframe to FreqAI: train/retrain on the sliding
        # window, then return it with prediction columns attached
        # (TARGET_COLUMN prediction + do_predict flag).
        return self.freqai.start(dataframe, metadata, self)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        gate = entry_gate_mask(
            dataframe, self.ENTRY_GATE_TYPE,
            period=self.ENTRY_GATE_PERIOD,
            target_regime=self.TARGET_REGIME,
        )
        dataframe.loc[
            (dataframe["do_predict"] == 1)
            & (dataframe[TARGET_COLUMN] > self.ENTRY_THRESHOLD)
            & gate,
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["do_predict"] != 1)
            | (dataframe[TARGET_COLUMN] < self.EXIT_THRESHOLD),
            "exit_long",
        ] = 1
        return dataframe
