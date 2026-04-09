"""
BaseGeneratedStrategy — Constrained template for LLM-generated strategies.

All strategies produced by the Strategy Generator MUST extend this class.
It enforces:
  - A standard interface compatible with Freqtrade
  - Required metadata (thesis, target_regime, generation_id)
  - Default risk settings that can be overridden within safe bounds

The Strategy Generator's LLM prompt tells the model to extend this class
and implement populate_indicators, populate_entry_trend, populate_exit_trend.

IMPORTANT: This file defines the contract. The validation pipeline checks
generated code against this contract before allowing execution.
"""

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, BooleanParameter
from pandas import DataFrame
import pandas_ta as ta
import numpy as np


class BaseGeneratedStrategy(IStrategy):
    """
    Base class for all LLM-generated strategies.

    Subclasses MUST define:
      - STRATEGY_THESIS: str — one-line description of why this strategy works
      - TARGET_REGIME: str — one of "trending", "ranging", "breakout", "all"
      - GENERATION_ID: str — unique ID from the generator (e.g., "gen-20260408-001")

    Subclasses MUST implement:
      - populate_indicators(self, dataframe, metadata) -> DataFrame
      - populate_entry_trend(self, dataframe, metadata) -> DataFrame
      - populate_exit_trend(self, dataframe, metadata) -> DataFrame
    """

    INTERFACE_VERSION = 3

    # --- Required metadata (subclass MUST override) ---
    STRATEGY_THESIS: str = "UNDEFINED — subclass must set this"
    TARGET_REGIME: str = "all"
    GENERATION_ID: str = "unknown"

    # --- Safe defaults (subclass CAN override within bounds) ---
    minimal_roi = {
        "0": 0.15,
        "60": 0.08,
        "120": 0.04,
        "240": 0.02,
    }

    stoploss = -0.06
    timeframe = '1h'
    startup_candle_count = 250

    # --- Whitelisted imports for generated strategies ---
    # The validation pipeline checks that generated code only imports from this set.
    ALLOWED_IMPORTS = frozenset([
        'freqtrade.strategy',
        'pandas',
        'pandas_ta',
        'numpy',
        'ta',         # ta-lib wrapper
        'talib',      # ta-lib direct
        'math',
    ])

    # --- Whitelisted indicator functions ---
    # Generated strategies can call any pandas_ta function and these numpy functions.
    # This is documentation for the LLM prompt, not enforced at runtime.
    AVAILABLE_INDICATORS = [
        "ta.ema", "ta.sma", "ta.rsi", "ta.macd", "ta.bbands", "ta.adx",
        "ta.atr", "ta.stoch", "ta.willr", "ta.cci", "ta.mfi", "ta.obv",
        "ta.vwap", "ta.alma", "ta.kc", "ta.donchian", "ta.ichimoku",
        "np.where", "np.nan", "DataFrame.rolling", "DataFrame.shift",
        "DataFrame.pct_change", "DataFrame.rank",
    ]
