"""
Custom Indicators Package for Freqtrade.

This package contains reusable indicator logic converted from TradingView Pine Script.
Import these in your strategies via:

    from indicators.whale_liquidity import add_whale_liquidity
    from indicators.chaikin_money_flow import add_chaikin_money_flow

Usage in strategy:
    dataframe = add_whale_liquidity(dataframe, smooth_len=40, spike_threshold=3.0)
    dataframe = add_chaikin_money_flow(dataframe, length=20)
"""

from .whale_liquidity import add_whale_liquidity
from .chaikin_money_flow import add_chaikin_money_flow

__all__ = [
    'add_whale_liquidity',
    'add_chaikin_money_flow',
]
