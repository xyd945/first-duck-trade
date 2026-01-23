"""
Custom Indicators Package for Freqtrade.

This package contains reusable indicator logic converted from TradingView Pine Script.
Import these in your strategies via:

    from indicators.whale_liquidity import add_whale_liquidity

Usage in strategy:
    dataframe = add_whale_liquidity(dataframe, smooth_len=40, spike_threshold=3.0)
"""

from .whale_liquidity import add_whale_liquidity

__all__ = [
    'add_whale_liquidity',
]
