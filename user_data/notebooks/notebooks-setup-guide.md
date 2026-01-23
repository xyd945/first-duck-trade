# Notebooks Setup Guide

This guide explains how to set up a local Python environment to run Jupyter notebooks for indicator testing and visualization.

> [!NOTE]
> Freqtrade runs inside Docker, but for **notebook development** we use a local conda environment for faster iteration and interactive plotting.

---

## Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/) installed
- VS Code with the [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)

---

## Quick Setup

```bash
# Create conda environment with Python 3.12 (required for pandas_ta)
conda create --name freqtrade python=3.12 numpy pandas matplotlib jupyter -y

# Activate the environment
conda activate freqtrade

# Install additional packages via pip
pip install pandas_ta pyarrow
```

---

## Package Reference

| Package | Purpose | Install Method |
|---------|---------|----------------|
| `python>=3.12` | Required by pandas_ta | conda |
| `numpy` | Array operations | conda |
| `pandas` | DataFrame handling | conda |
| `matplotlib` | Static plotting | conda |
| `jupyter` | Run notebooks | conda |
| `pandas_ta` | Technical indicators (ALMA, RSI, etc.) | pip |
| `pyarrow` | Read `.feather` data files | pip |

### Optional Packages

```bash
# For interactive Plotly charts
pip install plotly

# For additional TA indicators
pip install ta-lib
```

---

## Using in VS Code

### 1. Select the Kernel

1. Open any `.ipynb` notebook file
2. Click on **"Select Kernel"** in the top-right corner
3. Choose **"Python Environments..."**
4. Select **`freqtrade (Python 3.12.x)`**

### 2. Verify Setup

Run this cell to verify all packages are installed:

```python
import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt

print("✅ All packages loaded successfully!")
print(f"   pandas: {pd.__version__}")
print(f"   numpy: {np.__version__}")
print(f"   pandas_ta: {ta.version}")
```

---

## Common Issues

### `ModuleNotFoundError: No module named 'X'`

You're using the wrong kernel. Make sure to select the `freqtrade` conda environment in VS Code.

### `ArrowKeyError: pandas.period already defined`

Restart the kernel: `Cmd+Shift+P` → "Jupyter: Restart Kernel"

### `pandas_ta` requires Python 3.12+

Recreate the environment with Python 3.12:

```bash
conda deactivate
conda remove --name freqtrade --all -y
conda create --name freqtrade python=3.12 numpy pandas matplotlib jupyter -y
conda activate freqtrade
pip install pandas_ta pyarrow
```

---

## Environment Management

```bash
# List all conda environments
conda info --envs

# Activate the freqtrade environment
conda activate freqtrade

# Deactivate current environment
conda deactivate

# Remove environment completely
conda remove --name freqtrade --all -y

# Export environment to file (for sharing)
conda env export > environment.yml

# Create environment from file
conda env create -f environment.yml
```

---

## Available Notebooks

| Notebook | Description |
|----------|-------------|
| `plot_whale_liquidity.ipynb` | Visualize the whale liquidity indicator |
| `strategy_analysis_example.ipynb` | Example strategy analysis |

---

## Data Location

OHLCV data files are stored in:
```
user_data/data/okx/futures/
├── BTC_USDT_USDT-1h-futures.feather
├── ETH_USDT_USDT-1h-futures.feather
├── SOL_USDT_USDT-1h-futures.feather
└── ...
```

To load data in a notebook:

```python
import pandas as pd
from pathlib import Path

data_path = Path("../data/okx/futures/BTC_USDT_USDT-1h-futures.feather")
df = pd.read_feather(data_path)
print(f"Loaded {len(df)} candles")
```
