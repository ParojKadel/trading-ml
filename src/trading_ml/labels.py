from __future__ import annotations

import numpy as np
import pandas as pd


def make_future_return_label(df_15m: pd.DataFrame, horizon_bars: int = 4, threshold: float = 0.0) -> pd.Series:
    """
    Label:
      y = 1 if future return > threshold
      y = 0 otherwise
    Uses close-to-close return over horizon_bars.
    """
    close = df_15m["close"].astype("float64")
    fut = close.shift(-horizon_bars)
    ret = (fut / close) - 1.0
    y = (ret > threshold).astype("int64")
    y.name = f"y_up_h{horizon_bars}"
    return y
