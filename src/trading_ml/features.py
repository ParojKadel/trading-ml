from __future__ import annotations

import pandas as pd
import numpy as np


def _add_basic_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Safe features using only current/past bars.
    Assumes df index is timestamp, sorted.
    """
    out = df.copy()

    # log return
    out[f"{prefix}logret_1"] = np.log(out["close"]).diff()

    # candle anatomy
    out[f"{prefix}range"] = (out["high"] - out["low"]).replace(0, np.nan)
    out[f"{prefix}body"] = (out["close"] - out["open"])
    out[f"{prefix}body_pct_range"] = (out[f"{prefix}body"] / out[f"{prefix}range"]).replace([np.inf, -np.inf], np.nan)

    # rolling stats of returns
    for w in (4, 16, 64):  # 1h, 4h, 16h on 15m grid
        out[f"{prefix}ret_mean_{w}"] = out[f"{prefix}logret_1"].rolling(w).mean()
        out[f"{prefix}ret_std_{w}"] = out[f"{prefix}logret_1"].rolling(w).std()

    # volume z-score (rolling)
    for w in (16, 64):
        v = out["volume"]
        out[f"{prefix}vol_z_{w}"] = (v - v.rolling(w).mean()) / v.rolling(w).std()

    return out


def build_multitimeframe_features(df_15m: pd.DataFrame, df_1h: pd.DataFrame) -> pd.DataFrame:
    """
    Base index: 15m bars.
    Adds:
      - 15m features
      - 1h features forward-filled onto 15m timestamps (no future data)
    """
    df_15m = df_15m.sort_index()
    df_1h = df_1h.sort_index()

    f15 = _add_basic_features(df_15m, prefix="m15_")

    f1h = _add_basic_features(df_1h, prefix="h1_")

    # align 1h features to 15m timestamps by forward filling last known 1h bar
    f1h_aligned = f1h.reindex(f15.index, method="ffill")

    # keep only feature columns (not raw OHLCV duplicated)
    drop_raw = ["open", "high", "low", "close", "volume"]
    f15_feat = f15.drop(columns=drop_raw)
    f1h_feat = f1h_aligned.drop(columns=drop_raw)

    X = pd.concat([f15_feat, f1h_feat], axis=1)

    # drop rows with NaNs from rolling windows
    X = X.dropna().astype("float32")
    return X
