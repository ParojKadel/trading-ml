from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from .features import build_multitimeframe_features


def simple_backtest(
    df_15m: pd.DataFrame,
    proba_up: pd.Series,
    fee_bps: float = 4.0,       # 4 bps = 0.04% (rough placeholder)
    slippage_bps: float = 2.0,  # 2 bps
    long_th: float = 0.55,
    short_th: float = 0.45,
) -> pd.DataFrame:
    """
    Simple backtest:
      - enter long if proba > long_th
      - enter short if proba < short_th
      - else flat
    PnL uses next bar close-to-close return.
    Fees paid when position changes.
    """
    close = df_15m["close"].reindex(proba_up.index).astype("float64")
    ret1 = close.pct_change().shift(-1)  # next-bar return
    pos = pd.Series(0.0, index=proba_up.index)

    pos[proba_up > long_th] = 1.0
    pos[proba_up < short_th] = -1.0

    # transaction cost when position changes
    turn = pos.diff().abs().fillna(0.0)  # 0,1,2
    cost = (fee_bps + slippage_bps) / 10_000.0
    tc = turn * cost

    strat_ret = pos * ret1 - tc
    equity = (1.0 + strat_ret.fillna(0.0)).cumprod()

    out = pd.DataFrame(
        {
            "proba_up": proba_up,
            "pos": pos,
            "ret_next": ret1,
            "tc": tc,
            "strat_ret": strat_ret,
            "equity": equity,
        }
    ).dropna()

    return out


def main() -> None:
    df_15m = pd.read_parquet("data/raw/binanceusdm_ETHUSDT_15m.parquet").sort_index()
    df_1h = pd.read_parquet("data/raw/binanceusdm_ETHUSDT_1h.parquet").sort_index()

    X = build_multitimeframe_features(df_15m, df_1h)

    model = XGBClassifier()
    model.load_model("models/xgb_baseline.json")

    proba = pd.Series(model.predict_proba(X)[:, 1], index=X.index, name="proba_up")

    # backtest on the last 20% only (rough “out of sample-ish”)
    cut = int(len(proba) * 0.8)
    proba_bt = proba.iloc[cut:]
    df_bt = simple_backtest(df_15m, proba_bt)

    total_ret = df_bt["equity"].iloc[-1] - 1.0
    avg = df_bt["strat_ret"].mean()
    std = df_bt["strat_ret"].std()
    sharpe_like = (avg / std) * np.sqrt(365 * 24 * 4) if std > 0 else np.nan  # 15m bars

    print(f"Backtest bars: {len(df_bt):,}")
    print(f"Total return: {total_ret*100:.2f}%")
    print(f"Sharpe-like (rough): {sharpe_like:.2f}")
    print("Equity tail:")
    print(df_bt[['equity']].tail())


if __name__ == "__main__":
    main()
